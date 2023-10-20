use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{inner_product, powers, product, Field, PrimeField},
        expression::{rotate::Rotatable, CommonPolynomial, Expression, Query},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
        BitIndex, Itertools,
    },
    Error,
};
use std::{collections::BTreeMap, fmt::Debug};

pub mod classic;

#[derive(Clone, Debug)]
pub struct VirtualPolynomial<'a, F> {
    expression: &'a Expression<F>,
    polys: Vec<&'a MultilinearPolynomial<F>>,
    challenges: &'a [F],
    ys: &'a [Vec<F>],
}

impl<'a, F: PrimeField> VirtualPolynomial<'a, F> {
    pub fn new(
        expression: &'a Expression<F>,
        polys: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
        challenges: &'a [F],
        ys: &'a [Vec<F>],
    ) -> Self {
        Self {
            expression,
            polys: polys.into_iter().collect(),
            challenges,
            ys,
        }
    }
}

pub trait SumCheck<F: Field>: Clone + Debug {
    type ProverParam: Clone + Debug;
    type VerifierParam: Clone + Debug;

    #[allow(clippy::type_complexity)]
    fn prove(
        pp: &Self::ProverParam,
        num_vars: usize,
        virtual_poly: VirtualPolynomial<F>,
        sum: F,
        transcript: &mut impl FieldTranscriptWrite<F>,
    ) -> Result<(F, Vec<F>, BTreeMap<Query, F>), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        num_vars: usize,
        degree: usize,
        sum: F,
        transcript: &mut impl FieldTranscriptRead<F>,
    ) -> Result<(F, Vec<F>), Error>;
}

pub fn evaluate<F: PrimeField, R: Rotatable + From<usize>>(
    expression: &Expression<F>,
    num_vars: usize,
    evals: &BTreeMap<Query, F>,
    challenges: &[F],
    ys: &[&[F]],
    x: &[F],
) -> F {
    let rotatable = R::from(num_vars);

    let identity = identity_eval(x);
    let lagranges = {
        expression
            .used_langrange()
            .into_iter()
            .map(|i| (i, lagrange_eval(x, rotatable.nth(i))))
            .collect::<BTreeMap<_, _>>()
    };
    let eq_xys = ys.iter().map(|y| eq_xy_eval(x, y)).collect_vec();
    expression.evaluate(
        &|scalar| scalar,
        &|poly| match poly {
            CommonPolynomial::Identity => identity,
            CommonPolynomial::Lagrange(i) => lagranges[&i],
            CommonPolynomial::EqXY(idx) => eq_xys[idx],
        },
        &|query| evals[&query],
        &|idx| challenges[idx],
        &|scalar| -scalar,
        &|lhs, rhs| lhs + &rhs,
        &|lhs, rhs| lhs * &rhs,
        &|value, scalar| scalar * value,
    )
}

pub fn lagrange_eval<F: PrimeField>(x: &[F], b: usize) -> F {
    assert!(!x.is_empty());

    product(x.iter().enumerate().map(
        |(idx, x_i)| {
            if b.nth_bit(idx) {
                *x_i
            } else {
                F::ONE - x_i
            }
        },
    ))
}

pub fn eq_xy_eval<F: PrimeField>(x: &[F], y: &[F]) -> F {
    assert!(!x.is_empty());
    assert_eq!(x.len(), y.len());

    product(
        x.iter()
            .zip(y)
            .map(|(x_i, y_i)| (*x_i * y_i).double() + F::ONE - x_i - y_i),
    )
}

fn identity_eval<F: PrimeField>(x: &[F]) -> F {
    inner_product(x, &powers(F::from(2)).take(x.len()).collect_vec())
}

#[cfg(test)]
pub(super) mod test {
    use crate::{
        piop::sum_check::{evaluate, SumCheck, VirtualPolynomial},
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::Field,
            expression::{rotate::Rotatable, Expression},
            transcript::{InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::Fr;
    use itertools::Itertools;
    use std::ops::Range;

    pub fn run_sum_check<S: SumCheck<Fr>, R: Rotatable + From<usize>>(
        num_vars_range: Range<usize>,
        expression_fn: impl Fn(usize) -> Expression<Fr>,
        param_fn: impl Fn(usize) -> (S::ProverParam, S::VerifierParam),
        assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Vec<Fr>>),
        sum: Fr,
    ) {
        for num_vars in num_vars_range {
            let expression = expression_fn(num_vars);
            let degree = expression.degree();
            let (pp, vp) = param_fn(expression.degree());
            let (polys, challenges, ys) = assignment_fn(num_vars);
            let (evals, proof) = {
                let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
                let mut transcript = Keccak256Transcript::default();
                let (_, _, evals) =
                    S::prove(&pp, num_vars, virtual_poly, sum, &mut transcript).unwrap();
                (evals, transcript.into_proof())
            };
            let accept = {
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                let (x_eval, x) =
                    S::verify(&vp, num_vars, degree, Fr::ZERO, &mut transcript).unwrap();
                let ys = ys.iter().map(Vec::as_slice).collect_vec();
                x_eval == evaluate::<_, R>(&expression, num_vars, &evals, &challenges, &ys, &x)
            };
            assert!(accept);
        }
    }

    pub fn run_zero_check<S: SumCheck<Fr>, R: Rotatable + From<usize>>(
        num_vars_range: Range<usize>,
        expression_fn: impl Fn(usize) -> Expression<Fr>,
        param_fn: impl Fn(usize) -> (S::ProverParam, S::VerifierParam),
        assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Vec<Fr>>),
    ) {
        run_sum_check::<S, R>(
            num_vars_range,
            expression_fn,
            param_fn,
            assignment_fn,
            Fr::ZERO,
        )
    }

    macro_rules! tests {
        ($suffix:ident, $impl:ty, $rotatable:ident) => {
            paste::paste! {
                #[test]
                fn [<lagrange_w_ $suffix>]() {
                    use halo2_curves::bn256::Fr;
                    use $crate::{
                        piop::sum_check::test::run_zero_check,
                        poly::multilinear::MultilinearPolynomial,
                        util::{
                            arithmetic::Field,
                            expression::{
                                rotate::Rotatable, CommonPolynomial, Expression, Query, Rotation,
                            },
                            test::{rand_vec, seeded_std_rng},
                            Itertools,
                        },
                    };

                    run_zero_check::<$impl, $rotatable>(
                        2..4,
                        |num_vars| {
                            let polys = (0..$rotatable::new(num_vars).usable_indices().len())
                                .map(|idx| {
                                    Expression::<Fr>::Polynomial(Query::new(idx, Rotation::cur()))
                                })
                                .collect_vec();
                            let gates = polys
                                .iter()
                                .enumerate()
                                .map(|(i, poly)| {
                                    Expression::CommonPolynomial(CommonPolynomial::Lagrange(i as i32))
                                        - poly
                                })
                                .collect_vec();
                            let alpha = Expression::Challenge(0);
                            let eq = Expression::eq_xy(0);
                            Expression::distribute_powers(&gates, &alpha) * eq
                        },
                        |_| ((), ()),
                        |num_vars| {
                            let polys = $rotatable::new(num_vars)
                                .usable_indices()
                                .into_iter()
                                .map(|idx| {
                                    let mut polys =
                                        MultilinearPolynomial::new(vec![Fr::ZERO; 1 << num_vars]);
                                    polys[idx] = Fr::ONE;
                                    polys
                                })
                                .collect_vec();
                            let alpha = Fr::random(seeded_std_rng());
                            (polys, vec![alpha], vec![rand_vec(num_vars, seeded_std_rng())])
                        },
                    );
                }

                #[test]
                fn [<rotation_w_ $suffix>]() {
                    use halo2_curves::bn256::Fr;
                    use std::iter;
                    use $crate::{
                        piop::sum_check::test::run_zero_check,
                        poly::multilinear::MultilinearPolynomial,
                        util::{
                            arithmetic::Field,
                            expression::{rotate::Rotatable, Expression, Query, Rotation},
                            test::{rand_vec, seeded_std_rng},
                            Itertools,
                        },
                    };

                    run_zero_check::<$impl, $rotatable>(
                        2..16,
                        |num_vars| {
                            let polys = (-(num_vars as i32) + 1..num_vars as i32)
                                .rev()
                                .enumerate()
                                .map(|(idx, rotation)| {
                                    Expression::<Fr>::Polynomial(Query::new(idx, rotation))
                                })
                                .collect_vec();
                            let gates = polys
                                .windows(2)
                                .map(|polys| &polys[1] - &polys[0])
                                .collect_vec();
                            let alpha = Expression::Challenge(0);
                            let eq = Expression::eq_xy(0);
                            Expression::distribute_powers(&gates, &alpha) * eq
                        },
                        |_| ((), ()),
                        |num_vars| {
                            let rotatable = $rotatable::from(num_vars);
                            let rotate = |f: &Vec<Fr>| {
                                (0..1 << num_vars)
                                    .map(|idx| f[rotatable.rotate(idx, Rotation::next())])
                                    .collect_vec()
                            };
                            let poly = rand_vec(1 << num_vars, seeded_std_rng());
                            let polys = iter::successors(Some(poly), |poly| Some(rotate(poly)))
                                .map(MultilinearPolynomial::new)
                                .take(2 * num_vars - 1)
                                .collect_vec();
                            let alpha = Fr::random(seeded_std_rng());
                            (polys, vec![alpha], vec![rand_vec(num_vars, seeded_std_rng())])
                        },
                    );
                }

                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    use halo2_curves::bn256::Fr;
                    use $crate::{
                        backend::hyperplonk::util::{
                            rand_vanilla_plonk_assignment, vanilla_plonk_expression,
                        },
                        piop::sum_check::test::run_zero_check,
                        util::test::{rand_vec, seeded_std_rng},
                    };

                    run_zero_check::<$impl, $rotatable>(
                        2..16,
                        |num_vars| vanilla_plonk_expression(num_vars),
                        |_| ((), ()),
                        |num_vars| {
                            let (polys, challenges) = rand_vanilla_plonk_assignment::<_, $rotatable>(
                                num_vars,
                                seeded_std_rng(),
                                seeded_std_rng(),
                            );
                            (polys, challenges, vec![rand_vec(num_vars, seeded_std_rng())])
                        },
                    );
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    use halo2_curves::bn256::Fr;
                    use $crate::{
                        backend::hyperplonk::util::{
                            rand_vanilla_plonk_w_lookup_assignment,
                            vanilla_plonk_w_lookup_expression,
                        },
                        piop::sum_check::test::run_zero_check,
                        util::test::{rand_vec, seeded_std_rng},
                    };

                    run_zero_check::<$impl, $rotatable>(
                        2..16,
                        |num_vars| vanilla_plonk_w_lookup_expression(num_vars),
                        |_| ((), ()),
                        |num_vars| {
                            let (polys, challenges) = rand_vanilla_plonk_w_lookup_assignment::<
                                _,
                                $rotatable,
                            >(
                                num_vars, seeded_std_rng(), seeded_std_rng()
                            );
                            (polys, challenges, vec![rand_vec(num_vars, seeded_std_rng())])
                        },
                    );
                }
            }
        };
    }

    pub(super) use tests;
}
