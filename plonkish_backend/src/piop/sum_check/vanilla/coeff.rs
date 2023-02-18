use crate::{
    piop::sum_check::vanilla::{ProverState, VanillaSumCheckProver, VanillaSumCheckRoundMessage},
    poly::multilinear::zip_self,
    util::{
        arithmetic::{div_ceil, horner, PrimeField},
        expression::{CommonPolynomial, Expression, Rotation},
        impl_index,
        parallel::{num_threads, parallelize_iter},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
        Itertools,
    },
    Error,
};
use std::{fmt::Debug, iter, ops::AddAssign};

#[derive(Debug)]
pub struct Coefficients<F>(Vec<F>);

impl<F: PrimeField> VanillaSumCheckRoundMessage<F> for Coefficients<F> {
    type Auxiliary = ();

    fn write(&self, transcript: &mut impl FieldTranscriptWrite<F>) -> Result<(), Error> {
        for eval in self.0.iter() {
            transcript.write_field_element(eval)?;
        }
        Ok(())
    }

    fn read(degree: usize, transcript: &mut impl FieldTranscriptRead<F>) -> Result<Self, Error> {
        transcript.read_n_field_elements(degree + 1).map(Self)
    }

    fn sum(&self) -> F {
        self[1..]
            .iter()
            .fold(self[0].double(), |acc, coeff| acc + coeff)
    }

    fn evaluate(&self, _: &Self::Auxiliary, challenge: &F) -> F {
        horner(&self.0, challenge)
    }
}

impl<'rhs, F: PrimeField> AddAssign<&'rhs F> for Coefficients<F> {
    fn add_assign(&mut self, rhs: &'rhs F) {
        self[0] += rhs;
    }
}

impl<'rhs, F: PrimeField> AddAssign<(&'rhs F, &'rhs Coefficients<F>)> for Coefficients<F> {
    fn add_assign(&mut self, (scalar, rhs): (&'rhs F, &'rhs Coefficients<F>)) {
        if scalar == &F::one() {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(lhs, rhs)| *lhs += rhs)
        } else if scalar != &F::zero() {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(lhs, rhs)| *lhs += &(*scalar * rhs))
        }
    }
}

impl_index!(Coefficients, 0);

#[derive(Clone, Debug)]
pub struct CoefficientsProver<F: PrimeField>(F, Vec<(F, Vec<Expression<F>>)>);

impl<F> VanillaSumCheckProver<F> for CoefficientsProver<F>
where
    F: PrimeField,
{
    type RoundMessage = Coefficients<F>;

    fn new(state: &ProverState<F>) -> Self {
        let (constant, flattened) = state.expression.evaluate(
            &|constant| (constant, vec![]),
            &|poly| {
                (
                    F::zero(),
                    vec![(F::one(), vec![Expression::CommonPolynomial(poly)])],
                )
            },
            &|query| {
                (
                    F::zero(),
                    vec![(F::one(), vec![Expression::Polynomial(query)])],
                )
            },
            &|challenge| (state.challenges[challenge], vec![]),
            &|(constant, mut products)| {
                products.iter_mut().for_each(|(scalar, _)| {
                    *scalar = -*scalar;
                });
                (-constant, products)
            },
            &|(lhs_constnat, mut lhs_products), (rhs_constnat, rhs_products)| {
                lhs_products.extend(rhs_products);
                (lhs_constnat + rhs_constnat, lhs_products)
            },
            &|(lhs_constant, lhs_products), (rhs_constant, rhs_products)| {
                let mut outputs =
                    Vec::with_capacity((lhs_products.len() + 1) * (rhs_products.len() + 1));
                for (constant, products) in
                    [(lhs_constant, &rhs_products), (rhs_constant, &lhs_products)]
                {
                    if constant != F::zero() {
                        outputs.extend(
                            products
                                .iter()
                                .map(|(scalar, polys)| (constant * scalar, polys.clone())),
                        )
                    }
                }
                for ((lhs_scalar, lhs_polys), (rhs_scalar, rhs_polys)) in
                    lhs_products.iter().cartesian_product(rhs_products.iter())
                {
                    outputs.push((
                        *lhs_scalar * rhs_scalar,
                        iter::empty()
                            .chain(lhs_polys)
                            .chain(rhs_polys)
                            .cloned()
                            .collect_vec(),
                    ));
                }
                (lhs_constant * rhs_constant, outputs)
            },
            &|(constant, mut products), rhs| {
                products.iter_mut().for_each(|(lhs, _)| {
                    *lhs *= &rhs;
                });
                (constant * &rhs, products)
            },
        );
        Self(constant, flattened)
    }

    fn prove_round(&self, state: &ProverState<F>) -> Self::RoundMessage {
        let mut coeffs = Coefficients(vec![F::zero(); state.expression.degree() + 1]);
        coeffs += &(F::from(state.size() as u64) * &self.0);
        if self.1.iter().all(|(_, products)| products.len() == 2) {
            for (scalar, products) in self.1.iter() {
                let [lhs, rhs] = [0, 1].map(|idx| &products[idx]);
                coeffs += (scalar, &self.karatsuba::<true>(state, lhs, rhs));
            }
            coeffs[1] = state.sum - coeffs[0].double() - coeffs[2];
        } else {
            unimplemented!()
        }
        coeffs
    }
}

impl<F: PrimeField> CoefficientsProver<F> {
    fn karatsuba<const LAZY: bool>(
        &self,
        state: &ProverState<F>,
        lhs: &Expression<F>,
        rhs: &Expression<F>,
    ) -> Coefficients<F> {
        let mut coeffs = [F::zero(); 3];
        match (lhs, rhs) {
            (
                Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
                Expression::Polynomial(query),
            )
            | (
                Expression::Polynomial(query),
                Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)),
            ) if query.rotation() == Rotation::cur() => {
                let lhs = &state.eq_xys[*idx];
                let rhs = &state.polys[query.poly()][state.num_vars];

                let evaluate_serial = |coeffs: &mut [F; 3], start: usize, n: usize| {
                    zip_self!(lhs.iter(), 2, start)
                        .zip(zip_self!(rhs.iter(), 2, start))
                        .take(n)
                        .for_each(|((lhs_0, lhs_1), (rhs_0, rhs_1))| {
                            let coeff_0 = *lhs_0 * rhs_0;
                            let coeff_2 = (*lhs_1 - lhs_0) * &(*rhs_1 - rhs_0);
                            coeffs[0] += &coeff_0;
                            coeffs[2] += &coeff_2;
                            if !LAZY {
                                coeffs[1] += &(*lhs_1 * rhs_1 - &coeff_0 - &coeff_2);
                            }
                        });
                };

                let num_threads = num_threads();
                if state.size() < num_threads {
                    evaluate_serial(&mut coeffs, 0, state.size());
                } else {
                    let chunk_size = div_ceil(state.size(), num_threads);
                    let mut partials = vec![[F::zero(); 3]; num_threads];
                    parallelize_iter(
                        partials.iter_mut().zip((0..).step_by(chunk_size << 1)),
                        |(partial, start)| {
                            evaluate_serial(partial, start, chunk_size);
                        },
                    );
                    partials.iter().for_each(|partial| {
                        coeffs[0] += partial[0];
                        coeffs[2] += partial[2];
                        if !LAZY {
                            coeffs[1] += partial[1];
                        }
                    })
                };
            }
            _ => unimplemented!(),
        }
        Coefficients(coeffs.to_vec())
    }
}
