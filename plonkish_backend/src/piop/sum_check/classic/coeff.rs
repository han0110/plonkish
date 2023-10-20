use crate::{
    piop::sum_check::classic::{ClassicSumCheckProver, ClassicSumCheckRoundMessage, ProverState},
    poly::multilinear::{zip_self, MultilinearPolynomial},
    util::{
        arithmetic::{div_ceil, horner, PrimeField},
        chain,
        expression::{CommonPolynomial, Expression, Rotation},
        impl_index, izip_eq,
        parallel::{num_threads, parallelize_iter},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
        Itertools,
    },
    Error,
};
use std::{array, fmt::Debug, ops::AddAssign};

#[derive(Debug)]
pub struct Coefficients<F>(Vec<F>);

impl<F: PrimeField> ClassicSumCheckRoundMessage<F> for Coefficients<F> {
    type Auxiliary = ();

    fn write(&self, transcript: &mut impl FieldTranscriptWrite<F>) -> Result<(), Error> {
        transcript.write_field_elements(&self.0)
    }

    fn read(degree: usize, transcript: &mut impl FieldTranscriptRead<F>) -> Result<Self, Error> {
        transcript.read_field_elements(degree + 1).map(Self)
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
        if scalar == &F::ONE {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(lhs, rhs)| *lhs += rhs)
        } else if scalar != &F::ZERO {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(lhs, rhs)| *lhs += &(*scalar * rhs))
        }
    }
}

impl_index!(Coefficients, 0);

#[derive(Clone, Debug)]
pub struct CoefficientsProver<F: PrimeField> {
    constant: F,
    products: Vec<(F, Vec<Expression<F>>)>,
}

impl<F> ClassicSumCheckProver<F> for CoefficientsProver<F>
where
    F: PrimeField,
{
    type RoundMessage = Coefficients<F>;

    fn new(state: &ProverState<F>) -> Self {
        let (constant, products) = state.expression.evaluate(
            &|constant| (constant, vec![]),
            &|poly| {
                (
                    F::ZERO,
                    vec![(F::ONE, vec![Expression::CommonPolynomial(poly)])],
                )
            },
            &|query| (F::ZERO, vec![(F::ONE, vec![Expression::Polynomial(query)])]),
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
                    if constant != F::ZERO {
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
                        chain![lhs_polys, rhs_polys].cloned().collect_vec(),
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
        Self { constant, products }
    }

    fn prove_round(&self, state: &ProverState<F>) -> Self::RoundMessage {
        let mut coeffs = Coefficients(vec![F::ZERO; state.expression.degree() + 1]);
        coeffs += &(F::from(state.size() as u64) * &self.constant);

        for (scalar, products) in self.products.iter() {
            match products.len() {
                2 => coeffs += (scalar, &self.karatsuba::<true>(state, products)),
                _ => unimplemented!(),
            }
        }

        coeffs[1] = state.sum - coeffs.sum();
        coeffs
    }
}

impl<F: PrimeField> CoefficientsProver<F> {
    fn karatsuba<const LAZY: bool>(
        &self,
        state: &ProverState<F>,
        items: &[Expression<F>],
    ) -> Coefficients<F> {
        debug_assert_eq!(items.len(), 2);

        let [lhs, rhs] = array::from_fn(|idx| poly(state, &items[idx]));
        let evaluate_serial = |coeffs: &mut [F; 3], start: usize, n: usize| {
            izip_eq!(
                zip_self!(lhs.iter(), 2, start),
                zip_self!(rhs.iter(), 2, start)
            )
            .take(n)
            .for_each(|((lhs_0, lhs_1), (rhs_0, rhs_1))| {
                let eval_0 = *lhs_0 * rhs_0;
                let eval_2 = (*lhs_1 - lhs_0) * &(*rhs_1 - rhs_0);
                coeffs[0] += &eval_0;
                coeffs[2] += &eval_2;
                if !LAZY {
                    coeffs[1] += &(*lhs_1 * rhs_1 - &eval_0 - &eval_2);
                }
            });
        };

        let mut coeffs = [F::ZERO; 3];

        let num_threads = num_threads();
        if state.size() < 16 {
            evaluate_serial(&mut coeffs, 0, state.size());
        } else {
            let chunk_size = div_ceil(state.size(), num_threads);
            let mut partials = vec![[F::ZERO; 3]; num_threads];
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

        Coefficients(coeffs.to_vec())
    }
}

fn poly<'a, F: PrimeField>(
    state: &'a ProverState<F>,
    expr: &Expression<F>,
) -> &'a MultilinearPolynomial<F> {
    match expr {
        Expression::CommonPolynomial(CommonPolynomial::EqXY(idx)) => &state.eq_xys[*idx],
        Expression::Polynomial(query) if query.rotation() == Rotation::cur() => {
            &state.polys[&query]
        }
        _ => unimplemented!(),
    }
}
