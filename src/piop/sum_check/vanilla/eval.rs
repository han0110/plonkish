use crate::{
    piop::sum_check::vanilla::{ProverState, VanillaSumCheckProver, VanillaSumCheckRoundMessage},
    util::{
        arithmetic::{barycentric_interpolate, barycentric_weights, div_ceil, PrimeField},
        expression::CommonPolynomial,
        parallel::{num_threads, parallelize_iter},
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use num_integer::Integer;
use std::{fmt::Debug, marker::PhantomData, ops::Range};

#[derive(Debug)]
pub struct Evaluations<F: PrimeField>(Vec<F>);

impl<F: PrimeField> Evaluations<F> {
    fn points(degree: usize) -> Vec<F> {
        (0..degree as u64 + 1).map_into().collect_vec()
    }
}

impl<F: PrimeField> VanillaSumCheckRoundMessage<F> for Evaluations<F> {
    type Auxiliary = (Vec<F>, Vec<F>);

    fn write(&self, transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error> {
        for eval in self.0.iter().copied() {
            transcript.write_scalar(eval)?;
        }
        Ok(())
    }

    fn read(degree: usize, transcript: &mut impl TranscriptRead<F>) -> Result<Self, Error> {
        transcript.read_n_scalars(degree + 1).map(Self)
    }

    fn sum(&self) -> F {
        self.0[0] + self.0[1]
    }

    fn auxiliary(degree: usize) -> Self::Auxiliary {
        let points = Self::points(degree);
        (barycentric_weights(&points), points)
    }

    fn evaluate(&self, (weights, points): &Self::Auxiliary, challenge: &F) -> F {
        barycentric_interpolate(weights, points, &self.0, challenge)
    }
}

#[derive(Clone, Debug)]
pub struct EvaluationsProver<F: PrimeField>(PhantomData<F>);

impl<F> VanillaSumCheckProver<F> for EvaluationsProver<F>
where
    F: PrimeField,
{
    type RoundMessage = Evaluations<F>;

    fn new(_: &ProverState<F>) -> Self {
        Self(PhantomData)
    }

    fn prove_round<'a>(
        &self,
        state: &mut ProverState<'a, F>,
        challenge: Option<&F>,
    ) -> Self::RoundMessage {
        if let Some(challenge) = challenge {
            state.next_round(challenge);
        }
        self.evals(state)
    }
}

impl<F: PrimeField> EvaluationsProver<F> {
    fn evals<'a>(&self, state: &ProverState<'a, F>) -> Evaluations<F> {
        let size = state.size();
        let points = Evaluations::points(state.expression.degree());

        let mut evals = vec![F::zero(); points.len()];
        let num_threads = num_threads();
        if size < num_threads {
            evals
                .iter_mut()
                .zip(points.iter())
                .for_each(|(eval, point)| {
                    if state.round > 0 {
                        self.evaluate::<false>(eval, state, 0..size, point);
                    } else {
                        self.evaluate::<true>(eval, state, 0..size, point);
                    }
                });
        } else {
            let chunk_size = div_ceil(size, num_threads);
            evals
                .iter_mut()
                .zip(points.iter())
                .for_each(|(eval, point)| {
                    let mut partials = vec![F::zero(); num_threads];
                    parallelize_iter(
                        partials.iter_mut().zip((0..).step_by(chunk_size)),
                        |(partial, start)| {
                            let range = start..(start + chunk_size).min(size);
                            if state.round > 0 {
                                self.evaluate::<false>(partial, state, range, point);
                            } else {
                                self.evaluate::<true>(partial, state, range, point);
                            }
                        },
                    );
                    partials.iter().for_each(|partial| *eval += partial);
                });
        }
        Evaluations(evals)
    }

    fn evaluate<'a, const IS_FIRST_ROUND: bool>(
        &self,
        sum: &mut F,
        state: &ProverState<'a, F>,
        range: Range<usize>,
        point: &F,
    ) {
        let partial_identity_eval = F::from((1 << state.round) as u64) * point;
        for b in range {
            *sum += state.expression.evaluate(
                &|scalar| scalar,
                &|poly| match poly {
                    CommonPolynomial::Lagrange(i) => {
                        let lagrange = &state.lagranges[&i];
                        if b == lagrange.0 >> 1 {
                            if lagrange.0.is_even() {
                                lagrange.1 * &(F::one() - point)
                            } else {
                                lagrange.1 * point
                            }
                        } else {
                            F::zero()
                        }
                    }
                    CommonPolynomial::EqXY(idx) => {
                        let poly = &state.eq_xys[idx];
                        poly[b << 1] + &((poly[(b << 1) + 1] - &poly[b << 1]) * point)
                    }
                    CommonPolynomial::Identity(idx) => {
                        state.identities[idx]
                            + F::from((b << (state.round + 1)) as u64)
                            + &partial_identity_eval
                    }
                },
                &|query| {
                    let [b_0, b_1] = if IS_FIRST_ROUND {
                        [b << 1, (b << 1) + 1].map(|b| state.bh.rotate(b, query.rotation()))
                    } else {
                        [b << 1, (b << 1) + 1]
                    };
                    let poly = state.poly::<IS_FIRST_ROUND>(&query);
                    poly[b_0] + &((poly[b_1] - &poly[b_0]) * point)
                },
                &|idx| state.challenges[idx],
                &|scalar| -scalar,
                &|lhs, rhs| lhs + &rhs,
                &|lhs, rhs| lhs * &rhs,
                &|value, scalar| scalar * &value,
            );
        }
    }
}

#[cfg(test)]
mod test {
    use crate::piop::sum_check::{
        test::tests,
        vanilla::{EvaluationsProver, VanillaSumCheck},
    };

    tests!(VanillaSumCheck<EvaluationsProver<Fr>>);
}
