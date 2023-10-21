//! Implementation of GKR for fractional sumchecks in [PH23].
//! Notations are same as in section 3.
//!
//! [PH23]: https://eprint.iacr.org/2023/1284.pdf

use crate::{
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        evaluate, SumCheck as _, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, inner_product, powers, PrimeField},
        chain,
        expression::{Expression, Query, Rotation},
        izip,
        parallel::{num_threads, parallelize_iter},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
        Itertools,
    },
    Error,
};
use std::{array, iter};

type SumCheck<F> = ClassicSumCheck<EvaluationsProver<F>>;

struct Layer<F> {
    p_l: MultilinearPolynomial<F>,
    p_r: MultilinearPolynomial<F>,
    q_l: MultilinearPolynomial<F>,
    q_r: MultilinearPolynomial<F>,
}

impl<F> From<[Vec<F>; 4]> for Layer<F> {
    fn from(values: [Vec<F>; 4]) -> Self {
        let [p_l, p_r, q_l, q_r] = values.map(MultilinearPolynomial::new);
        Self { p_l, p_r, q_l, q_r }
    }
}

impl<F: PrimeField> Layer<F> {
    fn bottom((p, q): (&&MultilinearPolynomial<F>, &&MultilinearPolynomial<F>)) -> Self {
        let mid = p.evals().len() >> 1;
        [&p[..mid], &p[mid..], &q[..mid], &q[mid..]]
            .map(ToOwned::to_owned)
            .into()
    }

    fn num_vars(&self) -> usize {
        self.p_l.num_vars()
    }

    fn polys(&self) -> [&MultilinearPolynomial<F>; 4] {
        [&self.p_l, &self.p_r, &self.q_l, &self.q_r]
    }

    fn poly_chunks(&self, chunk_size: usize) -> impl Iterator<Item = (&[F], &[F], &[F], &[F])> {
        let [p_l, p_r, q_l, q_r] = self.polys().map(|poly| poly.evals().chunks(chunk_size));
        izip!(p_l, p_r, q_l, q_r)
    }

    fn up(&self) -> Self {
        assert!(self.num_vars() != 0);

        let len = 1 << self.num_vars();
        let chunk_size = div_ceil(len, num_threads()).next_power_of_two();

        let mut outputs: [_; 4] = array::from_fn(|_| vec![F::ZERO; len >> 1]);
        let (p, q) = outputs.split_at_mut(2);
        parallelize_iter(
            izip!(
                chain![p].flat_map(|p| p.chunks_mut(chunk_size)),
                chain![q].flat_map(|q| q.chunks_mut(chunk_size)),
                self.poly_chunks(chunk_size),
            ),
            |(p, q, (p_l, p_r, q_l, q_r))| {
                izip!(p, q, p_l, p_r, q_l, q_r).for_each(|(p, q, p_l, p_r, q_l, q_r)| {
                    *p = *p_l * q_r + *p_r * q_l;
                    *q = *q_l * q_r;
                })
            },
        );

        outputs.into()
    }
}

#[allow(clippy::type_complexity)]
pub fn prove_fractional_sum_check<'a, F: PrimeField>(
    claimed_p_0s: impl IntoIterator<Item = Option<F>>,
    claimed_q_0s: impl IntoIterator<Item = Option<F>>,
    ps: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
    qs: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>), Error> {
    let claimed_p_0s = claimed_p_0s.into_iter().collect_vec();
    let claimed_q_0s = claimed_q_0s.into_iter().collect_vec();
    let ps = ps.into_iter().collect_vec();
    let qs = qs.into_iter().collect_vec();
    let num_batching = claimed_p_0s.len();

    assert!(num_batching != 0);
    assert_eq!(num_batching, claimed_q_0s.len());
    assert_eq!(num_batching, ps.len());
    assert_eq!(num_batching, qs.len());
    for poly in chain![&ps, &qs] {
        assert_eq!(poly.num_vars(), ps[0].num_vars());
    }

    let bottom_layers = izip!(&ps, &qs).map(Layer::bottom).collect_vec();
    let layers = iter::successors(bottom_layers.into(), |layers| {
        (layers[0].num_vars() > 0).then(|| layers.iter().map(Layer::up).collect())
    })
    .collect_vec();

    let [claimed_p_0s, claimed_q_0s]: [_; 2] = {
        let (p_0s, q_0s) = chain![layers.last().unwrap()]
            .map(|layer| {
                let [p_l, p_r, q_l, q_r] = layer.polys().map(|poly| poly[0]);
                (p_l * q_r + p_r * q_l, q_l * q_r)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let mut hash_to_transcript = |claimed: Vec<_>, computed: Vec<_>| {
            izip!(claimed, computed)
                .map(|(claimed, computed)| match claimed {
                    Some(claimed) => {
                        if cfg!(feature = "sanity-check") {
                            assert_eq!(claimed, computed)
                        }
                        transcript.common_field_element(&computed).map(|_| computed)
                    }
                    None => transcript.write_field_element(&computed).map(|_| computed),
                })
                .try_collect::<_, Vec<_>, _>()
        };

        [
            hash_to_transcript(claimed_p_0s, p_0s)?,
            hash_to_transcript(claimed_q_0s, q_0s)?,
        ]
    };

    let expression = sum_check_expression(num_batching);

    let (p_xs, q_xs, x) = layers.iter().rev().try_fold(
        (claimed_p_0s, claimed_q_0s, Vec::new()),
        |result, layers| {
            let (claimed_p_ys, claimed_q_ys, y) = result;

            let num_vars = layers[0].num_vars();
            let polys = layers.iter().flat_map(|layer| layer.polys());

            let (mut x, evals) = if num_vars == 0 {
                (vec![], polys.map(|poly| poly[0]).collect_vec())
            } else {
                let gamma = transcript.squeeze_challenge();

                let (_, x, evals) = {
                    let claim = sum_check_claim(&claimed_p_ys, &claimed_q_ys, gamma);
                    SumCheck::prove(
                        &(),
                        num_vars,
                        VirtualPolynomial::new(&expression, polys, &[gamma], &[y]),
                        claim,
                        transcript,
                    )?
                };

                (x, evals.into_values().collect_vec())
            };

            transcript.write_field_elements(&evals)?;

            let mu = transcript.squeeze_challenge();

            let (p_xs, q_xs) = layer_down_claim(&evals, mu);
            x.push(mu);

            Ok((p_xs, q_xs, x))
        },
    )?;

    if cfg!(feature = "sanity-check") {
        izip!(chain![ps, qs], chain![&p_xs, &q_xs])
            .for_each(|(poly, eval)| assert_eq!(poly.evaluate(&x), *eval));
    }

    Ok((p_xs, q_xs, x))
}

#[allow(clippy::type_complexity)]
pub fn verify_fractional_sum_check<F: PrimeField>(
    num_vars: usize,
    claimed_p_0s: impl IntoIterator<Item = Option<F>>,
    claimed_q_0s: impl IntoIterator<Item = Option<F>>,
    transcript: &mut impl FieldTranscriptRead<F>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>), Error> {
    let claimed_p_0s = claimed_p_0s.into_iter().collect_vec();
    let claimed_q_0s = claimed_q_0s.into_iter().collect_vec();
    let num_batching = claimed_p_0s.len();

    assert!(num_batching != 0);
    assert_eq!(num_batching, claimed_q_0s.len());

    let [claimed_p_0s, claimed_q_0s]: [_; 2] = {
        [claimed_p_0s, claimed_q_0s]
            .into_iter()
            .map(|claimed| {
                claimed
                    .into_iter()
                    .map(|claimed| match claimed {
                        Some(claimed) => transcript.common_field_element(&claimed).map(|_| claimed),
                        None => transcript.read_field_element(),
                    })
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?
            .try_into()
            .unwrap()
    };

    let expression = sum_check_expression(num_batching);

    let (p_xs, q_xs, x) = (0..num_vars).try_fold(
        (claimed_p_0s, claimed_q_0s, Vec::new()),
        |result, num_vars| {
            let (claimed_p_ys, claimed_q_ys, y) = result;

            let (mut x, evals) = if num_vars == 0 {
                let evals = transcript.read_field_elements(4 * num_batching)?;

                for (claimed_p, claimed_q, (&p_l, &p_r, &q_l, &q_r)) in
                    izip!(claimed_p_ys, claimed_q_ys, evals.iter().tuples())
                {
                    if claimed_p != p_l * q_r + p_r * q_l || claimed_q != q_l * q_r {
                        return Err(err_unmatched_sum_check_output());
                    }
                }

                (Vec::new(), evals)
            } else {
                let gamma = transcript.squeeze_challenge();

                let (x_eval, x) = {
                    let claim = sum_check_claim(&claimed_p_ys, &claimed_q_ys, gamma);
                    SumCheck::verify(&(), num_vars, expression.degree(), claim, transcript)?
                };

                let evals = transcript.read_field_elements(4 * num_batching)?;

                let query_eval = {
                    let queries = (0..).map(|idx| Query::new(idx, Rotation::cur()));
                    let evals = izip!(queries, evals.iter().cloned()).collect();
                    evaluate::<_, usize>(&expression, num_vars, &evals, &[gamma], &[&y], &x)
                };
                if x_eval != query_eval {
                    return Err(err_unmatched_sum_check_output());
                }

                (x, evals)
            };

            let mu = transcript.squeeze_challenge();

            let (p_xs, q_xs) = layer_down_claim(&evals, mu);
            x.push(mu);

            Ok((p_xs, q_xs, x))
        },
    )?;

    Ok((p_xs, q_xs, x))
}

fn sum_check_expression<F: PrimeField>(num_batching: usize) -> Expression<F> {
    let exprs = &(0..4 * num_batching)
        .map(|idx| Expression::<F>::Polynomial(Query::new(idx, Rotation::cur())))
        .tuples()
        .flat_map(|(ref p_l, ref p_r, ref q_l, ref q_r)| [p_l * q_r + p_r * q_l, q_l * q_r])
        .collect_vec();
    let eq_xy = &Expression::eq_xy(0);
    let gamma = &Expression::Challenge(0);
    Expression::distribute_powers(exprs, gamma) * eq_xy
}

fn sum_check_claim<F: PrimeField>(claimed_p_ys: &[F], claimed_q_ys: &[F], gamma: F) -> F {
    inner_product(
        izip!(claimed_p_ys, claimed_q_ys).flat_map(|(p, q)| [p, q]),
        &powers(gamma).take(claimed_p_ys.len() * 2).collect_vec(),
    )
}

fn layer_down_claim<F: PrimeField>(evals: &[F], mu: F) -> (Vec<F>, Vec<F>) {
    evals
        .iter()
        .tuples()
        .map(|(&p_l, &p_r, &q_l, &q_r)| (p_l + mu * (p_r - p_l), q_l + mu * (q_r - q_l)))
        .unzip()
}

fn err_unmatched_sum_check_output() -> Error {
    Error::InvalidSumcheck("Unmatched between sum_check output and query evaluation".to_string())
}

#[cfg(test)]
mod test {
    use crate::{
        piop::gkr::fractional_sum_check::{
            prove_fractional_sum_check, verify_fractional_sum_check,
        },
        poly::multilinear::MultilinearPolynomial,
        util::{
            chain, izip_eq,
            test::{rand_vec, seeded_std_rng},
            transcript::{InMemoryTranscript, Keccak256Transcript},
            Itertools,
        },
    };
    use halo2_curves::bn256::Fr;
    use std::iter;

    #[test]
    fn fractional_sum_check() {
        let num_batching = 3;
        for num_vars in 1..16 {
            let mut rng = seeded_std_rng();

            let polys = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(MultilinearPolynomial::new)
                .take(2 * num_batching)
                .collect_vec();
            let claims = vec![None; 2 * num_batching];
            let (ps, qs) = polys.split_at(num_batching);
            let (p_0s, q_0s) = claims.split_at(num_batching);

            let proof = {
                let mut transcript = Keccak256Transcript::new(());
                prove_fractional_sum_check::<Fr>(
                    p_0s.to_vec(),
                    q_0s.to_vec(),
                    ps,
                    qs,
                    &mut transcript,
                )
                .unwrap();
                transcript.into_proof()
            };

            let result = {
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                verify_fractional_sum_check::<Fr>(
                    num_vars,
                    p_0s.to_vec(),
                    q_0s.to_vec(),
                    &mut transcript,
                )
            };
            assert_eq!(result.as_ref().map(|_| ()), Ok(()));

            let (p_xs, q_xs, x) = result.unwrap();
            for (poly, eval) in izip_eq!(chain![ps, qs], chain![p_xs, q_xs]) {
                assert_eq!(poly.evaluate(&x), eval);
            }
        }
    }
}
