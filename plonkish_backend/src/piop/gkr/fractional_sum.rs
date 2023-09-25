//! Implementation of GKR for fractional sumchecks in [PH23].
//! Notations are same as in section 3.
//!
//! [PH23]: https://eprint.iacr.org/2023/1284.pdf

use crate::{
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        eq_xy_eval, SumCheck, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, PrimeField},
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
    fn initial(p: &[F], q: &[F]) -> Self {
        let mid = p.len() >> 1;
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
}

pub fn prove_fractional_sum<F: PrimeField>(
    claimed_p: Option<F>,
    claimed_q: Option<F>,
    p: &[F],
    q: &[F],
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(F, F, F, F, Vec<F>), Error> {
    assert_eq!(p.len(), q.len());
    assert!(p.len().is_power_of_two());

    let num_threads = num_threads();

    let initial_layer = Layer::initial(p, q);
    let layers = iter::successors(Some(initial_layer), |layer| {
        let len = 1 << layer.num_vars();
        let chunk_size = div_ceil(len, num_threads).next_power_of_two();
        (len > 1).then(|| {
            let mut outputs: [_; 4] = array::from_fn(|_| vec![F::ZERO; len >> 1]);
            let (p, q) = outputs.split_at_mut(2);
            parallelize_iter(
                izip!(
                    chain![p].flat_map(|p| p.chunks_mut(chunk_size)),
                    chain![q].flat_map(|q| q.chunks_mut(chunk_size)),
                    layer.poly_chunks(chunk_size),
                ),
                |(p, q, (p_l, p_r, q_l, q_r))| {
                    izip!(p, q, p_l, p_r, q_l, q_r).for_each(|(p, q, p_l, p_r, q_l, q_r)| {
                        *p = *p_l * q_r + *p_r * q_l;
                        *q = *q_l * q_r;
                    })
                },
            );
            outputs.into()
        })
    })
    .collect_vec();

    let [claimed_p, claimed_q]: [_; 2] = {
        let [p_l, p_r, q_l, q_r] = layers.last().unwrap().polys().map(|poly| poly[0]);
        let (p, q) = (p_l * q_r + p_r * q_l, q_l * q_r);

        [(claimed_p, p), (claimed_q, q)]
            .into_iter()
            .map(|(claimed, computed)| match claimed {
                Some(claimed) => {
                    if cfg!(feature = "sanity-check") {
                        assert_eq!(claimed, computed)
                    }
                    transcript.common_field_element(&computed).map(|_| claimed)
                }
                None => transcript.write_field_element(&computed).map(|_| computed),
            })
            .try_collect::<_, Vec<_>, _>()?
            .try_into()
            .unwrap()
    };

    let expression = {
        let [p_l, p_r, q_l, q_r] =
            &array::from_fn(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())));
        let eq_xy = &Expression::eq_xy(0);
        let gamma = &Expression::Challenge(0);
        (p_l * q_r + p_r * q_l + gamma * q_l * q_r) * eq_xy
    };

    let (p, q, challenges) =
        layers
            .iter()
            .rev()
            .fold(Ok((claimed_p, claimed_q, Vec::new())), |result, layer| {
                let (claimed_p, claimed_q, y) = result?;
                let num_vars = layer.num_vars();

                let (mut challenges, evals) = if num_vars == 0 {
                    (vec![], layer.polys().map(|poly| poly[0]))
                } else {
                    let gamma = transcript.squeeze_challenge();

                    let claim = claimed_p + gamma * claimed_q;
                    let (challenges, evals) = ClassicSumCheck::<EvaluationsProver<_>>::prove(
                        &(),
                        num_vars,
                        VirtualPolynomial::new(&expression, layer.polys(), &[gamma], &[y]),
                        claim,
                        transcript,
                    )?;

                    (challenges, evals.try_into().unwrap())
                };

                transcript.write_field_elements(&evals)?;

                let mu = transcript.squeeze_challenge();

                let [p_l, p_r, q_l, q_r] = evals;
                let p = p_l + mu * (p_r - p_l);
                let q = q_l + mu * (q_r - q_l);
                challenges.push(mu);

                Ok((p, q, challenges))
            })?;

    if cfg!(feature = "sanity-check") {
        let [p_l, p_r, q_l, q_r] = layers[0].polys().map(|poly| poly.evals().to_vec());
        let p_poly = MultilinearPolynomial::new([p_l, p_r].concat());
        let q_poly = MultilinearPolynomial::new([q_l, q_r].concat());
        assert_eq!(p_poly.evaluate(&challenges), p);
        assert_eq!(q_poly.evaluate(&challenges), q);
    }

    Ok((claimed_p, claimed_q, p, q, challenges))
}

pub fn verify_fractional_sum<F: PrimeField>(
    num_vars: usize,
    claimed_p: Option<F>,
    claimed_q: Option<F>,
    transcript: &mut impl FieldTranscriptRead<F>,
) -> Result<(F, F, F, F, Vec<F>), Error> {
    let [claimed_p, claimed_q]: [_; 2] = {
        [claimed_p, claimed_q]
            .into_iter()
            .map(|claimed| match claimed {
                Some(claimed) => transcript.common_field_element(&claimed).map(|_| claimed),
                None => transcript.read_field_element(),
            })
            .try_collect::<_, Vec<_>, _>()?
            .try_into()
            .unwrap()
    };

    let (p, q, challenges) = (0..num_vars).fold(
        Ok((claimed_p, claimed_q, Vec::new())),
        |result, num_vars| {
            let (claimed_p, claimed_q, y) = result?;

            let (mut challenges, evals) = if num_vars == 0 {
                let evals: [_; 4] = transcript.read_field_elements(4)?.try_into().unwrap();
                let [p_l, p_r, q_l, q_r] = evals;

                if claimed_p != p_l * q_r + p_r * q_l || claimed_q != q_l * q_r {
                    return Err(err_unmatched_sum_check_output());
                }

                (Vec::new(), evals)
            } else {
                let gamma = transcript.squeeze_challenge();

                let claim = claimed_p + gamma * claimed_q;
                let (eval, challenges) = ClassicSumCheck::<EvaluationsProver<_>>::verify(
                    &(),
                    num_vars,
                    3,
                    claim,
                    transcript,
                )?;

                let evals: [_; 4] = transcript.read_field_elements(4)?.try_into().unwrap();
                let [p_l, p_r, q_l, q_r] = evals;

                if eval != (p_l * q_r + p_r * q_l + gamma * q_l * q_r) * eq_xy_eval(&challenges, &y)
                {
                    return Err(err_unmatched_sum_check_output());
                }

                (challenges, evals)
            };

            let mu = transcript.squeeze_challenge();

            let [p_l, p_r, q_l, q_r] = evals;
            let p = p_l + mu * (p_r - p_l);
            let q = q_l + mu * (q_r - q_l);
            challenges.push(mu);

            Ok((p, q, challenges))
        },
    )?;

    Ok((claimed_p, claimed_q, p, q, challenges))
}

fn err_unmatched_sum_check_output() -> Error {
    Error::InvalidSumcheck("Unmatched between sum_check output and query evaluation".to_string())
}

#[cfg(test)]
mod test {
    use crate::{
        piop::gkr::fractional_sum::{prove_fractional_sum, verify_fractional_sum},
        util::{
            test::{rand_vec, seeded_std_rng},
            transcript::{InMemoryTranscript, Keccak256Transcript},
        },
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn fractional_sum() {
        for num_vars in 1..16 {
            let mut rng = seeded_std_rng();

            let p = rand_vec(1 << num_vars, &mut rng);
            let q = rand_vec(1 << num_vars, &mut rng);

            let proof = {
                let mut transcript = Keccak256Transcript::new(());
                prove_fractional_sum::<Fr>(None, None, &p, &q, &mut transcript).unwrap();
                transcript.into_proof()
            };

            let result = {
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                verify_fractional_sum::<Fr>(num_vars, None, None, &mut transcript)
            };
            assert_eq!(result.map(|_| ()), Ok(()));
        }
    }
}
