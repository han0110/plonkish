use crate::{
    sum_check::{prover::ProvingState, verifier::consistency_check},
    util::{
        arithmetic::PrimeField,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use std::iter;

mod poly;
mod prover;
mod verifier;

pub use poly::{VirtualPolynomial, VirtualPolynomialInfo};

pub fn prove<F, T>(virtual_poly: &VirtualPolynomial<F>, transcript: &mut T) -> Result<Vec<F>, Error>
where
    F: PrimeField,
    T: TranscriptWrite<F>,
{
    let mut state = ProvingState::new(virtual_poly);
    iter::repeat_with(|| {
        for sample_eval in state.sample_evals() {
            transcript.write_scalar(sample_eval)?;
        }

        let challenge = transcript.squeeze_challenge();
        state.next_round(challenge);

        Ok(challenge)
    })
    .take(virtual_poly.info.num_vars())
    .collect::<Result<Vec<_>, Error>>()
}

pub fn verify<F, T>(
    sum: F,
    virtual_poly_info: &VirtualPolynomialInfo<F>,
    transcript: &mut T,
) -> Result<(F, Vec<F>), Error>
where
    F: PrimeField,
    T: TranscriptRead<F>,
{
    let rounds = iter::repeat_with(|| {
        Ok((
            transcript.read_n_scalars(virtual_poly_info.degree() + 1)?,
            transcript.squeeze_challenge(),
        ))
    })
    .take(virtual_poly_info.num_vars())
    .collect::<Result<Vec<_>, Error>>()?;
    consistency_check(virtual_poly_info, &rounds, sum).map(|eval| {
        let challenges = rounds
            .into_iter()
            .map(|(_, challenge)| challenge)
            .collect_vec();
        (eval, challenges)
    })
}

#[cfg(test)]
mod test {
    use crate::{
        plonk::test::{
            hyperplonk_expression, hyperplonk_plus_expression, rand_hyperplonk_assignments,
            rand_hyperplonk_plus_assignments,
        },
        poly::multilinear::{compute_rotation_eval, MultilinearPolynomial},
        sum_check::{prove, verify, VirtualPolynomial, VirtualPolynomialInfo},
        util::{
            expression::{Expression, Rotation},
            test::rand_vec,
            transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::rngs::OsRng;
    use std::ops::Range;

    fn run_sum_check(
        num_var_range: Range<usize>,
        expression: Expression<Fr>,
        assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Fr>),
    ) {
        for num_vars in num_var_range {
            let virtual_poly_info = { VirtualPolynomialInfo::new(num_vars, expression.clone()) };
            let (polys, challenges, y) = assignment_fn(num_vars);
            let proof = {
                let virtual_poly = VirtualPolynomial::new(
                    &virtual_poly_info,
                    &polys,
                    challenges.clone(),
                    vec![y.clone()],
                );
                let mut transcript = Keccak256Transcript::<_, Bn256>::new(Vec::new());
                prove(&virtual_poly, &mut transcript).unwrap();
                transcript.finalize()
            };
            let accept = {
                let mut transcript = Keccak256Transcript::<_, Bn256>::new(proof.as_slice());
                let (expected_eval, x) =
                    verify(Fr::zero(), &virtual_poly_info, &mut transcript).unwrap();
                let evals = virtual_poly_info
                    .expression()
                    .used_query()
                    .into_iter()
                    .map(|query| {
                        let eval = if query.rotation() == Rotation::cur() {
                            polys[query.poly()].evaluate(&x)
                        } else {
                            compute_rotation_eval(
                                &polys[query.poly()].evaluate_for_rotation(&x, query.rotation().0),
                                &x,
                            )
                        };
                        (query, eval)
                    })
                    .collect();
                expected_eval == virtual_poly_info.evaluate(&evals, &challenges, &[y], &x)
            };
            assert!(accept);
        }
    }

    #[test]
    fn test_sum_check_hyperplonk() {
        run_sum_check(2..16, hyperplonk_expression(), |num_vars| {
            let (polys, chalenges) =
                rand_hyperplonk_assignments(num_vars, |idx| Fr::from(idx as u64), OsRng);
            (
                polys.to_vec(),
                chalenges.to_vec(),
                rand_vec(num_vars, OsRng),
            )
        });
    }

    #[test]
    fn test_sum_check_hyperplonk_plus() {
        run_sum_check(2..16, hyperplonk_plus_expression(), |num_vars| {
            dbg!(num_vars);
            let (polys, chalenges) =
                rand_hyperplonk_plus_assignments(num_vars, |idx| Fr::from(idx as u64), OsRng);
            (
                polys.to_vec(),
                chalenges.to_vec(),
                rand_vec(num_vars, OsRng),
            )
        });
    }
}
