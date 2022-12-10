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
            plonk_expression, plonkup_expression, rand_plonk_assignments, rand_plonkup_assignments,
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
                                &x,
                                query.rotation().0,
                                &polys[query.poly()].evaluate_for_rotation(&x, query.rotation().0),
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
    fn test_sum_check_plonk() {
        run_sum_check(2..16, plonk_expression(), |num_vars| {
            let (polys, chalenges) = rand_plonk_assignments(num_vars, OsRng);
            (
                polys.to_vec(),
                chalenges.to_vec(),
                rand_vec(num_vars, OsRng),
            )
        });
    }

    #[test]
    fn test_sum_check_plonkup() {
        run_sum_check(2..16, plonkup_expression(), |num_vars| {
            let (polys, chalenges) = rand_plonkup_assignments(num_vars, OsRng);
            (
                polys.to_vec(),
                chalenges.to_vec(),
                rand_vec(num_vars, OsRng),
            )
        });
    }
}
