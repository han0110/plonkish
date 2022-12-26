use crate::{
    piop::sum_check::prover::ProvingState,
    util::{
        arithmetic::PrimeField,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use std::iter;

mod prover;
mod verifier;

pub use prover::VirtualPolynomial;
pub use verifier::{eq_xy_eval, lagrange_eval, VirtualPolynomialInfo};

pub fn prove<F: PrimeField>(
    num_vars: usize,
    virtual_poly: VirtualPolynomial<F>,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<Vec<F>, Error> {
    let mut state = ProvingState::new(num_vars, virtual_poly);
    iter::repeat_with(|| {
        for sample_eval in state.sample_evals() {
            transcript.write_scalar(sample_eval)?;
        }

        let challenge = transcript.squeeze_challenge();
        state.next_round(challenge);

        Ok(challenge)
    })
    .take(num_vars)
    .try_collect()
}

pub fn verify_consistency<F: PrimeField>(
    num_vars: usize,
    virtual_poly_info: &VirtualPolynomialInfo<F>,
    sum: F,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<(F, Vec<F>), Error> {
    let rounds = iter::repeat_with(|| {
        Ok((
            transcript.read_n_scalars(virtual_poly_info.degree() + 1)?,
            transcript.squeeze_challenge(),
        ))
    })
    .take(num_vars)
    .try_collect::<_, Vec<_>, _>()?;

    verifier::verify_consistency(virtual_poly_info, sum, &rounds)
}

#[cfg(test)]
mod test {
    use crate::{
        piop::sum_check::{prove, verify_consistency, VirtualPolynomial, VirtualPolynomialInfo},
        poly::multilinear::{rotation_eval, MultilinearPolynomial},
        snark::hyperplonk::{
            preprocess::test::{plonk_virtual_poly_info, plonk_with_lookup_virtual_poly_info},
            test::{rand_plonk_assignment, rand_plonk_with_lookup_assignment},
        },
        util::{
            arithmetic::{BooleanHypercube, Field},
            expression::{CommonPolynomial, Expression, Query, Rotation},
            test::rand_vec,
            transcript::Keccak256Transcript,
            Itertools,
        },
    };
    use halo2_curves::bn256::{Fr, G1Affine};
    use rand::rngs::OsRng;
    use std::{iter, ops::Range};

    fn run_sum_check(
        num_vars_range: Range<usize>,
        virtual_poly_info_fn: impl Fn(usize) -> VirtualPolynomialInfo<Fr>,
        assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Fr>),
    ) {
        for num_vars in num_vars_range {
            let virtual_poly_info = virtual_poly_info_fn(num_vars);
            let (polys, challenges, y) = assignment_fn(num_vars);
            let proof = {
                let virtual_poly = VirtualPolynomial::new(
                    &virtual_poly_info,
                    &polys,
                    challenges.clone(),
                    vec![y.clone()],
                );
                let mut transcript = Keccak256Transcript::<_, G1Affine>::new(Vec::new());
                prove(num_vars, virtual_poly, &mut transcript).unwrap();
                transcript.finalize()
            };
            let accept = {
                let mut transcript = Keccak256Transcript::<_, G1Affine>::new(proof.as_slice());
                let (x_eval, x) =
                    verify_consistency(num_vars, &virtual_poly_info, Fr::zero(), &mut transcript)
                        .unwrap();
                let evals = virtual_poly_info
                    .expression()
                    .used_query()
                    .into_iter()
                    .map(|query| {
                        let evaluate_for_rotation =
                            polys[query.poly()].evaluate_for_rotation(&x, query.rotation());
                        let eval = rotation_eval(&x, query.rotation(), &evaluate_for_rotation);
                        (query, eval)
                    })
                    .collect();
                x_eval == virtual_poly_info.evaluate(num_vars, &evals, &challenges, &[&y], &x)
            };
            assert!(accept);
        }
    }

    #[test]
    fn test_sum_check_lagrange() {
        run_sum_check(
            2..4,
            |num_vars| {
                let polys = (0..1 << num_vars)
                    .map(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())))
                    .collect_vec();
                let gates = polys
                    .iter()
                    .enumerate()
                    .map(|(i, poly)| {
                        Expression::CommonPolynomial(CommonPolynomial::Lagrange(i as i32)) - poly
                    })
                    .collect_vec();
                let alpha = Expression::Challenge(0);
                VirtualPolynomialInfo::new(Expression::distribute_powers(&gates, &alpha))
            },
            |num_vars| {
                let polys = BooleanHypercube::new(num_vars)
                    .iter()
                    .map(|idx| {
                        let mut polys = MultilinearPolynomial::new(vec![Fr::zero(); 1 << num_vars]);
                        polys[idx] = Fr::one();
                        polys
                    })
                    .collect_vec();
                let alpha = Fr::random(OsRng);
                (polys, vec![alpha], rand_vec(num_vars, OsRng))
            },
        );
    }

    #[test]
    fn test_sum_check_rotation() {
        run_sum_check(
            2..16,
            |num_vars| {
                let polys = (-(num_vars as i32) + 1..num_vars as i32)
                    .rev()
                    .enumerate()
                    .map(|(idx, rotation)| Expression::Polynomial(Query::new(idx, rotation.into())))
                    .collect_vec();
                let gates = polys
                    .windows(2)
                    .map(|polys| &polys[1] - &polys[0])
                    .collect_vec();
                let alpha = Expression::Challenge(0);
                VirtualPolynomialInfo::new(Expression::distribute_powers(&gates, &alpha))
            },
            |num_vars| {
                let bh = BooleanHypercube::new(num_vars);
                let rotate = |f: &Vec<Fr>| {
                    (0..1 << num_vars)
                        .map(|idx| f[bh.rotate(idx, Rotation::next())])
                        .collect_vec()
                };
                let poly = rand_vec(1 << num_vars, OsRng);
                let polys = iter::successors(Some(poly), |poly| Some(rotate(poly)))
                    .map(MultilinearPolynomial::new)
                    .take(2 * num_vars - 1)
                    .collect_vec();
                let alpha = Fr::random(OsRng);
                (polys, vec![alpha], rand_vec(num_vars, OsRng))
            },
        );
    }

    #[test]
    fn test_sum_check_plonk() {
        run_sum_check(
            2..16,
            |_| plonk_virtual_poly_info(),
            |num_vars| {
                let (polys, challenges) = rand_plonk_assignment(num_vars, OsRng);
                (polys, challenges, rand_vec(num_vars, OsRng))
            },
        );
    }

    #[test]
    fn test_sum_check_plonk_with_lookup() {
        run_sum_check(
            2..16,
            |_| plonk_with_lookup_virtual_poly_info(),
            |num_vars| {
                let (polys, challenges) = rand_plonk_with_lookup_assignment(num_vars, OsRng);
                (polys, challenges, rand_vec(num_vars, OsRng))
            },
        );
    }
}
