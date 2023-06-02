use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{arithmetic::Field, Itertools},
    Error,
};

mod brakedown;
mod ipa;
mod kzg;
mod simulator;

pub use brakedown::{
    MultilinearBrakedown, MultilinearBrakedownCommitment, MultilinearBrakedownParams,
};
pub use ipa::{MultilinearIpa, MultilinearIpaCommitment, MultilinearIpaParams};
pub use kzg::{
    MultilinearKzg, MultilinearKzgCommitment, MultilinearKzgParams, MultilinearKzgProverParams,
    MultilinearKzgVerifierParams,
};
pub use simulator::MultilinearSimulator;

fn validate_input<'a, F: Field>(
    function: &str,
    param_num_vars: usize,
    polys: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
    points: impl IntoIterator<Item = &'a Vec<F>>,
) -> Result<(), Error> {
    let polys = polys.into_iter().collect_vec();
    let points = points.into_iter().collect_vec();
    for poly in polys.iter() {
        if param_num_vars < poly.num_vars() {
            return Err(err_too_many_variates(
                function,
                param_num_vars,
                poly.num_vars(),
            ));
        }
    }
    let input_num_vars = polys
        .iter()
        .map(|poly| poly.num_vars())
        .chain(points.iter().map(|point| point.len()))
        .next()
        .expect("To have at least 1 poly or point");
    for point in points.into_iter() {
        if point.len() != input_num_vars {
            return Err(Error::InvalidPcsParam(format!(
                "Invalid point (expect point to have {input_num_vars} variates but got {})",
                point.len()
            )));
        }
    }
    Ok(())
}

fn err_too_many_variates(function: &str, upto: usize, got: usize) -> Error {
    if function == "trim" {
        Error::InvalidPcsParam(format!(
            "Too many variates to {function} (param supports variates up to {upto} but got {got})"
        ))
    } else {
        Error::InvalidPcsParam(format!(
            "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
        ))
    }
}

mod additive {
    use crate::{
        pcs::{
            multilinear::validate_input, AdditiveCommitment, Evaluation, Point, Polynomial,
            PolynomialCommitmentScheme,
        },
        piop::sum_check::{
            classic::{ClassicSumCheck, CoefficientsProver},
            eq_xy_eval, SumCheck as _, VirtualPolynomial,
        },
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::{inner_product, PrimeField},
            end_timer,
            expression::{Expression, Query, Rotation},
            start_timer,
            transcript::{TranscriptRead, TranscriptWrite},
            Itertools,
        },
        Error,
    };
    use std::{borrow::Cow, ops::Deref};

    type SumCheck<F> = ClassicSumCheck<CoefficientsProver<F>>;

    pub fn batch_open<F, Pcs>(
        pp: &Pcs::ProverParam,
        num_vars: usize,
        polys: Vec<&Pcs::Polynomial>,
        comms: Option<Vec<&Pcs::CommitmentWithAux>>,
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Pcs::Commitment, F>,
    ) -> Result<(), Error>
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        Pcs::CommitmentWithAux: AdditiveCommitment<F>,
    {
        validate_input("batch open", num_vars, polys.clone(), points)?;

        let ell = evals.len().next_power_of_two().ilog2() as usize;
        let t = transcript.squeeze_challenges(ell);

        let timer = start_timer(|| "merged_polys");
        let eq_xt = MultilinearPolynomial::eq_xy(&t);
        let merged_polys = evals.iter().zip(eq_xt.evals().iter()).fold(
            vec![(F::ONE, Cow::<MultilinearPolynomial<_>>::default()); points.len()],
            |mut merged_polys, (eval, eq_xt_i)| {
                if merged_polys[eval.point()].1.is_zero() {
                    merged_polys[eval.point()] = (*eq_xt_i, Cow::Borrowed(polys[eval.poly()]));
                } else {
                    let coeff = merged_polys[eval.point()].0;
                    if coeff != F::ONE {
                        merged_polys[eval.point()].0 = F::ONE;
                        *merged_polys[eval.point()].1.to_mut() *= &coeff;
                    }
                    *merged_polys[eval.point()].1.to_mut() += (eq_xt_i, polys[eval.poly()]);
                }
                merged_polys
            },
        );
        end_timer(timer);

        let expression = merged_polys
            .iter()
            .enumerate()
            .map(|(idx, (scalar, _))| {
                Expression::<F>::eq_xy(idx)
                    * Expression::Polynomial(Query::new(idx, Rotation::cur()))
                    * scalar
            })
            .sum();
        let virtual_poly = VirtualPolynomial::new(
            &expression,
            merged_polys.iter().map(|(_, poly)| poly.deref()),
            &[],
            points,
        );
        let tilde_gs_sum =
            inner_product(evals.iter().map(Evaluation::value), &eq_xt[..evals.len()]);
        let (challenges, _) =
            SumCheck::prove(&(), num_vars, virtual_poly, tilde_gs_sum, transcript)?;

        let timer = start_timer(|| "g_prime");
        let eq_xy_evals = points
            .iter()
            .map(|point| eq_xy_eval(&challenges, point))
            .collect_vec();
        let g_prime = merged_polys
            .into_iter()
            .zip(eq_xy_evals.iter())
            .map(|((scalar, poly), eq_xy_eval)| (scalar * eq_xy_eval, poly.into_owned()))
            .sum::<MultilinearPolynomial<_>>();
        let g_prime_comm = comms
            .map(|comms| {
                let scalars = evals
                    .iter()
                    .zip(eq_xt.evals())
                    .map(|(eval, eq_xt_i)| eq_xy_evals[eval.point()] * eq_xt_i)
                    .collect_vec();
                let bases = evals.iter().map(|eval| comms[eval.poly()]);
                Pcs::CommitmentWithAux::sum_with_scalar(&scalars, bases)
            })
            .unwrap_or_default();
        end_timer(timer);

        let g_prime_eval = if cfg!(feature = "sanity-check") {
            g_prime.evaluate(&challenges)
        } else {
            F::ZERO
        };
        Pcs::open(
            pp,
            &g_prime,
            &g_prime_comm,
            &challenges,
            &g_prime_eval,
            transcript,
        )
    }

    pub fn batch_verify<F, Pcs>(
        vp: &Pcs::VerifierParam,
        num_vars: usize,
        comms: &[Pcs::Commitment],
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Pcs::Commitment, F>,
    ) -> Result<(), Error>
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        Pcs::Commitment: AdditiveCommitment<F>,
    {
        validate_input("batch verify", num_vars, [], points)?;

        let ell = evals.len().next_power_of_two().ilog2() as usize;
        let t = transcript.squeeze_challenges(ell);

        let eq_xt = MultilinearPolynomial::eq_xy(&t);
        let tilde_gs_sum =
            inner_product(evals.iter().map(Evaluation::value), &eq_xt[..evals.len()]);
        let (g_prime_eval, challenges) =
            SumCheck::verify(&(), num_vars, 2, tilde_gs_sum, transcript)?;

        let eq_xy_evals = points
            .iter()
            .map(|point| eq_xy_eval(&challenges, point))
            .collect_vec();
        let g_prime_comm = {
            let scalars = evals
                .iter()
                .zip(eq_xt.evals())
                .map(|(eval, eq_xt_i)| eq_xy_evals[eval.point()] * eq_xt_i)
                .collect_vec();
            let bases = evals.iter().map(|eval| &comms[eval.poly()]);
            Pcs::Commitment::sum_with_scalar(&scalars, bases)
        };
        Pcs::verify(vp, &g_prime_comm, &challenges, &g_prime_eval, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{Evaluation, PolynomialCommitmentScheme},
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::PrimeField,
            transcript::{
                InMemoryTranscriptRead, InMemoryTranscriptWrite, TranscriptRead, TranscriptWrite,
            },
            Itertools,
        },
    };
    use rand::rngs::OsRng;
    use std::iter;

    pub(super) fn run_commit_open_verify<F, Pcs, T>()
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        T: TranscriptRead<Pcs::Commitment, F>
            + TranscriptWrite<Pcs::Commitment, F>
            + InMemoryTranscriptRead
            + InMemoryTranscriptWrite,
    {
        for num_vars in 3..16 {
            // Setup
            let (pp, vp) = {
                let mut rng = OsRng;
                let size = 1 << num_vars;
                let param = Pcs::setup(size, &mut rng).unwrap();
                Pcs::trim(&param, size).unwrap()
            };
            // Commit and open
            let proof = {
                let mut transcript = T::default();
                let poly = MultilinearPolynomial::rand(num_vars, OsRng);
                let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
                let point = transcript.squeeze_challenges(num_vars);
                let eval = poly.evaluate(point.as_slice());
                transcript.write_field_element(&eval).unwrap();
                Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
                transcript.into_proof()
            };
            // Verify
            let result = {
                let mut transcript = T::from_proof(proof.as_slice());
                Pcs::verify(
                    &vp,
                    &transcript.read_commitment().unwrap(),
                    &transcript.squeeze_challenges(num_vars),
                    &transcript.read_field_element().unwrap(),
                    &mut transcript,
                )
            };
            assert_eq!(result, Ok(()));
        }
    }

    pub(super) fn run_batch_commit_open_verify<F, Pcs, T>()
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        T: TranscriptRead<Pcs::Commitment, F>
            + TranscriptWrite<Pcs::Commitment, F>
            + InMemoryTranscriptRead
            + InMemoryTranscriptWrite,
    {
        for num_vars in 3..16 {
            // Setup
            let (pp, vp) = {
                let mut rng = OsRng;
                let size = 1 << num_vars;
                let param = Pcs::setup(size, &mut rng).unwrap();
                Pcs::trim(&param, size).unwrap()
            };
            // Batch commit and open
            let batch_size = 4;
            let proof = {
                let mut transcript = T::default();
                let polys = iter::repeat_with(|| MultilinearPolynomial::rand(num_vars, OsRng))
                    .take(batch_size)
                    .collect_vec();
                let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();
                let points = iter::repeat_with(|| transcript.squeeze_challenges(num_vars))
                    .take(batch_size * batch_size)
                    .collect_vec();
                let evals = points
                    .iter()
                    .enumerate()
                    .map(|(idx, point)| {
                        Evaluation::new(
                            idx % batch_size,
                            idx,
                            polys[idx % batch_size].evaluate(point.as_slice()),
                        )
                    })
                    .collect_vec();
                transcript
                    .write_field_elements(evals.iter().map(Evaluation::value))
                    .unwrap();
                Pcs::batch_open(&pp, &polys, &comms, &points, &evals, &mut transcript).unwrap();
                transcript.into_proof()
            };
            // Batch verify
            let result = {
                let mut transcript = T::from_proof(proof.as_slice());
                Pcs::batch_verify(
                    &vp,
                    &transcript.read_commitments(batch_size).unwrap(),
                    &iter::repeat_with(|| transcript.squeeze_challenges(num_vars))
                        .take(batch_size * batch_size)
                        .collect_vec(),
                    &transcript
                        .read_field_elements(batch_size * batch_size)
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(|(idx, eval)| Evaluation::new(idx % batch_size, idx, eval))
                        .collect_vec(),
                    &mut transcript,
                )
            };
            assert_eq!(result, Ok(()));
        }
    }
}
