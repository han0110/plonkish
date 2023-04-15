use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{arithmetic::Field, Itertools},
    Error,
};

mod brakedown;
mod kzg;

pub use brakedown::{
    MultilinearBrakedown, MultilinearBrakedownCommitment, MultilinearBrakedownParams,
};
pub use kzg::{
    MultilinearKzg, MultilinearKzgCommitment, MultilinearKzgParams, MultilinearKzgProverParams,
    MultilinearKzgVerifierParams,
};

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
