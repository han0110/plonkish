use crate::{
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{arithmetic::Field, end_timer, izip, parallel::parallelize, start_timer, Itertools},
    Error,
};

mod brakedown;
mod gemini;
mod hyrax;
mod ipa;
mod kzg;
mod zeromorph;

pub use brakedown::{
    MultilinearBrakedown, MultilinearBrakedownCommitment, MultilinearBrakedownParams,
};
pub use gemini::Gemini;
pub use hyrax::{MultilinearHyrax, MultilinearHyraxCommitment, MultilinearHyraxParams};
pub use ipa::{MultilinearIpa, MultilinearIpaCommitment, MultilinearIpaParams};
pub use kzg::{
    MultilinearKzg, MultilinearKzgCommitment, MultilinearKzgParams, MultilinearKzgProverParams,
    MultilinearKzgVerifierParams,
};
pub use zeromorph::{Zeromorph, ZeromorphKzgProverParam, ZeromorphKzgVerifierParam};

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
    Error::InvalidPcsParam(if function == "trim" {
        format!(
            "Too many variates to {function} (param supports variates up to {upto} but got {got})"
        )
    } else {
        format!(
            "Too many variates of poly to {function} (param supports variates up to {upto} but got {got})"
        )
    })
}

fn quotients<F: Field, T>(
    poly: &MultilinearPolynomial<F>,
    point: &[F],
    f: impl Fn(usize, Vec<F>) -> T,
) -> (Vec<T>, F) {
    assert_eq!(poly.num_vars(), point.len());

    let mut remainder = poly.evals().to_vec();
    let mut quotients = point
        .iter()
        .zip(0..poly.num_vars())
        .rev()
        .map(|(x_i, num_vars)| {
            let timer = start_timer(|| "quotients");
            let (remaimder_lo, remainder_hi) = remainder.split_at_mut(1 << num_vars);
            let mut quotient = vec![F::ZERO; remaimder_lo.len()];

            parallelize(&mut quotient, |(quotient, start)| {
                izip!(quotient, &remaimder_lo[start..], &remainder_hi[start..])
                    .for_each(|(q, r_lo, r_hi)| *q = *r_hi - r_lo);
            });
            parallelize(remaimder_lo, |(remaimder_lo, start)| {
                izip!(remaimder_lo, &remainder_hi[start..])
                    .for_each(|(r_lo, r_hi)| *r_lo += (*r_hi - r_lo as &_) * x_i);
            });

            remainder.truncate(1 << num_vars);
            end_timer(timer);

            f(num_vars, quotient)
        })
        .collect_vec();
    quotients.reverse();

    (quotients, remainder[0])
}

mod additive {
    use crate::{
        pcs::{
            multilinear::validate_input, AdditiveCommitment, Evaluation, Point,
            PolynomialCommitmentScheme,
        },
        piop::sum_check::{
            classic::{ClassicSumCheck, CoefficientsProver},
            eq_xy_eval, SumCheck as _, VirtualPolynomial,
        },
        poly::{multilinear::MultilinearPolynomial, Polynomial},
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
    use std::{
        borrow::Cow,
        collections::{BTreeMap, HashMap},
        ops::Deref,
        ptr::addr_of,
    };

    type SumCheck<F> = ClassicSumCheck<CoefficientsProver<F>>;

    pub fn batch_open<F, Pcs>(
        pp: &Pcs::ProverParam,
        num_vars: usize,
        polys: Vec<&Pcs::Polynomial>,
        comms: Vec<&Pcs::Commitment>,
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        Pcs::Commitment: AdditiveCommitment<F>,
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

        let unique_merged_polys = merged_polys
            .iter()
            .unique_by(|(_, poly)| addr_of!(*poly.deref()))
            .collect_vec();
        let unique_merged_poly_indices = unique_merged_polys
            .iter()
            .enumerate()
            .map(|(idx, (_, poly))| (addr_of!(*poly.deref()), idx))
            .collect::<HashMap<_, _>>();
        let expression = merged_polys
            .iter()
            .enumerate()
            .map(|(idx, (scalar, poly))| {
                let poly = unique_merged_poly_indices[&addr_of!(*poly.deref())];
                Expression::<F>::eq_xy(idx)
                    * Expression::Polynomial(Query::new(poly, Rotation::cur()))
                    * scalar
            })
            .sum();
        let virtual_poly = VirtualPolynomial::new(
            &expression,
            unique_merged_polys.iter().map(|(_, poly)| poly.deref()),
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
        end_timer(timer);

        let (g_prime_comm, g_prime_eval) = if cfg!(feature = "sanity-check") {
            let scalars = evals
                .iter()
                .zip(eq_xt.evals())
                .map(|(eval, eq_xt_i)| eq_xy_evals[eval.point()] * eq_xt_i)
                .collect_vec();
            let bases = evals.iter().map(|eval| comms[eval.poly()]);
            let comm = Pcs::Commitment::sum_with_scalar(&scalars, bases);
            (comm, g_prime.evaluate(&challenges))
        } else {
            (Pcs::Commitment::default(), F::ZERO)
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
        comms: Vec<&Pcs::Commitment>,
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
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
            let scalars = evals.iter().zip(eq_xt.evals()).fold(
                BTreeMap::<_, _>::new(),
                |mut scalars, (eval, eq_xt_i)| {
                    let rhs = eq_xy_evals[eval.point()] * eq_xt_i;
                    scalars
                        .entry(eval.poly())
                        .and_modify(|lhs| *lhs += rhs)
                        .or_insert(rhs);
                    scalars
                },
            );
            let (bases, scalars) = scalars
                .iter()
                .map(|(poly, scalar)| (comms[*poly], scalar))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            Pcs::Commitment::sum_with_scalar(scalars, bases)
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
            chain,
            transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
            Itertools,
        },
    };
    use rand::{rngs::OsRng, Rng};
    use std::iter;

    pub(super) fn run_commit_open_verify<F, Pcs, T>()
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        T: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
    {
        for num_vars in 3..16 {
            // Setup
            let (pp, vp) = {
                let mut rng = OsRng;
                let poly_size = 1 << num_vars;
                let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
                Pcs::trim(&param, poly_size, 1).unwrap()
            };
            // Commit and open
            let proof = {
                let mut transcript = T::new(());
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
                let mut transcript = T::from_proof((), proof.as_slice());
                Pcs::verify(
                    &vp,
                    &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
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
        T: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
    {
        for num_vars in 3..16 {
            let batch_size = 8;
            let num_points = batch_size >> 1;
            let mut rng = OsRng;
            // Setup
            let (pp, vp) = {
                let poly_size = 1 << num_vars;
                let param = Pcs::setup(poly_size, batch_size, &mut rng).unwrap();
                Pcs::trim(&param, poly_size, batch_size).unwrap()
            };
            // Batch commit and open
            let evals = chain![
                (0..num_points).map(|point| (0, point)),
                (0..batch_size).map(|poly| (poly, 0)),
                iter::repeat_with(|| (rng.gen_range(0..batch_size), rng.gen_range(0..num_points)))
                    .take(batch_size)
            ]
            .unique()
            .collect_vec();
            let proof = {
                let mut transcript = T::new(());
                let polys = iter::repeat_with(|| MultilinearPolynomial::rand(num_vars, OsRng))
                    .take(batch_size)
                    .collect_vec();
                let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();
                let points = iter::repeat_with(|| transcript.squeeze_challenges(num_vars))
                    .take(num_points)
                    .collect_vec();
                let evals = evals
                    .iter()
                    .copied()
                    .map(|(poly, point)| Evaluation {
                        poly,
                        point,
                        value: polys[poly].evaluate(&points[point]),
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
                let mut transcript = T::from_proof((), proof.as_slice());
                Pcs::batch_verify(
                    &vp,
                    &Pcs::read_commitments(&vp, batch_size, &mut transcript).unwrap(),
                    &iter::repeat_with(|| transcript.squeeze_challenges(num_vars))
                        .take(num_points)
                        .collect_vec(),
                    &evals
                        .iter()
                        .copied()
                        .zip(transcript.read_field_elements(evals.len()).unwrap())
                        .map(|((poly, point), eval)| Evaluation::new(poly, point, eval))
                        .collect_vec(),
                    &mut transcript,
                )
            };
            assert_eq!(result, Ok(()));
        }
    }
}
