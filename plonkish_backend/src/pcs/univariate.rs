use crate::{
    poly::univariate::{UnivariateBasis::*, UnivariatePolynomial},
    util::{
        arithmetic::{
            batch_projective_to_affine, radix2_fft, root_of_unity_inv, squares, CurveAffine, Field,
            PrimeField,
        },
        parallel::parallelize,
        Itertools,
    },
    Error,
};

mod hyrax;
pub(super) mod ipa;
mod kzg;

pub use hyrax::{
    UnivariateHyrax, UnivariateHyraxCommitment, UnivariateHyraxParam, UnivariateHyraxVerifierParam,
};
pub use ipa::{
    UnivariateIpa, UnivariateIpaCommitment, UnivariateIpaParam, UnivariateIpaVerifierParam,
};
pub use kzg::{
    UnivariateKzg, UnivariateKzgCommitment, UnivariateKzgParam, UnivariateKzgProverParam,
    UnivariateKzgVerifierParam,
};

fn monomial_g_to_lagrange_g<C: CurveAffine>(monomial_g: &[C]) -> Vec<C> {
    assert!(monomial_g.len().is_power_of_two());

    let k = monomial_g.len().ilog2() as usize;
    let n_inv = squares(C::Scalar::TWO_INV).nth(k).unwrap();
    let omega_inv = root_of_unity_inv(k);

    let mut lagrange = monomial_g.iter().map(C::to_curve).collect_vec();
    radix2_fft(&mut lagrange, omega_inv, k);
    parallelize(&mut lagrange, |(g, _)| {
        g.iter_mut().for_each(|g| *g *= n_inv)
    });

    batch_projective_to_affine(&lagrange)
}

fn validate_input<'a, F: Field>(
    function: &str,
    param_degree: usize,
    polys: impl IntoIterator<Item = &'a UnivariatePolynomial<F>>,
) -> Result<(), Error> {
    let polys = polys.into_iter().collect_vec();
    for poly in polys.iter() {
        match poly.basis() {
            Monomial => {
                if param_degree < poly.degree() {
                    return Err(err_too_large_deree(function, param_degree, poly.degree()));
                }
            }
            Lagrange => {
                if param_degree + 1 != poly.coeffs().len() {
                    return Err(err_invalid_evals_len(param_degree, poly.coeffs().len() - 1));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn err_too_large_deree(function: &str, upto: usize, got: usize) -> Error {
    Error::InvalidPcsParam(if function == "trim" {
        format!("Too large degree to {function} (param supports degree up to {upto} but got {got})")
    } else {
        format!(
            "Too large degree of poly to {function} (param supports degree up to {upto} but got {got})"
        )
    })
}

fn err_invalid_evals_len(expected: usize, got: usize) -> Error {
    Error::InvalidPcsParam(format!(
        "Invalid number of poly evaluations to commit (param needs {expected} evaluations but got {got})"
    ))
}

mod additive {
    use crate::{
        pcs::{Additive, Evaluation, Point, PolynomialCommitmentScheme},
        poly::univariate::UnivariatePolynomial,
        util::{
            arithmetic::{
                barycentric_interpolate, barycentric_weights, fe_to_bytes, inner_product, powers,
                Field, PrimeField,
            },
            chain, izip, izip_eq,
            transcript::{TranscriptRead, TranscriptWrite},
            Itertools,
        },
        Error,
    };
    use std::collections::BTreeSet;

    pub fn batch_open<F, Pcs>(
        pp: &Pcs::ProverParam,
        polys: Vec<&Pcs::Polynomial>,
        comms: Vec<&Pcs::Commitment>,
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
        Pcs::Commitment: Additive<F>,
    {
        if cfg!(feature = "sanity-check") {
            assert_eq!(
                points.iter().map(fe_to_bytes::<F>).unique().count(),
                points.len()
            );
            for eval in evals {
                let (poly, point) = (&polys[eval.poly()], &points[eval.point()]);
                assert_eq!(poly.evaluate(point), *eval.value());
            }
        }

        let (sets, superset) = eval_sets(evals);

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let max_set_len = sets.iter().map(|set| set.polys.len()).max().unwrap();
        let powers_of_beta = powers(beta).take(max_set_len).collect_vec();
        let powers_of_gamma = powers(gamma).take(sets.len()).collect_vec();
        let (fs, (qs, rs)) = sets
            .iter()
            .map(|set| {
                let vanishing_poly = set.vanishing_poly(points);
                let f = izip!(&powers_of_beta, set.polys.iter().map(|poly| polys[*poly]))
                    .sum::<UnivariatePolynomial<_>>();
                let (q, r) = f.div_rem(&vanishing_poly);
                (f, (q, r))
            })
            .unzip::<_, _, Vec<_>, (Vec<_>, Vec<_>)>();
        let q = izip_eq!(&powers_of_gamma, qs.iter()).sum::<UnivariatePolynomial<_>>();

        let q_comm = Pcs::commit_and_write(pp, &q, transcript)?;

        let z = transcript.squeeze_challenge();

        let (normalized_scalars, normalizer) = set_scalars(&sets, &powers_of_gamma, points, &z);
        let superset_eval = vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
        let q_scalar = -superset_eval * normalizer;
        let f = {
            let mut f = izip_eq!(&normalized_scalars, &fs).sum::<UnivariatePolynomial<_>>();
            f += (&q_scalar, &q);
            f
        };
        let (comm, eval) = if cfg!(feature = "sanity-check") {
            let scalars = comm_scalars(comms.len(), &sets, &powers_of_beta, &normalized_scalars);
            let comm =
                Pcs::Commitment::msm(chain![&scalars, [&q_scalar]], chain![comms, [&q_comm]]);
            let r_evals = rs.iter().map(|r| r.evaluate(&z)).collect_vec();
            (comm, inner_product(&normalized_scalars, &r_evals))
        } else {
            (Pcs::Commitment::default(), F::ZERO)
        };
        Pcs::open(pp, &f, &comm, &z, &eval, transcript)
    }

    pub fn batch_verify<F, Pcs>(
        vp: &Pcs::VerifierParam,
        comms: Vec<&Pcs::Commitment>,
        points: &[Point<F, Pcs::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
        Pcs::Commitment: Additive<F>,
    {
        let (sets, superset) = eval_sets(evals);

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let q_comm = Pcs::read_commitment(vp, transcript)?;

        let z = transcript.squeeze_challenge();

        let max_set_len = sets.iter().map(|set| set.polys.len()).max().unwrap();
        let powers_of_beta = powers(beta).take(max_set_len).collect_vec();
        let powers_of_gamma = powers(gamma).take(sets.len()).collect_vec();

        let (normalized_scalars, normalizer) = set_scalars(&sets, &powers_of_gamma, points, &z);
        let f = {
            let scalars = comm_scalars(comms.len(), &sets, &powers_of_beta, &normalized_scalars);
            let superset_eval = vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
            let q_scalar = -superset_eval * normalizer;
            Pcs::Commitment::msm(chain![&scalars, [&q_scalar]], chain![comms, [&q_comm]])
        };
        let eval = inner_product(
            &normalized_scalars,
            &sets
                .iter()
                .map(|set| set.r_eval(points, &z, &powers_of_beta))
                .collect_vec(),
        );
        Pcs::verify(vp, &f, &z, &eval, transcript)
    }

    #[derive(Debug)]
    struct EvaluationSet<F: Field> {
        polys: Vec<usize>,
        points: Vec<usize>,
        diffs: Vec<usize>,
        evals: Vec<Vec<F>>,
    }

    impl<F: Field> EvaluationSet<F> {
        fn vanishing_diff_eval(&self, points: &[F], z: &F) -> F {
            self.diffs
                .iter()
                .map(|idx| points[*idx])
                .fold(F::ONE, |eval, point| eval * (*z - point))
        }

        fn vanishing_poly(&self, points: &[F]) -> UnivariatePolynomial<F> {
            UnivariatePolynomial::vanishing(self.points.iter().map(|point| &points[*point]), F::ONE)
        }

        fn r_eval(&self, points: &[F], z: &F, powers_of_beta: &[F]) -> F {
            let points = self.points.iter().map(|idx| points[*idx]).collect_vec();
            let weights = barycentric_weights(&points);
            let r_evals = self
                .evals
                .iter()
                .map(|evals| barycentric_interpolate(&weights, &points, evals, z))
                .collect_vec();
            inner_product(&powers_of_beta[..r_evals.len()], &r_evals)
        }
    }

    fn eval_sets<F: Field>(evals: &[Evaluation<F>]) -> (Vec<EvaluationSet<F>>, BTreeSet<usize>) {
        let (poly_shifts, superset) = evals.iter().fold(
            (Vec::<(usize, Vec<usize>, Vec<F>)>::new(), BTreeSet::new()),
            |(mut poly_shifts, mut superset), eval| {
                if let Some(pos) = poly_shifts
                    .iter()
                    .position(|(poly, _, _)| *poly == eval.poly)
                {
                    let (_, points, evals) = &mut poly_shifts[pos];
                    if !points.contains(&eval.point) {
                        points.push(eval.point);
                        evals.push(*eval.value());
                    }
                } else {
                    poly_shifts.push((eval.poly, vec![eval.point], vec![*eval.value()]));
                }
                superset.insert(eval.point());
                (poly_shifts, superset)
            },
        );

        let sets = poly_shifts.into_iter().fold(
            Vec::<EvaluationSet<_>>::new(),
            |mut sets, (poly, points, evals)| {
                if let Some(pos) = sets.iter().position(|set| {
                    BTreeSet::from_iter(set.points.iter()) == BTreeSet::from_iter(points.iter())
                }) {
                    let set = &mut sets[pos];
                    if !set.polys.contains(&poly) {
                        set.polys.push(poly);
                        set.evals.push(
                            set.points
                                .iter()
                                .map(|lhs| {
                                    let idx = points.iter().position(|rhs| lhs == rhs).unwrap();
                                    evals[idx]
                                })
                                .collect(),
                        );
                    }
                } else {
                    let diffs = superset
                        .iter()
                        .filter(|idx| !points.contains(idx))
                        .copied()
                        .collect();
                    sets.push(EvaluationSet {
                        polys: vec![poly],
                        points,
                        diffs,
                        evals: vec![evals],
                    });
                }
                sets
            },
        );

        (sets, superset)
    }

    fn set_scalars<F: Field>(
        sets: &[EvaluationSet<F>],
        powers_of_gamma: &[F],
        points: &[F],
        z: &F,
    ) -> (Vec<F>, F) {
        let vanishing_diff_evals = sets
            .iter()
            .map(|set| set.vanishing_diff_eval(points, z))
            .collect_vec();
        // Adopt fflonk's trick to normalize the set scalars by the one of first set,
        // to save 1 EC scalar multiplication for verifier.
        let normalizer = vanishing_diff_evals[0].invert().unwrap_or(F::ONE);
        let normalized_scalars = izip_eq!(powers_of_gamma, &vanishing_diff_evals)
            .map(|(power_of_gamma, vanishing_diff_eval)| {
                normalizer * vanishing_diff_eval * power_of_gamma
            })
            .collect_vec();
        (normalized_scalars, normalizer)
    }

    fn vanishing_eval<'a, F: Field>(points: impl IntoIterator<Item = &'a F>, z: &F) -> F {
        points
            .into_iter()
            .fold(F::ONE, |eval, point| eval * (*z - point))
    }

    fn comm_scalars<F: Field>(
        num_polys: usize,
        sets: &[EvaluationSet<F>],
        powers_of_beta: &[F],
        normalized_scalars: &[F],
    ) -> Vec<F> {
        sets.iter().zip(normalized_scalars).fold(
            vec![F::ZERO; num_polys],
            |mut scalars, (set, coeff)| {
                izip!(&set.polys, powers_of_beta)
                    .for_each(|(poly, power_of_beta)| scalars[*poly] = *coeff * power_of_beta);
                scalars
            },
        )
    }
}
