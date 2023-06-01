use crate::{
    pcs::{AdditiveCommitment, Evaluation, Point, PolynomialCommitmentScheme},
    poly::univariate::{CoefficientBasis, UnivariatePolynomial},
    util::{
        arithmetic::{
            barycentric_interpolate, barycentric_weights, fixed_base_msm, inner_product, powers,
            variable_base_msm, window_size, window_table, Curve, Field, MultiMillerLoop,
            PrimeCurveAffine,
        },
        chain, izip_eq,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{collections::BTreeSet, marker::PhantomData, ops::Neg};

#[derive(Clone, Debug)]
pub struct UnivariateKzg<M: MultiMillerLoop>(PhantomData<M>);

impl<M: MultiMillerLoop> UnivariateKzg<M> {
    pub(crate) fn commit_coeffs(
        pp: &UnivariateKzgProverParam<M>,
        coeffs: &[M::Scalar],
    ) -> UnivariateKzgCommitment<M> {
        UnivariateKzgCommitment(variable_base_msm(coeffs, &pp.powers_of_s[..coeffs.len()]).into())
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgParam<M: MultiMillerLoop> {
    g1: M::G1Affine,
    powers_of_s: Vec<M::G1Affine>,
    g2: M::G2Affine,
    s_g2: M::G2Affine,
}

impl<M: MultiMillerLoop> UnivariateKzgParam<M> {
    pub fn degree(&self) -> usize {
        self.powers_of_s.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn powers_of_s(&self) -> &[M::G1Affine] {
        &self.powers_of_s
    }

    pub fn g2(&self) -> M::G2Affine {
        self.g2
    }

    pub fn s_g2(&self) -> M::G2Affine {
        self.s_g2
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgProverParam<M: MultiMillerLoop> {
    g1: M::G1Affine,
    powers_of_s: Vec<M::G1Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgProverParam<M> {
    pub fn degree(&self) -> usize {
        self.powers_of_s.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn powers_of_s(&self) -> &[M::G1Affine] {
        &self.powers_of_s
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgVerifierParam<M: MultiMillerLoop> {
    g1: M::G1Affine,
    g2: M::G2Affine,
    s_g2: M::G2Affine,
}

impl<M: MultiMillerLoop> UnivariateKzgVerifierParam<M> {
    pub fn g1(&self) -> M::G1Affine {
        self.g1
    }

    pub fn g2(&self) -> M::G2Affine {
        self.g2
    }

    pub fn s_g2(&self) -> M::G2Affine {
        self.s_g2
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgCommitment<M: MultiMillerLoop>(M::G1Affine);

impl<M: MultiMillerLoop> Default for UnivariateKzgCommitment<M> {
    fn default() -> Self {
        Self(M::G1Affine::identity())
    }
}

impl<M: MultiMillerLoop> AsRef<M::G1Affine> for UnivariateKzgCommitment<M> {
    fn as_ref(&self) -> &M::G1Affine {
        &self.0
    }
}

impl<M: MultiMillerLoop> AdditiveCommitment<M::Scalar> for UnivariateKzgCommitment<M> {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a M::Scalar> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(AsRef::as_ref).collect_vec();
        assert_eq!(scalars.len(), bases.len());

        UnivariateKzgCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<M: MultiMillerLoop> PolynomialCommitmentScheme<M::Scalar> for UnivariateKzg<M> {
    type Param = UnivariateKzgParam<M>;
    type ProverParam = UnivariateKzgProverParam<M>;
    type VerifierParam = UnivariateKzgVerifierParam<M>;
    type Polynomial = UnivariatePolynomial<M::Scalar, CoefficientBasis>;
    type Commitment = M::G1Affine;
    type CommitmentWithAux = UnivariateKzgCommitment<M>;

    fn setup(size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        let s = M::Scalar::random(rng);

        let g1 = M::G1Affine::generator();
        let powers_of_s = {
            let powers_of_s = powers(s).take(size).collect_vec();
            let window_size = window_size(size);
            let window_table = window_table(window_size, g1);
            let powers_of_s_projective = fixed_base_msm(window_size, &window_table, &powers_of_s);

            let mut powers_of_s = vec![M::G1Affine::identity(); powers_of_s_projective.len()];
            parallelize(&mut powers_of_s, |(powers_of_s, starts)| {
                M::G1::batch_normalize(
                    &powers_of_s_projective[starts..(starts + powers_of_s.len())],
                    powers_of_s,
                );
            });
            powers_of_s
        };

        let g2 = M::G2Affine::generator();
        let s_g2 = (g2 * s).into();

        Ok(Self::Param {
            g1,
            powers_of_s,
            g2,
            s_g2,
        })
    }

    fn trim(
        param: &Self::Param,
        size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        if param.powers_of_s.len() < size {
            return Err(Error::InvalidPcsParam(format!(
                "Too large size to trim to (param supports size up to {} but got {size})",
                param.powers_of_s.len(),
            )));
        }

        let powers_of_s = param.powers_of_s[..size].to_vec();
        let pp = Self::ProverParam {
            g1: param.g1,
            powers_of_s,
        };
        let vp = Self::VerifierParam {
            g1: param.powers_of_s[0],
            g2: param.g2,
            s_g2: param.s_g2,
        };
        Ok((pp, vp))
    }

    fn commit(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
    ) -> Result<Self::CommitmentWithAux, Error> {
        if pp.degree() < poly.degree() {
            return Err(Error::InvalidPcsParam(format!(
                "Too large degree of poly to commit (param supports degree up to {} but got {})",
                pp.degree(),
                poly.degree()
            )));
        }

        Ok(Self::commit_coeffs(pp, poly.coeffs()))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::CommitmentWithAux>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        _: &Self::CommitmentWithAux,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptWrite<M::G1Affine, M::Scalar>,
    ) -> Result<(), Error> {
        if pp.degree() < poly.degree() {
            return Err(Error::InvalidPcsParam(format!(
                "Too large degree of poly to open (param supports degree up to {} but got {})",
                pp.degree(),
                poly.degree()
            )));
        }

        let divisor = Self::Polynomial::new(vec![point.neg(), M::Scalar::ONE]);
        let (quotient, remainder) = poly.div_rem(&divisor);

        if cfg!(feature = "sanity-check") {
            if eval == &M::Scalar::ZERO {
                assert!(remainder.is_zero());
            } else {
                assert_eq!(&remainder[0], eval);
            }
        }

        transcript.write_commitment(Self::commit_coeffs(pp, quotient.coeffs()).as_ref())?;

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        _: impl IntoIterator<Item = &'a Self::CommitmentWithAux>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<M::G1Affine, M::Scalar>,
    ) -> Result<(), Error> {
        let polys = polys.into_iter().collect_vec();
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
                let f = izip_eq!(
                    powers_of_beta[..set.polys.len()].iter().rev(),
                    set.polys.iter().map(|poly| polys[*poly])
                )
                .sum::<UnivariatePolynomial<_, _>>();
                let (q, r) = f.div_rem(&vanishing_poly);
                (f, (q, r))
            })
            .unzip::<_, _, Vec<_>, (Vec<_>, Vec<_>)>();
        let q =
            izip_eq!(powers_of_gamma.iter().rev(), qs.iter()).sum::<UnivariatePolynomial<_, _>>();

        Self::commit_and_write(pp, &q, transcript)?;

        let z = transcript.squeeze_challenge();

        let set_coeffs = set_coeffs(&sets, &powers_of_gamma, points, &z);
        let f = {
            let mut f = izip_eq!(&set_coeffs, &fs).sum::<UnivariatePolynomial<_, _>>();
            let neg_superset_eval = -vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
            f += (&neg_superset_eval, &q);
            f
        };
        let eval = if cfg!(feature = "sanity-check") {
            let r_evals = rs.iter().map(|r| r.evaluate(&z)).collect_vec();
            inner_product(&set_coeffs, &r_evals)
        } else {
            M::Scalar::ZERO
        };
        Self::open(pp, &f, &Default::default(), &z, &eval, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<M::G1Affine, M::Scalar>,
    ) -> Result<(), Error> {
        let quotient = transcript.read_commitment()?;
        let lhs = (quotient * point + comm - vp.g1 * eval).into();
        let rhs = quotient;
        M::pairings_product_is_identity(&[(&lhs, &vp.g2.neg().into()), (&rhs, &vp.s_g2.into())])
            .then_some(())
            .ok_or_else(|| Error::InvalidPcsOpen("Invalid univariate KZG open".to_string()))
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        comms: &[Self::Commitment],
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<M::G1Affine, M::Scalar>,
    ) -> Result<(), Error> {
        let (sets, superset) = eval_sets(evals);

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let q = transcript.read_commitment()?;

        let z = transcript.squeeze_challenge();

        let max_set_len = sets.iter().map(|set| set.polys.len()).max().unwrap();
        let powers_of_beta = powers(beta).take(max_set_len).collect_vec();
        let powers_of_gamma = powers(gamma).take(sets.len()).collect_vec();

        let set_coeffs = set_coeffs(&sets, &powers_of_gamma, points, &z);
        let f = {
            let scalars = sets.iter().zip(&set_coeffs).fold(
                vec![M::Scalar::ZERO; comms.len()],
                |mut scalars, (set, coeff)| {
                    izip_eq!(&set.polys, powers_of_beta[..set.polys.len()].iter().rev())
                        .for_each(|(poly, power_of_beta)| scalars[*poly] = *coeff * power_of_beta);
                    scalars
                },
            );
            let neg_superset_eval = -vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
            variable_base_msm(chain![&scalars, [&neg_superset_eval]], chain![comms, [&q]]).into()
        };
        let eval = inner_product(
            &set_coeffs,
            &sets
                .iter()
                .map(|set| set.r_eval(points, &z, &powers_of_beta))
                .collect_vec(),
        );
        Self::verify(vp, &f, &z, &eval, transcript)
    }
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

    fn vanishing_poly(&self, points: &[F]) -> UnivariatePolynomial<F, CoefficientBasis> {
        UnivariatePolynomial::basis(self.points.iter().map(|point| &points[*point]), F::ONE)
    }

    fn r_eval(&self, points: &[F], z: &F, powers_of_beta: &[F]) -> F {
        let points = self.points.iter().map(|idx| points[*idx]).collect_vec();
        let weights = barycentric_weights(&points);
        let r_evals = self
            .evals
            .iter()
            .map(|evals| barycentric_interpolate(&weights, &points, evals, z))
            .collect_vec();
        inner_product(powers_of_beta[..r_evals.len()].iter().rev(), &r_evals)
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

fn set_coeffs<F: Field>(
    sets: &[EvaluationSet<F>],
    powers_of_gamma: &[F],
    points: &[F],
    z: &F,
) -> Vec<F> {
    izip_eq!(powers_of_gamma.iter().rev(), sets.iter())
        .map(|(power_of_gamma, set)| set.vanishing_diff_eval(points, z) * power_of_gamma)
        .collect_vec()
}

fn vanishing_eval<'a, F: Field>(points: impl IntoIterator<Item = &'a F>, z: &F) -> F {
    points
        .into_iter()
        .fold(F::ONE, |eval, point| eval * (*z - point))
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{univariate::kzg::UnivariateKzg, Evaluation, PolynomialCommitmentScheme},
        util::{
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscriptRead,
                InMemoryTranscriptWrite, Keccak256Transcript, TranscriptRead,
            },
            Itertools,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::rngs::OsRng;
    use std::iter;

    type Pcs = UnivariateKzg<Bn256>;
    type Polynomial = <Pcs as PolynomialCommitmentScheme<Fr>>::Polynomial;

    #[test]
    fn commit_open_verify() {
        // Setup
        let (pp, vp) = {
            let mut rng = OsRng;
            let size = 1 << 10;
            let param = Pcs::setup(size, &mut rng).unwrap();
            Pcs::trim(&param, size).unwrap()
        };
        // Commit and open
        let proof = {
            let mut transcript = Keccak256Transcript::default();
            let poly = Polynomial::rand(pp.degree(), OsRng);
            let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
            let point = transcript.squeeze_challenge();
            let eval = poly.evaluate(&point);
            transcript.write_field_element(&eval).unwrap();
            Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
            transcript.into_proof()
        };
        // Verify
        let result = {
            let mut transcript = Keccak256Transcript::from_proof(proof.as_slice());
            Pcs::verify(
                &vp,
                &transcript.read_commitment().unwrap(),
                &transcript.squeeze_challenge(),
                &transcript.read_field_element().unwrap(),
                &mut transcript,
            )
        };
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn batch_commit_open_verify() {
        // Setup
        let (pp, vp) = {
            let mut rng = OsRng;
            let size = 1 << 10;
            let param = Pcs::setup(size, &mut rng).unwrap();
            Pcs::trim(&param, size).unwrap()
        };
        // Batch commit and open
        let batch_size = 4;
        let proof = {
            let mut transcript = Keccak256Transcript::default();
            let polys = iter::repeat_with(|| Polynomial::rand(pp.degree(), OsRng))
                .take(batch_size)
                .collect_vec();
            let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();
            let points = transcript.squeeze_challenges(batch_size);
            let evals = polys
                .iter()
                .zip(points.iter())
                .enumerate()
                .map(|(idx, (poly, point))| Evaluation::new(idx, idx, poly.evaluate(point)))
                .collect_vec();
            transcript
                .write_field_elements(evals.iter().map(Evaluation::value))
                .unwrap();
            Pcs::batch_open(&pp, &polys, &comms, &points, &evals, &mut transcript).unwrap();
            transcript.into_proof()
        };
        // Batch verify
        let result = {
            let mut transcript = Keccak256Transcript::from_proof(proof.as_slice());
            Pcs::batch_verify(
                &vp,
                &transcript.read_commitments(batch_size).unwrap(),
                &transcript.squeeze_challenges(batch_size),
                &transcript
                    .read_field_elements(batch_size)
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .map(|(idx, eval)| Evaluation::new(idx, idx, eval))
                    .collect_vec(),
                &mut transcript,
            )
        };
        assert_eq!(result, Ok(()));
    }
}
