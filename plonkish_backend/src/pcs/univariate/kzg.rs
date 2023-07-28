use crate::{
    pcs::{AdditiveCommitment, Evaluation, Point, PolynomialCommitmentScheme},
    poly::univariate::{CoefficientBasis, UnivariatePolynomial},
    util::{
        arithmetic::{
            barycentric_interpolate, barycentric_weights, fixed_base_msm, inner_product, powers,
            variable_base_msm, window_size, window_table, Curve, CurveAffine, Field,
            MultiMillerLoop, PrimeCurveAffine,
        },
        chain, izip, izip_eq,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{collections::BTreeSet, marker::PhantomData, ops::Neg, slice};

#[derive(Clone, Debug)]
pub struct UnivariateKzg<M: MultiMillerLoop>(PhantomData<M>);

impl<M: MultiMillerLoop> UnivariateKzg<M> {
    pub(crate) fn commit_coeffs(
        pp: &UnivariateKzgProverParam<M>,
        coeffs: &[M::Scalar],
    ) -> UnivariateKzgCommitment<M::G1Affine> {
        let comm = variable_base_msm(coeffs, &pp.powers_of_s_g1[..coeffs.len()]).into();
        UnivariateKzgCommitment(comm)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize",
    deserialize = "M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned",
))]
pub struct UnivariateKzgParam<M: MultiMillerLoop> {
    powers_of_s_g1: Vec<M::G1Affine>,
    powers_of_s_g2: Vec<M::G2Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgParam<M> {
    pub fn degree(&self) -> usize {
        self.powers_of_s_g1.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.powers_of_s_g1[0]
    }

    pub fn powers_of_s_g1(&self) -> &[M::G1Affine] {
        &self.powers_of_s_g1
    }

    pub fn g2(&self) -> M::G2Affine {
        self.powers_of_s_g2[0]
    }

    pub fn powers_of_s_g2(&self) -> &[M::G2Affine] {
        &self.powers_of_s_g2
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "M::G1Affine: Serialize",
    deserialize = "M::G1Affine: DeserializeOwned",
))]
pub struct UnivariateKzgProverParam<M: MultiMillerLoop> {
    powers_of_s_g1: Vec<M::G1Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgProverParam<M> {
    pub(crate) fn new(powers_of_s_g1: Vec<M::G1Affine>) -> Self {
        Self { powers_of_s_g1 }
    }

    pub fn degree(&self) -> usize {
        self.powers_of_s_g1.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.powers_of_s_g1[0]
    }

    pub fn powers_of_s_g1(&self) -> &[M::G1Affine] {
        &self.powers_of_s_g1
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateKzgCommitment<C: CurveAffine>(pub C);

impl<C: CurveAffine> Default for UnivariateKzgCommitment<C> {
    fn default() -> Self {
        Self(C::identity())
    }
}

impl<C: CurveAffine> PartialEq for UnivariateKzgCommitment<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<C: CurveAffine> Eq for UnivariateKzgCommitment<C> {}

impl<C: CurveAffine> AsRef<[C]> for UnivariateKzgCommitment<C> {
    fn as_ref(&self) -> &[C] {
        slice::from_ref(&self.0)
    }
}

impl<C: CurveAffine> AsRef<C> for UnivariateKzgCommitment<C> {
    fn as_ref(&self) -> &C {
        &self.0
    }
}

impl<C: CurveAffine> From<C> for UnivariateKzgCommitment<C> {
    fn from(comm: C) -> Self {
        Self(comm)
    }
}

impl<C: CurveAffine> AdditiveCommitment<C::Scalar> for UnivariateKzgCommitment<C> {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a C::Scalar> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
        assert_eq!(scalars.len(), bases.len());

        UnivariateKzgCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<M> PolynomialCommitmentScheme<M::Scalar> for UnivariateKzg<M>
where
    M: MultiMillerLoop,
    M::Scalar: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned,
    M::G2Affine: Serialize + DeserializeOwned,
{
    type Param = UnivariateKzgParam<M>;
    type ProverParam = UnivariateKzgProverParam<M>;
    type VerifierParam = UnivariateKzgVerifierParam<M>;
    type Polynomial = UnivariatePolynomial<M::Scalar, CoefficientBasis>;
    type Commitment = UnivariateKzgCommitment<M::G1Affine>;
    type CommitmentChunk = M::G1Affine;

    fn setup(poly_size: usize, _: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        let s = M::Scalar::random(rng);

        let g1 = M::G1Affine::generator();
        let powers_of_s_g1 = {
            let powers_of_s_g1 = powers(s).take(poly_size).collect_vec();
            let window_size = window_size(poly_size);
            let window_table = window_table(window_size, g1);
            let powers_of_s_projective =
                fixed_base_msm(window_size, &window_table, &powers_of_s_g1);

            let mut powers_of_s_g1 = vec![M::G1Affine::identity(); powers_of_s_projective.len()];
            parallelize(&mut powers_of_s_g1, |(powers_of_s_g1, starts)| {
                M::G1::batch_normalize(
                    &powers_of_s_projective[starts..(starts + powers_of_s_g1.len())],
                    powers_of_s_g1,
                );
            });
            powers_of_s_g1
        };

        let g2 = M::G2Affine::generator();
        let powers_of_s_g2 = {
            let powers_of_s_g2 = powers(s).take(poly_size).collect_vec();
            let window_size = window_size(poly_size);
            let window_table = window_table(window_size, g2);
            let powers_of_s_projective =
                fixed_base_msm(window_size, &window_table, &powers_of_s_g2);

            let mut powers_of_s_g2 = vec![M::G2Affine::identity(); powers_of_s_projective.len()];
            parallelize(&mut powers_of_s_g2, |(powers_of_s_g2, starts)| {
                M::G2::batch_normalize(
                    &powers_of_s_projective[starts..(starts + powers_of_s_g2.len())],
                    powers_of_s_g2,
                );
            });
            powers_of_s_g2
        };

        Ok(Self::Param {
            powers_of_s_g1,
            powers_of_s_g2,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        if param.powers_of_s_g1.len() < poly_size {
            return Err(Error::InvalidPcsParam(format!(
                "Too large poly_size to trim to (param supports poly_size up to {} but got {poly_size})",
                param.powers_of_s_g1.len(),
            )));
        }

        let powers_of_s_g1 = param.powers_of_s_g1[..poly_size].to_vec();
        let pp = Self::ProverParam { powers_of_s_g1 };
        let vp = Self::VerifierParam {
            g1: param.g1(),
            g2: param.g2(),
            s_g2: param.powers_of_s_g2[1],
        };
        Ok((pp, vp))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
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
    ) -> Result<Vec<Self::Commitment>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
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

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
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

        transcript.write_commitment(&Self::commit_coeffs(pp, quotient.coeffs()).0)?;

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
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
                let f = izip!(&powers_of_beta, set.polys.iter().map(|poly| polys[*poly]))
                    .sum::<UnivariatePolynomial<_, _>>();
                let (q, r) = f.div_rem(&vanishing_poly);
                (f, (q, r))
            })
            .unzip::<_, _, Vec<_>, (Vec<_>, Vec<_>)>();
        let q = izip_eq!(&powers_of_gamma, qs.iter()).sum::<UnivariatePolynomial<_, _>>();

        let q_comm = Self::commit_and_write(pp, &q, transcript)?;

        let z = transcript.squeeze_challenge();

        let (normalized_scalars, normalizer) = set_scalars(&sets, &powers_of_gamma, points, &z);
        let superset_eval = vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
        let q_scalar = -superset_eval * normalizer;
        let f = {
            let mut f = izip_eq!(&normalized_scalars, &fs).sum::<UnivariatePolynomial<_, _>>();
            f += (&q_scalar, &q);
            f
        };
        let (comm, eval) = if cfg!(feature = "sanity-check") {
            let comms = comms.into_iter().map(|comm| &comm.0).collect_vec();
            let scalars = comm_scalars(comms.len(), &sets, &powers_of_beta, &normalized_scalars);
            let comm = UnivariateKzgCommitment(
                variable_base_msm(chain![&scalars, [&q_scalar]], chain![comms, [&q_comm.0]]).into(),
            );
            let r_evals = rs.iter().map(|r| r.evaluate(&z)).collect_vec();
            (comm, inner_product(&normalized_scalars, &r_evals))
        } else {
            (UnivariateKzgCommitment::default(), M::Scalar::ZERO)
        };
        Self::open(pp, &f, &comm, &z, &eval, transcript)
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        transcript
            .read_commitments(num_polys)
            .map(|comms| comms.into_iter().map(UnivariateKzgCommitment).collect_vec())
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let pi = transcript.read_commitment()?;
        let c = (pi * point + comm.0 - vp.g1 * eval).into();
        M::pairings_product_is_identity(&[(&c, &(-vp.g2).into()), (&pi, &vp.s_g2.into())])
            .then_some(())
            .ok_or_else(|| Error::InvalidPcsOpen("Invalid univariate KZG open".to_string()))
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let comms = comms.into_iter().collect_vec();
        let (sets, superset) = eval_sets(evals);

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let q_comm = transcript.read_commitment()?;

        let z = transcript.squeeze_challenge();

        let max_set_len = sets.iter().map(|set| set.polys.len()).max().unwrap();
        let powers_of_beta = powers(beta).take(max_set_len).collect_vec();
        let powers_of_gamma = powers(gamma).take(sets.len()).collect_vec();

        let (normalized_scalars, normalizer) = set_scalars(&sets, &powers_of_gamma, points, &z);
        let f = {
            let comms = comms.iter().map(|comm| &comm.0).collect_vec();
            let scalars = comm_scalars(comms.len(), &sets, &powers_of_beta, &normalized_scalars);
            let superset_eval = vanishing_eval(superset.iter().map(|idx| &points[*idx]), &z);
            let q_scalar = -superset_eval * normalizer;
            UnivariateKzgCommitment(
                variable_base_msm(chain![&scalars, [&q_scalar]], chain![comms, [&q_comm]]).into(),
            )
        };
        let eval = inner_product(
            &normalized_scalars,
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

#[cfg(test)]
mod test {
    use crate::{
        pcs::{univariate::kzg::UnivariateKzg, Evaluation, PolynomialCommitmentScheme},
        util::{
            chain,
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
                Keccak256Transcript,
            },
            Itertools,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::{rngs::OsRng, Rng};
    use std::iter;

    type Pcs = UnivariateKzg<Bn256>;
    type Polynomial = <Pcs as PolynomialCommitmentScheme<Fr>>::Polynomial;

    #[test]
    fn commit_open_verify() {
        for k in 3..16 {
            // Setup
            let (pp, vp) = {
                let mut rng = OsRng;
                let poly_size = 1 << k;
                let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
                Pcs::trim(&param, poly_size, 1).unwrap()
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
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                Pcs::verify(
                    &vp,
                    &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
                    &transcript.squeeze_challenge(),
                    &transcript.read_field_element().unwrap(),
                    &mut transcript,
                )
            };
            assert_eq!(result, Ok(()));
        }
    }

    #[test]
    fn batch_commit_open_verify() {
        for k in 3..16 {
            let batch_size = 8;
            let num_points = batch_size >> 1;
            let mut rng = OsRng;
            // Setup
            let (pp, vp) = {
                let poly_size = 1 << k;
                let param = Pcs::setup(poly_size, batch_size, &mut rng).unwrap();
                Pcs::trim(&param, poly_size, batch_size).unwrap()
            };
            // Batch commit and open
            let evals = chain![
                (0..num_points).map(|point| (0, point)),
                (1..batch_size).map(|poly| (poly, 0)),
                iter::repeat_with(|| (rng.gen_range(0..batch_size), rng.gen_range(0..num_points)))
                    .take(batch_size)
            ]
            .unique()
            .collect_vec();
            let proof = {
                let mut transcript = Keccak256Transcript::default();
                let polys = iter::repeat_with(|| Polynomial::rand(pp.degree(), OsRng))
                    .take(batch_size)
                    .collect_vec();
                let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();
                let points = transcript.squeeze_challenges(num_points);
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
                let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
                Pcs::batch_verify(
                    &vp,
                    &Pcs::read_commitments(&vp, batch_size, &mut transcript).unwrap(),
                    &transcript.squeeze_challenges(num_points),
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
