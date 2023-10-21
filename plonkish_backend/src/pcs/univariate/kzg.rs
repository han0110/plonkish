use crate::{
    pcs::{
        univariate::{additive, err_too_large_deree, monomial_g_to_lagrange_g, validate_input},
        Additive, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::univariate::{UnivariateBasis::*, UnivariatePolynomial},
    util::{
        arithmetic::{
            batch_projective_to_affine, fixed_base_msm, powers, radix2_fft, root_of_unity_inv,
            variable_base_msm, window_size, window_table, Curve, CurveAffine, Field,
            MultiMillerLoop, PrimeCurveAffine, PrimeField,
        },
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{marker::PhantomData, ops::Neg, slice};

#[derive(Clone, Debug)]
pub struct UnivariateKzg<M: MultiMillerLoop>(PhantomData<M>);

impl<M: MultiMillerLoop> UnivariateKzg<M> {
    pub(crate) fn commit_monomial(
        pp: &UnivariateKzgProverParam<M>,
        coeffs: &[M::Scalar],
    ) -> UnivariateKzgCommitment<M::G1Affine> {
        let comm = variable_base_msm(coeffs, &pp.monomial_g1[..coeffs.len()]).into();
        UnivariateKzgCommitment(comm)
    }

    pub(crate) fn commit_lagrange(
        pp: &UnivariateKzgProverParam<M>,
        evals: &[M::Scalar],
    ) -> UnivariateKzgCommitment<M::G1Affine> {
        let comm = variable_base_msm(evals, &pp.lagrange_g1[..evals.len()]).into();
        UnivariateKzgCommitment(comm)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "M::G1Affine: Serialize, M::G2Affine: Serialize",
    deserialize = "M::G1Affine: DeserializeOwned, M::G2Affine: DeserializeOwned",
))]
pub struct UnivariateKzgParam<M: MultiMillerLoop> {
    k: usize,
    monomial_g1: Vec<M::G1Affine>,
    lagrange_g1: Vec<M::G1Affine>,
    powers_of_s_g2: Vec<M::G2Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgParam<M> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.monomial_g1.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.monomial_g1[0]
    }

    pub fn monomial_g1(&self) -> &[M::G1Affine] {
        &self.monomial_g1
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
    k: usize,
    monomial_g1: Vec<M::G1Affine>,
    lagrange_g1: Vec<M::G1Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgProverParam<M> {
    pub(crate) fn new(
        k: usize,
        monomial_g1: Vec<M::G1Affine>,
        lagrange_g1: Vec<M::G1Affine>,
    ) -> Self {
        Self {
            k,
            monomial_g1,
            lagrange_g1,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.monomial_g1.len() - 1
    }

    pub fn g1(&self) -> M::G1Affine {
        self.monomial_g1[0]
    }

    pub fn monomial_g1(&self) -> &[M::G1Affine] {
        &self.monomial_g1
    }

    pub fn lagrange_g1(&self) -> &[M::G1Affine] {
        &self.lagrange_g1
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

impl<C: CurveAffine> Additive<C::Scalar> for UnivariateKzgCommitment<C> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a C::Scalar>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
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
    type Polynomial = UnivariatePolynomial<M::Scalar>;
    type Commitment = UnivariateKzgCommitment<M::G1Affine>;
    type CommitmentChunk = M::G1Affine;

    fn setup(poly_size: usize, _: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        // TODO: Support arbitrary degree.
        assert!(poly_size.is_power_of_two());
        assert!(poly_size.ilog2() <= M::Scalar::S);

        let s = M::Scalar::random(rng);

        let g1 = M::G1Affine::generator();
        let (monomial_g1, lagrange_g1) = {
            let window_size = window_size(poly_size);
            let window_table = window_table(window_size, g1);
            let monomial = powers(s).take(poly_size).collect_vec();
            let monomial_g1 =
                batch_projective_to_affine(&fixed_base_msm(window_size, &window_table, &monomial));
            let lagrange_g1 = {
                let k = poly_size.ilog2() as usize;
                let n_inv = M::Scalar::TWO_INV.pow_vartime([k as u64]);
                let mut lagrange = monomial;
                radix2_fft(&mut lagrange, root_of_unity_inv(k), k);
                lagrange.iter_mut().for_each(|v| *v *= n_inv);
                batch_projective_to_affine(&fixed_base_msm(window_size, &window_table, &lagrange))
            };
            (monomial_g1, lagrange_g1)
        };

        let g2 = M::G2Affine::generator();
        let powers_of_s_g2 = {
            let powers_of_s_g2 = powers(s).take(poly_size).collect_vec();
            let window_size = window_size(poly_size);
            let window_table = window_table(window_size, g2);
            batch_projective_to_affine(&fixed_base_msm(window_size, &window_table, &powers_of_s_g2))
        };

        Ok(Self::Param {
            k: poly_size.ilog2() as usize,
            monomial_g1,
            lagrange_g1,
            powers_of_s_g2,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());

        if param.monomial_g1.len() < poly_size {
            return Err(err_too_large_deree("trim", param.degree(), poly_size - 1));
        }

        let monomial_g1 = param.monomial_g1[..poly_size].to_vec();
        let lagrange_g1 = if param.lagrange_g1.len() == poly_size {
            param.lagrange_g1.clone()
        } else {
            monomial_g_to_lagrange_g(&monomial_g1)
        };

        let pp = Self::ProverParam::new(poly_size.ilog2() as usize, monomial_g1, lagrange_g1);
        let vp = Self::VerifierParam {
            g1: param.g1(),
            g2: param.g2(),
            s_g2: param.powers_of_s_g2[1],
        };
        Ok((pp, vp))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.degree(), [poly])?;

        match poly.basis() {
            Monomial => Ok(Self::commit_monomial(pp, poly.coeffs())),
            Lagrange => Ok(Self::commit_lagrange(pp, poly.coeffs())),
        }
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
        assert_eq!(poly.basis(), Monomial);

        validate_input("open", pp.degree(), [poly])?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let divisor = Self::Polynomial::monomial(vec![point.neg(), M::Scalar::ONE]);
        let (quotient, remainder) = poly.div_rem(&divisor);

        if cfg!(feature = "sanity-check") {
            if eval == &M::Scalar::ZERO {
                assert!(remainder.is_empty());
            } else {
                assert_eq!(&remainder[0], eval);
            }
        }

        transcript.write_commitment(&Self::commit_monomial(pp, quotient.coeffs()).0)?;

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
        let comms = comms.into_iter().collect_vec();
        validate_input("batch open", pp.degree(), polys.clone())?;
        additive::batch_open::<_, Self>(pp, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let comms = transcript.read_commitments(num_polys)?;
        Ok(comms.into_iter().map(UnivariateKzgCommitment).collect())
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
        additive::batch_verify::<_, Self>(vp, comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::kzg::UnivariateKzg,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = UnivariateKzg<Bn256>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
