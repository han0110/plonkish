use crate::{
    pcs::{
        multilinear::{additive, err_too_many_variates, validate_input},
        univariate::ipa::{prove_bulletproof_reduction, verify_bulletproof_reduction},
        Additive, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{
            batch_projective_to_affine, variable_base_msm, Curve, CurveAffine, CurveExt, Group,
        },
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Either, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{marker::PhantomData, slice};

#[derive(Clone, Debug)]
pub struct MultilinearIpa<C: CurveAffine>(PhantomData<C>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearIpaParam<C: CurveAffine> {
    num_vars: usize,
    g: Vec<C>,
    h: C,
}

impl<C: CurveAffine> MultilinearIpaParam<C> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn g(&self) -> &[C] {
        &self.g
    }

    pub fn h(&self) -> &C {
        &self.h
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultilinearIpaCommitment<C: CurveAffine>(pub C);

impl<C: CurveAffine> Default for MultilinearIpaCommitment<C> {
    fn default() -> Self {
        Self(C::identity())
    }
}

impl<C: CurveAffine> AsRef<[C]> for MultilinearIpaCommitment<C> {
    fn as_ref(&self) -> &[C] {
        slice::from_ref(&self.0)
    }
}

impl<C: CurveAffine> AsRef<C> for MultilinearIpaCommitment<C> {
    fn as_ref(&self) -> &C {
        &self.0
    }
}

impl<C: CurveAffine> From<C> for MultilinearIpaCommitment<C> {
    fn from(comm: C) -> Self {
        Self(comm)
    }
}

impl<C: CurveAffine> Additive<C::Scalar> for MultilinearIpaCommitment<C> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a C::Scalar>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
        MultilinearIpaCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for MultilinearIpa<C>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = MultilinearIpaParam<C>;
    type ProverParam = MultilinearIpaParam<C>;
    type VerifierParam = MultilinearIpaParam<C>;
    type Polynomial = MultilinearPolynomial<C::Scalar>;
    type Commitment = MultilinearIpaCommitment<C>;
    type CommitmentChunk = C;

    fn setup(poly_size: usize, _: usize, _: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        let g = {
            let mut g = vec![C::Curve::identity(); poly_size];
            parallelize(&mut g, |(g, start)| {
                let hasher = C::CurveExt::hash_to_curve("MultilinearIpa::setup");
                for (g, idx) in g.iter_mut().zip(start as u32..) {
                    let mut message = [0u8; 5];
                    message[1..5].copy_from_slice(&idx.to_le_bytes());
                    *g = hasher(&message);
                }
            });
            batch_projective_to_affine(&g)
        };

        let hasher = C::CurveExt::hash_to_curve("MultilinearIpa::setup");
        let h = hasher(&[1]).to_affine();

        Ok(Self::Param { num_vars, g, h })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        if param.num_vars() < num_vars {
            return Err(err_too_many_variates("trim", param.num_vars(), num_vars));
        }
        let param = Self::ProverParam {
            num_vars,
            g: param.g[..poly_size].to_vec(),
            h: param.h,
        };
        Ok((param.clone(), param))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        Ok(variable_base_msm(poly.evals(), pp.g()).into()).map(MultilinearIpaCommitment)
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let polys = polys.into_iter().collect_vec();
        if polys.is_empty() {
            return Ok(Vec::new());
        }
        validate_input("batch commit", pp.num_vars(), polys.iter().copied(), None)?;

        Ok(polys
            .iter()
            .map(|poly| variable_base_msm(poly.evals(), pp.g()).into())
            .map(MultilinearIpaCommitment)
            .collect())
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptWrite<C, C::Scalar>,
    ) -> Result<(), Error> {
        validate_input("open", pp.num_vars(), [poly], [point])?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let bases = pp.g();
        let coeffs = poly.evals();
        let zs = MultilinearPolynomial::eq_xy(point).into_evals();
        prove_bulletproof_reduction(bases, pp.h(), coeffs, zs, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<C::Scalar, Self::Polynomial>],
        evals: &[Evaluation<C::Scalar>],
        transcript: &mut impl TranscriptWrite<C, C::Scalar>,
    ) -> Result<(), Error> {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        additive::batch_open::<_, Self>(pp, pp.num_vars(), polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        transcript.read_commitments(num_polys).map(|comms| {
            comms
                .into_iter()
                .map(MultilinearIpaCommitment)
                .collect_vec()
        })
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
    ) -> Result<(), Error> {
        let bases = vp.g();
        let point = Either::Right(point.as_slice());
        verify_bulletproof_reduction(bases, vp.h(), comm, point, eval, transcript)
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<C::Scalar, Self::Polynomial>],
        evals: &[Evaluation<C::Scalar>],
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
    ) -> Result<(), Error> {
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, vp.num_vars(), comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::ipa::MultilinearIpa,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::pasta::pallas::Affine;

    type Pcs = MultilinearIpa<Affine>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
