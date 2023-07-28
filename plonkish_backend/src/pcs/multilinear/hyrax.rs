use crate::{
    pcs::{
        multilinear::{
            additive, err_too_many_variates,
            ipa::{MultilinearIpa, MultilinearIpaCommitment, MultilinearIpaParams},
            validate_input,
        },
        AdditiveCommitment, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, variable_base_msm, Curve, CurveAffine, Group},
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};

use rand::RngCore;
use std::{borrow::Cow, iter, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct MultilinearHyrax<C: CurveAffine>(PhantomData<C>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearHyraxParams<C: CurveAffine> {
    num_vars: usize,
    batch_num_vars: usize,
    row_num_vars: usize,
    ipa: MultilinearIpaParams<C>,
}

impl<C: CurveAffine> MultilinearHyraxParams<C> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn batch_num_vars(&self) -> usize {
        self.batch_num_vars
    }

    pub fn row_num_vars(&self) -> usize {
        self.row_num_vars
    }

    pub fn row_len(&self) -> usize {
        1 << self.row_num_vars
    }

    pub fn num_chunks(&self) -> usize {
        1 << (self.num_vars - self.row_num_vars)
    }

    pub fn g(&self) -> &[C] {
        self.ipa.g()
    }

    pub fn h(&self) -> &C {
        self.ipa.h()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultilinearHyraxCommitment<C: CurveAffine>(pub Vec<C>);

impl<C: CurveAffine> Default for MultilinearHyraxCommitment<C> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<C: CurveAffine> AsRef<[C]> for MultilinearHyraxCommitment<C> {
    fn as_ref(&self) -> &[C] {
        &self.0
    }
}

// TODO: Batch all MSMs into one
impl<C: CurveAffine> AdditiveCommitment<C::Scalar> for MultilinearHyraxCommitment<C> {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a C::Scalar> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self {
        let (scalars, bases) = scalars
            .into_iter()
            .zip_eq(bases)
            .filter_map(|(scalar, bases)| (bases != &Self::default()).then_some((scalar, bases)))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let num_chunks = bases[0].0.len();
        for bases in bases.iter() {
            assert_eq!(bases.0.len(), num_chunks);
        }

        let mut output_projective = vec![C::CurveExt::identity(); num_chunks];
        parallelize(&mut output_projective, |(output, start)| {
            for (output, idx) in output.iter_mut().zip(start..) {
                *output = variable_base_msm(scalars.clone(), bases.iter().map(|base| &base.0[idx]))
            }
        });
        let mut output = vec![C::identity(); num_chunks];
        C::CurveExt::batch_normalize(&output_projective, &mut output);

        MultilinearHyraxCommitment(output)
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for MultilinearHyrax<C>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = MultilinearHyraxParams<C>;
    type ProverParam = MultilinearHyraxParams<C>;
    type VerifierParam = MultilinearHyraxParams<C>;
    type Polynomial = MultilinearPolynomial<C::Scalar>;
    type Commitment = MultilinearHyraxCommitment<C>;
    type CommitmentChunk = C;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        assert!(batch_size > 0 && batch_size <= poly_size);

        let num_vars = poly_size.ilog2() as usize;
        let batch_num_vars = (poly_size * batch_size).next_power_of_two().ilog2() as usize;
        let row_num_vars = div_ceil(batch_num_vars, 2);

        let ipa = MultilinearIpa::setup(1 << row_num_vars, 0, rng)?;

        Ok(Self::Param {
            num_vars,
            batch_num_vars,
            row_num_vars,
            ipa,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        assert!(batch_size > 0 && batch_size <= poly_size);

        let num_vars = poly_size.ilog2() as usize;
        let batch_num_vars = (poly_size * batch_size).next_power_of_two().ilog2() as usize;
        let row_num_vars = div_ceil(batch_num_vars, 2);
        if param.row_num_vars() < row_num_vars {
            return Err(err_too_many_variates(
                "trim",
                param.row_num_vars(),
                row_num_vars,
            ));
        }

        let (ipa, _) = MultilinearIpa::trim(&param.ipa, 1 << row_num_vars, 0)?;

        let param = Self::ProverParam {
            num_vars,
            batch_num_vars,
            row_num_vars,
            ipa,
        };
        Ok((param.clone(), param))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.row_len();
        let scalars = poly.evals();
        let comm_projective = {
            let mut comm = vec![C::CurveExt::identity(); pp.num_chunks()];
            parallelize(&mut comm, |(comm, start)| {
                for (comm, start) in comm.iter_mut().zip((start * row_len..).step_by(row_len)) {
                    *comm = variable_base_msm(&scalars[start..start + row_len], pp.g());
                }
            });
            comm
        };
        let mut comm = vec![C::identity(); pp.num_chunks()];
        C::CurveExt::batch_normalize(&comm_projective, &mut comm);

        Ok(MultilinearHyraxCommitment(comm))
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

        let scalars = polys
            .iter()
            .flat_map(|poly| poly.evals().chunks(pp.row_len()))
            .collect_vec();
        let comms_projective = {
            let mut comms = vec![C::CurveExt::identity(); scalars.len()];
            parallelize(&mut comms, |(comms, start)| {
                for (comm, scalars) in comms.iter_mut().zip(&scalars[start..]) {
                    *comm = variable_base_msm(*scalars, pp.g());
                }
            });
            comms
        };
        let mut comms = vec![C::identity(); scalars.len()];
        C::CurveExt::batch_normalize(&comms_projective, &mut comms);

        Ok(comms
            .into_iter()
            .chunks(pp.num_chunks())
            .into_iter()
            .map(|comm| MultilinearHyraxCommitment(comm.collect_vec()))
            .collect_vec())
    }

    // TODO: Batch all MSMs into one
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

        let (lo, hi) = point.split_at(pp.row_num_vars());
        let poly = if hi.is_empty() {
            Cow::Borrowed(poly)
        } else {
            Cow::Owned(poly.fix_last_vars(hi))
        };
        let comm = if cfg!(feature = "sanity-check") {
            MultilinearIpaCommitment(if hi.is_empty() {
                assert_eq!(comm.0.len(), 1);
                comm.0[0]
            } else {
                let scalars = MultilinearPolynomial::eq_xy(hi).into_evals();
                variable_base_msm(&scalars, &comm.0).into()
            })
        } else {
            MultilinearIpaCommitment::default()
        };

        MultilinearIpa::open(&pp.ipa, &poly, &comm, &lo.to_vec(), eval, transcript)
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
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let comms = iter::repeat_with(|| {
            transcript
                .read_commitments(vp.num_chunks())
                .map(MultilinearHyraxCommitment)
        })
        .take(num_polys)
        .try_collect()?;
        Ok(comms)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
    ) -> Result<(), Error> {
        assert_eq!(comm.0.len(), vp.num_chunks());

        let (lo, hi) = point.split_at(vp.row_num_vars());
        let comm = {
            MultilinearIpaCommitment(if hi.is_empty() {
                assert_eq!(vp.num_chunks(), 1);
                comm.0[0]
            } else {
                let scalars = MultilinearPolynomial::eq_xy(hi).into_evals();
                variable_base_msm(&scalars, &comm.0).into()
            })
        };

        MultilinearIpa::verify(&vp.ipa, &comm, &lo.to_vec(), eval, transcript)
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
        pcs::multilinear::{
            hyrax::MultilinearHyrax,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::pasta::pallas::Affine;

    type Pcs = MultilinearHyrax<Affine>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
