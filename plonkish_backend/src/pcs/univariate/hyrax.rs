use crate::{
    pcs::{
        univariate::{
            additive, err_too_large_deree,
            ipa::{
                UnivariateIpa, UnivariateIpaCommitment, UnivariateIpaParam,
                UnivariateIpaVerifierParam,
            },
            validate_input,
        },
        Additive, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::univariate::{UnivariateBasis::*, UnivariatePolynomial},
    util::{
        arithmetic::{
            batch_projective_to_affine, div_ceil, powers, squares, variable_base_msm, CurveAffine,
            Field, Group,
        },
        chain, izip,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::Cow, iter, marker::PhantomData};

#[derive(Clone, Debug)]
pub struct UnivariateHyrax<C: CurveAffine>(PhantomData<C>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateHyraxParam<C: CurveAffine> {
    k: usize,
    batch_k: usize,
    row_k: usize,
    ipa: UnivariateIpaParam<C>,
}

impl<C: CurveAffine> UnivariateHyraxParam<C> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        (1 << self.k) - 1
    }

    pub fn batch_k(&self) -> usize {
        self.batch_k
    }

    pub fn row_k(&self) -> usize {
        self.row_k
    }

    pub fn row_len(&self) -> usize {
        1 << self.row_k
    }

    pub fn num_chunks(&self) -> usize {
        1 << (self.k - self.row_k)
    }

    pub fn monomial(&self) -> &[C] {
        self.ipa.monomial()
    }

    pub fn lagrange(&self) -> &[C] {
        self.ipa.lagrange()
    }

    pub fn h(&self) -> &C {
        self.ipa.h()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateHyraxVerifierParam<C: CurveAffine> {
    k: usize,
    batch_k: usize,
    row_k: usize,
    ipa: UnivariateIpaVerifierParam<C>,
}

impl<C: CurveAffine> UnivariateHyraxVerifierParam<C> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn row_k(&self) -> usize {
        self.row_k
    }

    pub fn num_chunks(&self) -> usize {
        1 << (self.k - self.row_k)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnivariateHyraxCommitment<C: CurveAffine>(pub Vec<C>);

impl<C: CurveAffine> Default for UnivariateHyraxCommitment<C> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<C: CurveAffine> AsRef<[C]> for UnivariateHyraxCommitment<C> {
    fn as_ref(&self) -> &[C] {
        &self.0
    }
}

// TODO: Batch all MSMs into one
impl<C: CurveAffine> Additive<C::Scalar> for UnivariateHyraxCommitment<C> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a C::Scalar>,
        bases: impl IntoIterator<Item = &'b Self>,
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

        let mut output = vec![C::CurveExt::identity(); num_chunks];
        parallelize(&mut output, |(output, start)| {
            for (output, idx) in output.iter_mut().zip(start..) {
                *output = variable_base_msm(scalars.clone(), bases.iter().map(|base| &base.0[idx]))
            }
        });
        UnivariateHyraxCommitment(batch_projective_to_affine(&output))
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for UnivariateHyrax<C>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = UnivariateHyraxParam<C>;
    type ProverParam = UnivariateHyraxParam<C>;
    type VerifierParam = UnivariateHyraxVerifierParam<C>;
    type Polynomial = UnivariatePolynomial<C::Scalar>;
    type Commitment = UnivariateHyraxCommitment<C>;
    type CommitmentChunk = C;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        // TODO: Support arbitrary degree.
        assert!(poly_size.is_power_of_two());
        assert!(batch_size > 0 && batch_size <= poly_size);

        let k = poly_size.ilog2() as usize;
        let batch_k = (poly_size * batch_size).next_power_of_two().ilog2() as usize;
        let row_k = div_ceil(batch_k, 2);

        let ipa = UnivariateIpa::setup(1 << row_k, 0, rng)?;

        Ok(Self::Param {
            k,
            batch_k,
            row_k,
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

        let k = poly_size.ilog2() as usize;
        let batch_k = (poly_size * batch_size).next_power_of_two().ilog2() as usize;
        let row_k = div_ceil(batch_k, 2);
        if param.row_k() < row_k {
            return Err(err_too_large_deree("trim", param.degree(), poly_size - 1));
        }

        let (ipa_pp, ipa_vp) = UnivariateIpa::trim(&param.ipa, 1 << row_k, 0)?;

        let pp = Self::ProverParam {
            k,
            batch_k,
            row_k,
            ipa: ipa_pp,
        };
        let vp = Self::VerifierParam {
            k,
            batch_k,
            row_k,
            ipa: ipa_vp,
        };
        Ok((pp, vp))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.degree(), [poly])?;

        let bases = match poly.basis() {
            Monomial => pp.monomial(),
            Lagrange => pp.lagrange(),
        };

        let row_len = pp.row_len();
        let scalars = poly.coeffs();
        let comm = {
            let mut comm = vec![C::CurveExt::identity(); pp.num_chunks()];
            parallelize(&mut comm, |(comm, start)| {
                for (comm, offset) in comm.iter_mut().zip((start * row_len..).step_by(row_len)) {
                    let row = &scalars[offset..(offset + row_len).min(scalars.len())];
                    *comm = variable_base_msm(row, &bases[..row.len()]);
                }
            });
            batch_projective_to_affine(&comm)
        };

        Ok(UnivariateHyraxCommitment(comm))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let polys = polys.into_iter().collect_vec();
        if polys.is_empty() {
            return Ok(Vec::new());
        }
        validate_input("batch commit", pp.degree(), polys.iter().copied())?;

        let row_len = pp.row_len();
        let scalars = polys
            .iter()
            .flat_map(|poly| {
                chain![poly.coeffs().chunks(row_len), iter::repeat([].as_slice())]
                    .take(pp.num_chunks())
            })
            .collect_vec();
        let comms = {
            let mut comms = vec![C::CurveExt::identity(); scalars.len()];
            parallelize(&mut comms, |(comms, start)| {
                for (comm, row) in comms.iter_mut().zip(&scalars[start..]) {
                    *comm = variable_base_msm(*row, &pp.monomial()[..row.len()]);
                }
            });
            batch_projective_to_affine(&comms)
        };

        Ok(comms
            .into_iter()
            .chunks(pp.num_chunks())
            .into_iter()
            .map(|comm| UnivariateHyraxCommitment(comm.collect_vec()))
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
        assert_eq!(poly.basis(), Monomial);

        validate_input("open", pp.degree(), [poly])?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(comm.0.len(), pp.num_chunks());
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let row_len = pp.row_len();
        let scalars = powers(squares(*point).nth(pp.row_k()).unwrap())
            .take(pp.num_chunks())
            .collect_vec();
        let poly = if pp.num_chunks() == 1 {
            Cow::Borrowed(poly)
        } else {
            let mut coeffs = vec![C::Scalar::ZERO; row_len];
            if let Some(row) = poly.coeffs().chunks(row_len).next() {
                coeffs[..row.len()].copy_from_slice(row);
            }
            izip!(&scalars, poly.coeffs().chunks(row_len))
                .skip(1)
                .for_each(|(scalar, row)| {
                    parallelize(&mut coeffs, |(coeffs, start)| {
                        let scalar = *scalar;
                        izip!(coeffs, &row[start..]).for_each(|(lhs, rhs)| *lhs += scalar * rhs)
                    });
                });
            Cow::Owned(UnivariatePolynomial::monomial(coeffs))
        };
        let comm = if cfg!(feature = "sanity-check") {
            UnivariateIpaCommitment(if pp.num_chunks() == 1 {
                comm.0[0]
            } else {
                variable_base_msm(&scalars, &comm.0).into()
            })
        } else {
            UnivariateIpaCommitment::default()
        };

        UnivariateIpa::open(&pp.ipa, &poly, &comm, point, eval, transcript)
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
        additive::batch_open::<_, Self>(pp, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        iter::repeat_with(|| {
            transcript
                .read_commitments(vp.num_chunks())
                .map(UnivariateHyraxCommitment)
        })
        .take(num_polys)
        .collect()
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
    ) -> Result<(), Error> {
        assert_eq!(comm.0.len(), vp.num_chunks());

        let comm = {
            UnivariateIpaCommitment(if vp.num_chunks() == 1 {
                comm.0[0]
            } else {
                let scalars = powers(squares(*point).nth(vp.row_k()).unwrap())
                    .take(vp.num_chunks())
                    .collect_vec();
                variable_base_msm(&scalars, &comm.0).into()
            })
        };

        UnivariateIpa::verify(&vp.ipa, &comm, point, eval, transcript)
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<C::Scalar, Self::Polynomial>],
        evals: &[Evaluation<C::Scalar>],
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
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
            univariate::hyrax::UnivariateHyrax,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::pasta::pallas::Affine;

    type Pcs = UnivariateHyrax<Affine>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
