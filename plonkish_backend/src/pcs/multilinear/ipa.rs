use crate::{
    pcs::{
        multilinear::{additive, err_too_many_variates, validate_input},
        AdditiveCommitment, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{
            inner_product, variable_base_msm, Curve, CurveAffine, CurveExt, Field, Group,
        },
        chain,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::group::ff::BatchInvert;
use rand::RngCore;
use std::{iter, marker::PhantomData, slice};

#[derive(Clone, Debug)]
pub struct MultilinearIpa<C: CurveAffine>(PhantomData<C>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearIpaParams<C: CurveAffine> {
    num_vars: usize,
    g: Vec<C>,
    h: C,
}

impl<C: CurveAffine> MultilinearIpaParams<C> {
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

impl<C: CurveAffine> AdditiveCommitment<C::Scalar> for MultilinearIpaCommitment<C> {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a C::Scalar> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
        assert_eq!(scalars.len(), bases.len());

        MultilinearIpaCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for MultilinearIpa<C>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = MultilinearIpaParams<C>;
    type ProverParam = MultilinearIpaParams<C>;
    type VerifierParam = MultilinearIpaParams<C>;
    type Polynomial = MultilinearPolynomial<C::Scalar>;
    type Commitment = MultilinearIpaCommitment<C>;
    type CommitmentChunk = C;

    fn setup(poly_size: usize, _: usize, _: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        let g_projective = {
            let mut g = vec![C::Curve::identity(); poly_size];
            parallelize(&mut g, |(g, start)| {
                let hasher = C::CurveExt::hash_to_curve("MultilinearIpa::setup");
                for (g, idx) in g.iter_mut().zip(start as u32..) {
                    let mut message = [0u8; 5];
                    message[1..5].copy_from_slice(&idx.to_le_bytes());
                    *g = hasher(&message);
                }
            });
            g
        };

        let g = {
            let mut g = vec![C::identity(); poly_size];
            parallelize(&mut g, |(g, start)| {
                C::Curve::batch_normalize(&g_projective[start..(start + g.len())], g);
            });
            g
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

        let xi_0 = transcript.squeeze_challenge();
        let h_prime = (pp.h * xi_0).to_affine();

        let mut bases = pp.g().to_vec();
        let mut coeffs = poly.evals().to_vec();
        let mut zs = MultilinearPolynomial::eq_xy(point).into_evals();

        for i in 0..pp.num_vars() {
            let mid = 1 << (pp.num_vars() - i - 1);

            let (bases_l, bases_r) = bases.split_at(mid);
            let (coeffs_l, coeffs_r) = coeffs.split_at(mid);
            let (zs_l, zs_r) = zs.split_at(mid);
            let (c_l, c_r) = (inner_product(coeffs_r, zs_l), inner_product(coeffs_l, zs_r));
            let l_i = variable_base_msm(
                chain![coeffs_r, Some(&c_l)],
                chain![bases_l, Some(&h_prime)],
            );
            let r_i = variable_base_msm(
                chain![coeffs_l, Some(&c_r)],
                chain![bases_r, Some(&h_prime)],
            );
            transcript.write_commitment(&l_i.to_affine())?;
            transcript.write_commitment(&r_i.to_affine())?;

            let xi_i = transcript.squeeze_challenge();
            let xi_i_inv = xi_i.invert().unwrap();

            let (bases_l, bases_r) = bases.split_at_mut(mid);
            let (coeffs_l, coeffs_r) = coeffs.split_at_mut(mid);
            let (zs_l, zs_r) = zs.split_at_mut(mid);
            parallelize(bases_l, |(bases_l, start)| {
                let mut tmp = Vec::with_capacity(bases_l.len());
                for (lhs, rhs) in bases_l.iter().zip(bases_r[start..].iter()) {
                    tmp.push(lhs.to_curve() + *rhs * xi_i);
                }
                C::Curve::batch_normalize(&tmp, bases_l);
            });
            parallelize(coeffs_l, |(coeffs_l, start)| {
                for (lhs, rhs) in coeffs_l.iter_mut().zip(coeffs_r[start..].iter()) {
                    *lhs += xi_i_inv * rhs;
                }
            });
            parallelize(zs_l, |(zs_l, start)| {
                for (lhs, rhs) in zs_l.iter_mut().zip(zs_r[start..].iter()) {
                    *lhs += xi_i * rhs;
                }
            });
            bases.truncate(mid);
            coeffs.truncate(mid);
            zs.truncate(mid);
        }

        transcript.write_commitment(&bases[0])?;
        transcript.write_field_element(&coeffs[0])?;

        Ok(())
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
        validate_input("verify", vp.num_vars(), [], [point])?;

        let xi_0 = transcript.squeeze_challenge();

        let (ls, rs, xis) = iter::repeat_with(|| {
            Ok((
                transcript.read_commitment()?,
                transcript.read_commitment()?,
                transcript.squeeze_challenge(),
            ))
        })
        .take(vp.num_vars())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
        let g_k = transcript.read_commitment()?;
        let c = transcript.read_field_element()?;

        let xi_invs = {
            let mut xi_invs = xis.clone();
            xi_invs.iter_mut().batch_invert();
            xi_invs
        };
        let eval_prime = xi_0 * eval;
        let c_k = variable_base_msm(
            chain![&xi_invs, &xis, Some(&eval_prime)],
            chain![&ls, &rs, Some(vp.h())],
        ) + comm.0;
        let h = MultilinearPolynomial::new(h_coeffs(&xis));

        (c_k == variable_base_msm(&[c, c * h.evaluate(point) * xi_0], [&g_k, vp.h()])
            && g_k == variable_base_msm(h.evals(), vp.g()).to_affine())
        .then_some(())
        .ok_or_else(|| Error::InvalidPcsOpen("Invalid multilinear IPA open".to_string()))
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

fn h_coeffs<F: Field>(xi: &[F]) -> Vec<F> {
    assert!(!xi.is_empty());

    let mut coeffs = vec![F::ZERO; 1 << xi.len()];
    coeffs[0] = F::ONE;

    for (len, xi) in xi.iter().rev().enumerate().map(|(i, xi)| (1 << i, xi)) {
        let (left, right) = coeffs.split_at_mut(len);
        let right = &mut right[0..len];
        right.copy_from_slice(left);
        parallelize(right, |(right, _)| {
            for coeff in right {
                *coeff *= xi;
            }
        });
    }

    coeffs
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::multilinear::{
            ipa::MultilinearIpa,
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
