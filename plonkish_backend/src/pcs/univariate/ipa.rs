use crate::{
    pcs::{
        univariate::{additive, err_too_large_deree, monomial_g_to_lagrange_g, validate_input},
        Additive, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{
        multilinear,
        univariate::{UnivariateBasis::*, UnivariatePolynomial},
    },
    util::{
        arithmetic::{
            batch_projective_to_affine, inner_product, powers, squares, variable_base_msm, Curve,
            CurveAffine, CurveExt, Field, Group, PrimeField,
        },
        chain, izip,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Either, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::group::ff::BatchInvert;
use rand::RngCore;
use std::{borrow::Cow, iter, marker::PhantomData, slice};

#[derive(Clone, Debug)]
pub struct UnivariateIpa<C: CurveAffine>(PhantomData<C>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateIpaParam<C: CurveAffine> {
    k: usize,
    monomial: Vec<C>,
    lagrange: Vec<C>,
    h: C,
}

impl<C: CurveAffine> UnivariateIpaParam<C> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn degree(&self) -> usize {
        self.monomial.len() - 1
    }

    pub fn monomial(&self) -> &[C] {
        &self.monomial
    }

    pub fn lagrange(&self) -> &[C] {
        &self.lagrange
    }

    pub fn h(&self) -> &C {
        &self.h
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnivariateIpaVerifierParam<C: CurveAffine> {
    k: usize,
    monomial: Vec<C>,
    h: C,
}

impl<C: CurveAffine> UnivariateIpaVerifierParam<C> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn h(&self) -> &C {
        &self.h
    }

    pub fn monomial(&self) -> &[C] {
        &self.monomial
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnivariateIpaCommitment<C: CurveAffine>(pub C);

impl<C: CurveAffine> Default for UnivariateIpaCommitment<C> {
    fn default() -> Self {
        Self(C::identity())
    }
}

impl<C: CurveAffine> AsRef<[C]> for UnivariateIpaCommitment<C> {
    fn as_ref(&self) -> &[C] {
        slice::from_ref(&self.0)
    }
}

impl<C: CurveAffine> AsRef<C> for UnivariateIpaCommitment<C> {
    fn as_ref(&self) -> &C {
        &self.0
    }
}

impl<C: CurveAffine> From<C> for UnivariateIpaCommitment<C> {
    fn from(comm: C) -> Self {
        Self(comm)
    }
}

impl<C: CurveAffine> Additive<C::Scalar> for UnivariateIpaCommitment<C> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a C::Scalar>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().map(|base| &base.0).collect_vec();
        UnivariateIpaCommitment(variable_base_msm(scalars, bases).to_affine())
    }
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for UnivariateIpa<C>
where
    C: CurveAffine + Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Param = UnivariateIpaParam<C>;
    type ProverParam = UnivariateIpaParam<C>;
    type VerifierParam = UnivariateIpaVerifierParam<C>;
    type Polynomial = UnivariatePolynomial<C::Scalar>;
    type Commitment = UnivariateIpaCommitment<C>;
    type CommitmentChunk = C;

    fn setup(poly_size: usize, _: usize, _: impl RngCore) -> Result<Self::Param, Error> {
        // TODO: Support arbitrary degree.
        assert!(poly_size.is_power_of_two());
        assert!(poly_size.ilog2() <= C::Scalar::S);

        let k = poly_size.ilog2() as usize;

        let monomial = {
            let mut g = vec![C::Curve::identity(); poly_size];
            parallelize(&mut g, |(g, start)| {
                let hasher = C::CurveExt::hash_to_curve("UnivariateIpa::setup");
                for (g, idx) in g.iter_mut().zip(start as u32..) {
                    let mut message = [0u8; 5];
                    message[1..5].copy_from_slice(&idx.to_le_bytes());
                    *g = hasher(&message);
                }
            });
            batch_projective_to_affine(&g)
        };

        let lagrange = monomial_g_to_lagrange_g(&monomial);

        let hasher = C::CurveExt::hash_to_curve("UnivariateIpa::setup");
        let h = hasher(&[1]).to_affine();

        Ok(Self::Param {
            k,
            monomial,
            lagrange,
            h,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());

        let k = poly_size.ilog2() as usize;

        if param.monomial.len() < poly_size {
            return Err(err_too_large_deree("trim", param.degree(), poly_size - 1));
        }

        let monomial = param.monomial[..poly_size].to_vec();
        let lagrange = if param.lagrange.len() == poly_size {
            param.lagrange.clone()
        } else {
            monomial_g_to_lagrange_g(&monomial)
        };

        let pp = Self::ProverParam {
            k,
            monomial: monomial.clone(),
            lagrange,
            h: param.h,
        };
        let vp = Self::VerifierParam {
            k,
            monomial,
            h: param.h,
        };
        Ok((pp, vp))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.degree(), [poly])?;

        let coeffs = poly.coeffs();
        let bases = match poly.basis() {
            Monomial => pp.monomial(),
            Lagrange => pp.lagrange(),
        };
        Ok(variable_base_msm(coeffs, &bases[..coeffs.len()]).into()).map(UnivariateIpaCommitment)
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
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptWrite<C, C::Scalar>,
    ) -> Result<(), Error> {
        assert_eq!(poly.basis(), Monomial);

        validate_input("open", pp.degree(), [poly])?;

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let bases = pp.monomial();
        let coeffs = chain![poly.coeffs().iter().cloned(), iter::repeat(C::Scalar::ZERO)]
            .take(bases.len())
            .collect_vec();
        let zs = powers(*point).take(bases.len()).collect_vec();
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
        additive::batch_open::<_, Self>(pp, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, C::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let comms = transcript.read_commitments(num_polys)?;
        Ok(comms.into_iter().map(UnivariateIpaCommitment).collect())
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<C::Scalar, Self::Polynomial>,
        eval: &C::Scalar,
        transcript: &mut impl TranscriptRead<C, C::Scalar>,
    ) -> Result<(), Error> {
        let bases = vp.monomial();
        let point = Either::Left(point);
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
        additive::batch_verify::<_, Self>(vp, comms, points, evals, transcript)
    }
}

pub(crate) fn prove_bulletproof_reduction<'a, C: CurveAffine>(
    bases: impl Into<Cow<'a, [C]>>,
    h: &C,
    coeffs: impl Into<Cow<'a, [C::Scalar]>>,
    zs: impl Into<Cow<'a, [C::Scalar]>>,
    transcript: &mut impl TranscriptWrite<C, C::Scalar>,
) -> Result<(), Error> {
    let mut bases = bases.into().into_owned();
    let mut coeffs = coeffs.into().into_owned();
    let mut zs = zs.into().into_owned();

    assert_eq!(bases.len(), coeffs.len());
    assert_eq!(bases.len(), zs.len());
    assert!(bases.len().is_power_of_two());

    let xi_0 = transcript.squeeze_challenge();
    let h_prime = (*h * xi_0).to_affine();

    let k = bases.len().ilog2() as usize;
    for i in 0..k {
        let mid = 1 << (k - i - 1);

        let (bases_l, bases_r) = bases.split_at(mid);
        let (coeffs_l, coeffs_r) = coeffs.split_at(mid);
        let (zs_l, zs_r) = zs.split_at(mid);
        let (c_l, c_r) = (inner_product(coeffs_r, zs_l), inner_product(coeffs_l, zs_r));
        let l_i = variable_base_msm(chain![coeffs_r, [&c_l]], chain![bases_l, [&h_prime]]);
        let r_i = variable_base_msm(chain![coeffs_l, [&c_r]], chain![bases_r, [&h_prime]]);
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

    transcript.write_field_element(&coeffs[0])?;

    Ok(())
}

pub(crate) fn verify_bulletproof_reduction<C: CurveAffine>(
    bases: &[C],
    h: &C,
    comm: impl AsRef<C>,
    point: Either<&C::Scalar, &[C::Scalar]>,
    eval: &C::Scalar,
    transcript: &mut impl TranscriptRead<C, C::Scalar>,
) -> Result<(), Error> {
    assert!(bases.len().is_power_of_two());
    if let Either::Right(point) = point {
        assert_eq!(1 << point.len(), bases.len());
    }

    let k = bases.len().ilog2() as usize;

    let xi_0 = transcript.squeeze_challenge();

    let (ls, rs, xis) = iter::repeat_with(|| {
        Ok((
            transcript.read_commitment()?,
            transcript.read_commitment()?,
            transcript.squeeze_challenge(),
        ))
    })
    .take(k)
    .collect::<Result<Vec<_>, _>>()?
    .into_iter()
    .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
    let neg_c = -transcript.read_field_element()?;

    let xi_invs = {
        let mut xi_invs = xis.clone();
        xi_invs.batch_invert();
        xi_invs
    };
    let neg_c_h = h_coeffs(neg_c, &xis);
    let (kind, neg_c_h_eval) = match point {
        Either::Left(point) => ("univariate", h_eval(neg_c, &xis, point)),
        Either::Right(point) => ("multivariate", multilinear::evaluate(&neg_c_h, point)),
    };
    let u = xi_0 * (neg_c_h_eval + eval);
    let scalars = chain![&xi_invs, &xis, &neg_c_h, [&u]];
    let bases = chain![&ls, &rs, bases, [h]];
    bool::from((variable_base_msm(scalars, bases) + comm.as_ref()).is_identity())
        .then_some(())
        .ok_or_else(|| Error::InvalidPcsOpen(format!("Invalid {kind} IPA open")))
}

pub(crate) fn h_coeffs<F: Field>(init: F, xi: &[F]) -> Vec<F> {
    assert!(!xi.is_empty());

    let mut coeffs = vec![F::ZERO; 1 << xi.len()];
    coeffs[0] = init;

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

fn h_eval<F: Field>(init: F, xis: &[F], x: &F) -> F {
    izip!(squares(*x), xis.iter().rev())
        .map(|(square_of_x, xi)| F::ONE + square_of_x * xi)
        .fold(init, |acc, item| acc * item)
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::ipa::UnivariateIpa,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::pasta::pallas::Affine;

    type Pcs = UnivariateIpa<Affine>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
