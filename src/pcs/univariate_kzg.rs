use crate::{
    pcs::{Evaluation, PolynomialCommitmentScheme},
    poly::univariate::UnivariatePolynomial,
    util::{
        arithmetic::{
            fixed_base_msm, inner_product, powers, variable_base_msm, window_size, window_table,
            Curve, Field, MultiMillerLoop, PrimeCurveAffine,
        },
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{iter, marker::PhantomData, ops::Neg};

#[derive(Clone, Debug)]
pub struct UnivariateKzg<M: MultiMillerLoop>(PhantomData<M>);

#[derive(Clone, Debug)]
pub struct UnivariateKzgParam<M: MultiMillerLoop> {
    pub g1: M::G1Affine,
    pub powers_of_s: Vec<M::G1Affine>,
    pub g2: M::G2Affine,
    pub s_g2: M::G2Affine,
}

impl<M: MultiMillerLoop> UnivariateKzgParam<M> {
    pub fn degree(&self) -> usize {
        self.powers_of_s.len() - 1
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgProverParam<M: MultiMillerLoop> {
    pub g1: M::G1Affine,
    pub powers_of_s: Vec<M::G1Affine>,
}

impl<M: MultiMillerLoop> UnivariateKzgProverParam<M> {
    pub fn degree(&self) -> usize {
        self.powers_of_s.len() - 1
    }
}

#[derive(Clone, Debug)]
pub struct UnivariateKzgVerifierParam<M: MultiMillerLoop> {
    pub g1: M::G1Affine,
    pub g2: M::G2Affine,
    pub s_g2: M::G2Affine,
}

impl<M: MultiMillerLoop> PolynomialCommitmentScheme<M::Scalar> for UnivariateKzg<M> {
    type Param = UnivariateKzgParam<M>;
    type ProverParam = UnivariateKzgProverParam<M>;
    type VerifierParam = UnivariateKzgVerifierParam<M>;
    type Polynomial = UnivariatePolynomial<M::Scalar>;
    type Point = M::Scalar;
    type Commitment = M::G1Affine;
    type BatchCommitment = Vec<M::G1Affine>;

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

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() < poly.degree() {
            return Err(Error::InvalidPcsParam(format!(
                "Too large degree of poly to commit (param supports degree up to {} but got {})",
                pp.degree(),
                poly.degree()
            )));
        }

        Ok(variable_base_msm(&poly[..], &pp.powers_of_s[..=poly.degree()]).into())
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Self::BatchCommitment, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        point: &Self::Point,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptWrite<M::Scalar, Commitment = M::G1Affine>,
    ) -> Result<(), Error> {
        if pp.degree() < poly.degree() {
            return Err(Error::InvalidPcsParam(format!(
                "Too large degree of poly to open (param supports degree up to {} but got {})",
                pp.degree(),
                poly.degree()
            )));
        }

        let divisor = Self::Polynomial::new(vec![point.neg(), M::Scalar::one()]);
        let (quotient, remainder) = poly.div_rem(&divisor);

        if cfg!(feature = "sanity-check") {
            assert_eq!(&remainder[0], eval);
        }

        transcript.write_commitment(
            variable_base_msm(&quotient[..], &pp.powers_of_s[..=quotient.degree()]).into(),
        )?;

        Ok(())
    }

    // TODO: Implement 2020/081
    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        points: &[Self::Point],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<M::Scalar, Commitment = M::G1Affine>,
    ) -> Result<(), Error> {
        let polys = polys.into_iter().collect_vec();
        for eval in evals {
            Self::open(
                pp,
                polys[eval.poly()],
                &points[eval.point()],
                eval.value(),
                transcript,
            )?;
        }
        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Self::Point,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<M::Scalar, Commitment = M::G1Affine>,
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
        batch_comm: &Self::BatchCommitment,
        points: &[Self::Point],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<M::Scalar, Commitment = M::G1Affine>,
    ) -> Result<(), Error> {
        let quotients = transcript.read_n_commitments(evals.len())?;
        let seps = transcript.squeeze_n_challenges(evals.len());
        let sep_points = seps
            .iter()
            .zip(evals.iter())
            .map(|(sep, eval)| *sep * &points[eval.point()])
            .collect_vec();
        let lhs = &variable_base_msm(
            iter::once(&-inner_product(evals.iter().map(Evaluation::value), &seps))
                .chain(&seps)
                .chain(&sep_points),
            iter::once(&vp.g1)
                .chain(evals.iter().map(|eval| &batch_comm[eval.poly()]))
                .chain(&quotients),
        )
        .into();
        let rhs = &variable_base_msm(&seps, &quotients).into();
        M::pairings_product_is_identity(&[(lhs, &vp.g2.neg().into()), (rhs, &vp.s_g2.into())])
            .then_some(())
            .ok_or_else(|| Error::InvalidPcsOpen("Invalid univariate KZG batch open".to_string()))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{univariate_kzg::UnivariateKzg, Evaluation, PolynomialCommitmentScheme},
        util::{
            transcript::{Keccak256Transcript, Transcript, TranscriptRead, TranscriptWrite},
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
            let mut transcript = Keccak256Transcript::new(Vec::new());
            let poly = Polynomial::rand(pp.degree(), OsRng);
            transcript
                .write_commitment(Pcs::commit(&pp, &poly).unwrap())
                .unwrap();
            let point = transcript.squeeze_challenge();
            let eval = poly.evaluate(&point);
            transcript.write_scalar(eval).unwrap();
            Pcs::open(&pp, &poly, &point, &eval, &mut transcript).unwrap();
            transcript.finalize()
        };
        // Verify
        let accept = {
            let mut transcript = Keccak256Transcript::new(proof.as_slice());
            Pcs::verify(
                &vp,
                &transcript.read_commitment().unwrap(),
                &transcript.squeeze_challenge(),
                &transcript.read_scalar().unwrap(),
                &mut transcript,
            )
            .is_ok()
        };
        assert!(accept);
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
            let mut transcript = Keccak256Transcript::new(Vec::new());
            let polys = iter::repeat_with(|| Polynomial::rand(pp.degree(), OsRng))
                .take(batch_size)
                .collect_vec();
            for comm in Pcs::batch_commit(&pp, &polys).unwrap() {
                transcript.write_commitment(comm).unwrap();
            }
            let points = transcript.squeeze_n_challenges(batch_size);
            let evals = polys
                .iter()
                .zip(points.iter())
                .enumerate()
                .map(|(idx, (poly, point))| Evaluation::new(idx, idx, poly.evaluate(point)))
                .collect_vec();
            for eval in evals.iter() {
                transcript.write_scalar(*eval.value()).unwrap();
            }
            Pcs::batch_open(&pp, &polys, &points, &evals, &mut transcript).unwrap();
            transcript.finalize()
        };
        // Batch verify
        let accept = {
            let mut transcript = Keccak256Transcript::new(proof.as_slice());
            Pcs::batch_verify(
                &vp,
                &transcript.read_n_commitments(batch_size).unwrap(),
                &transcript.squeeze_n_challenges(batch_size),
                &transcript
                    .read_n_scalars(batch_size)
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .map(|(idx, eval)| Evaluation::new(idx, idx, eval))
                    .collect_vec(),
                &mut transcript,
            )
            .is_ok()
        };
        assert!(accept);
    }
}
