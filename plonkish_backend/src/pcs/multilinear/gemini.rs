//! Implementation of section 2.4.2 of 2022/420, with improvement ported from Aztec's Barretenberg
//! https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/honk/pcs/gemini/gemini.cpp.

use crate::{
    pcs::{
        multilinear::additive,
        univariate::{err_too_large_deree, UnivariateKzg, UnivariateKzgCommitment},
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{
        multilinear::{merge_into, MultilinearPolynomial},
        univariate::UnivariatePolynomial,
    },
    util::{
        arithmetic::{squares, Field, MultiMillerLoop},
        chain,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{marker::PhantomData, ops::Neg};

#[derive(Clone, Debug)]
pub struct Gemini<Pcs>(PhantomData<Pcs>);

impl<M> PolynomialCommitmentScheme<M::Scalar> for Gemini<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
    M::Scalar: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned,
    M::G2Affine: Serialize + DeserializeOwned,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Param;
    type ProverParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::ProverParam;
    type VerifierParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<M::Scalar>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Commitment;
    type CommitmentChunk =
        <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        UnivariateKzg::<M>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateKzg::<M>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("commit", pp.degree(), got));
        }

        Ok(UnivariateKzg::commit_monomial(pp, poly.evals()))
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
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let fs = {
            let mut fs = Vec::with_capacity(num_vars);
            fs.push(UnivariatePolynomial::monomial(poly.evals().to_vec()));
            for x_i in &point[..num_vars - 1] {
                let f_i_minus_one = fs.last().unwrap().coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                fs.push(UnivariatePolynomial::monomial(f_i));
            }

            if cfg!(feature = "sanity-check") {
                let f_last = fs.last().unwrap();
                let x_last = point.last().unwrap();
                assert_eq!(
                    f_last[0] * (M::Scalar::ONE - x_last) + f_last[1] * x_last,
                    *eval
                );
            }

            fs
        };
        let comms = chain![
            [comm.clone()],
            UnivariateKzg::<M>::batch_commit_and_write(pp, &fs[1..], transcript)?
        ]
        .collect_vec();

        let beta = transcript.squeeze_challenge();
        let points = chain![[beta], squares(beta).map(Neg::neg)]
            .take(num_vars + 1)
            .collect_vec();

        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .map(|(idx, point)| Evaluation::new(idx, point, fs[idx].evaluate(&points[point])))
            .collect_vec();
        transcript.write_field_elements(evals[1..].iter().map(Evaluation::value))?;

        UnivariateKzg::<M>::batch_open(pp, &fs, &comms, &points, &evals, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        UnivariateKzg::read_commitments(vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        let comms = chain![[comm.0], transcript.read_commitments(num_vars - 1)?]
            .map(UnivariateKzgCommitment)
            .collect_vec();

        let beta = transcript.squeeze_challenge();
        let squares_of_beta = squares(beta).take(num_vars).collect_vec();

        let evals = transcript.read_field_elements(num_vars)?;

        let one = M::Scalar::ONE;
        let two = one.double();
        let eval_0 = evals.iter().zip(&squares_of_beta).zip(point).rev().fold(
            *eval,
            |eval_pos, ((eval_neg, sqaure_of_beta), x_i)| {
                (two * sqaure_of_beta * eval_pos - ((one - x_i) * sqaure_of_beta - x_i) * eval_neg)
                    * ((one - x_i) * sqaure_of_beta + x_i).invert().unwrap()
            },
        );
        let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
            .zip(chain![[eval_0], evals])
            .map(|((idx, point), eval)| Evaluation::new(idx, point, eval))
            .collect_vec();
        let points = chain!([beta], squares_of_beta.into_iter().map(Neg::neg)).collect_vec();

        UnivariateKzg::<M>::batch_verify(vp, &comms, &points, &evals, transcript)
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::gemini::Gemini,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::UnivariateKzg,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = Gemini<UnivariateKzg<Bn256>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
