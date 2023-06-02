use crate::{
    pcs::{
        multilinear::additive, univariate::UnivariateKzg, Evaluation, Point, Polynomial,
        PolynomialCommitmentScheme,
    },
    poly::{
        multilinear::{merge_into, MultilinearPolynomial},
        univariate::UnivariatePolynomial,
    },
    util::{
        arithmetic::{Field, MultiMillerLoop},
        chain,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct MultilinearSimulator<Pcs>(PhantomData<Pcs>);

impl<M> PolynomialCommitmentScheme<M::Scalar> for MultilinearSimulator<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Param;
    type ProverParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::ProverParam;
    type VerifierParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<M::Scalar>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Commitment;
    type CommitmentWithAux =
        <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::CommitmentWithAux;

    fn setup(size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        UnivariateKzg::<M>::setup(size, rng)
    }

    fn trim(
        param: &Self::Param,
        size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateKzg::<M>::trim(param, size)
    }

    fn commit(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
    ) -> Result<Self::CommitmentWithAux, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            return Err(Error::InvalidPcsParam(format!(
                "Too large degree of poly to commit (param supports degree up to {} but got {})",
                pp.degree(),
                poly.evals().len()
            )));
        }

        Ok(UnivariateKzg::<M>::commit_coeffs(pp, poly.evals()))
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
        comm: &Self::CommitmentWithAux,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptWrite<Self::Commitment, M::Scalar>,
    ) -> Result<(), Error> {
        let fs = {
            let mut fs = Vec::with_capacity(point.len());
            fs.push(UnivariatePolynomial::new(poly.evals().to_vec()));
            for x_i in &point[..point.len() - 1] {
                let f_i_minus_one = fs.last().unwrap().coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                fs.push(UnivariatePolynomial::new(f_i));
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
        let points = [beta, -beta, beta.square()];

        let evals = fs
            .iter()
            .enumerate()
            .flat_map(|(idx, f)| {
                chain![(idx != 0).then_some(2), [0, 1]]
                    .map(move |point| Evaluation::new(idx, point, f.evaluate(&points[point])))
            })
            .collect_vec();
        transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

        UnivariateKzg::<M>::batch_open(pp, &fs, &comms, &points, &evals, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::CommitmentWithAux>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<Self::Commitment, M::Scalar>,
    ) -> Result<(), Error>
    where
        Self::CommitmentWithAux: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, Some(comms), points, evals, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::Commitment, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        let comms = chain![[*comm], transcript.read_commitments(num_vars - 1)?].collect_vec();

        let beta = transcript.squeeze_challenge();
        let points = [beta, -beta, beta.square()];

        let evals = (0..num_vars)
            .flat_map(|idx| {
                chain![(idx != 0).then_some(2), [0, 1]]
                    .map(|point| {
                        transcript
                            .read_field_element()
                            .map(|eval| Evaluation::new(idx, point, eval))
                    })
                    .collect_vec()
            })
            .try_collect::<_, Vec<_>, _>()?;

        let two = M::Scalar::ONE + M::Scalar::ONE;
        let beta_inv = beta.invert().unwrap();
        for ((a, b, c), x_i) in chain![evals.iter().map(Evaluation::value), [eval]]
            .tuples()
            .zip_eq(point)
        {
            (two * c == (*a + b) * (M::Scalar::ONE - x_i) + (*a - b) * beta_inv * x_i)
                .then_some(())
                .ok_or_else(|| Error::InvalidPcsOpen("Consistency failure".to_string()))?;
        }

        UnivariateKzg::<M>::batch_verify(vp, &comms, &points, &evals, transcript)
    }

    fn batch_verify(
        vp: &Self::VerifierParam,
        comms: &[Self::Commitment],
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<Self::Commitment, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::{
                simulator::MultilinearSimulator,
                test::{run_batch_commit_open_verify, run_commit_open_verify},
            },
            univariate::UnivariateKzg,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = MultilinearSimulator<UnivariateKzg<Bn256>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
