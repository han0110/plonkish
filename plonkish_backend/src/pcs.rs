use crate::{
    poly::Polynomial,
    util::{
        arithmetic::Field,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::fmt::Debug;

pub mod multilinear;
pub mod univariate;

pub type Point<F, P> = <P as Polynomial<F>>::Point;

pub type Commitment<F, Pcs> = <Pcs as PolynomialCommitmentScheme<F>>::Commitment;

pub type CommitmentChunk<F, Pcs> = <Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk;

pub trait PolynomialCommitmentScheme<F: Field>: Clone + Debug {
    type Param: Clone + Debug + Serialize + DeserializeOwned;
    type ProverParam: Clone + Debug + Serialize + DeserializeOwned;
    type VerifierParam: Clone + Debug + Serialize + DeserializeOwned;
    type Polynomial: Polynomial<F> + Serialize + DeserializeOwned;
    type Commitment: Clone
        + Debug
        + Default
        + AsRef<[Self::CommitmentChunk]>
        + Serialize
        + DeserializeOwned;
    type CommitmentChunk: Clone + Debug + Default;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error>;

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error>;

    fn commit_and_write(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<Self::Commitment, Error> {
        let comm = Self::commit(pp, poly)?;
        transcript.write_commitments(comm.as_ref())?;
        Ok(comm)
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a;

    fn batch_commit_and_write<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        let comms = Self::batch_commit(pp, polys)?;
        for comm in comms.iter() {
            transcript.write_commitments(comm.as_ref())?;
        }
        Ok(comms)
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>;

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a,
        Self::Commitment: 'a;

    fn read_commitment(
        vp: &Self::VerifierParam,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Self::Commitment, Error> {
        let comms = Self::read_commitments(vp, 1, transcript)?;
        assert_eq!(comms.len(), 1);
        Ok(comms.into_iter().next().unwrap())
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error>;

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>;

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a;
}

#[derive(Clone, Debug)]
pub struct Evaluation<F> {
    poly: usize,
    point: usize,
    value: F,
}

impl<F> Evaluation<F> {
    pub fn new(poly: usize, point: usize, value: F) -> Self {
        Self { poly, point, value }
    }

    pub fn poly(&self) -> usize {
        self.poly
    }

    pub fn point(&self) -> usize {
        self.point
    }

    pub fn value(&self) -> &F {
        &self.value
    }
}

pub trait Additive<F: Field>: Clone + Debug + Default + PartialEq + Eq {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a F>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self
    where
        Self: 'b;
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{Evaluation, PolynomialCommitmentScheme},
        poly::Polynomial,
        util::{
            arithmetic::PrimeField,
            chain,
            transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
            Itertools,
        },
    };
    use rand::{rngs::OsRng, Rng};
    use std::iter;

    pub(super) fn run_commit_open_verify<F, Pcs, T>()
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F>,
        T: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
    {
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
                let mut transcript = T::new(());
                let poly = <Pcs::Polynomial as Polynomial<F>>::rand(1 << k, OsRng);
                let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
                let point = <Pcs::Polynomial as Polynomial<F>>::squeeze_point(k, &mut transcript);
                let eval = poly.evaluate(&point);
                transcript.write_field_element(&eval).unwrap();
                Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
                transcript.into_proof()
            };
            // Verify
            let result = {
                let mut transcript = T::from_proof((), proof.as_slice());
                Pcs::verify(
                    &vp,
                    &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
                    &<Pcs::Polynomial as Polynomial<F>>::squeeze_point(k, &mut transcript),
                    &transcript.read_field_element().unwrap(),
                    &mut transcript,
                )
            };
            assert_eq!(result, Ok(()));
        }
    }

    pub(super) fn run_batch_commit_open_verify<F, Pcs, T>()
    where
        F: PrimeField,
        Pcs: PolynomialCommitmentScheme<F>,
        T: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
    {
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
                (0..batch_size).map(|poly| (poly, 0)),
                iter::repeat_with(|| (rng.gen_range(0..batch_size), rng.gen_range(0..num_points)))
                    .take(batch_size)
            ]
            .unique()
            .collect_vec();
            let proof = {
                let mut transcript = T::new(());
                let polys =
                    iter::repeat_with(|| <Pcs::Polynomial as Polynomial<F>>::rand(1 << k, OsRng))
                        .take(batch_size)
                        .collect_vec();
                let comms = Pcs::batch_commit_and_write(&pp, &polys, &mut transcript).unwrap();
                let points = iter::repeat_with(|| {
                    <Pcs::Polynomial as Polynomial<F>>::squeeze_point(k, &mut transcript)
                })
                .take(num_points)
                .collect_vec();
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
                let mut transcript = T::from_proof((), proof.as_slice());
                Pcs::batch_verify(
                    &vp,
                    &Pcs::read_commitments(&vp, batch_size, &mut transcript).unwrap(),
                    &iter::repeat_with(|| {
                        <Pcs::Polynomial as Polynomial<F>>::squeeze_point(k, &mut transcript)
                    })
                    .take(num_points)
                    .collect_vec(),
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
