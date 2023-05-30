use crate::{
    util::{
        arithmetic::{variable_base_msm, Curve, CurveAffine, Field},
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::fmt::Debug;

pub mod multilinear;
pub mod univariate;

pub trait Polynomial<F: Field>: Clone + Debug {
    type Point: Clone + Debug;

    fn from_evals(evals: Vec<F>) -> Self;

    fn into_evals(self) -> Vec<F>;

    fn evals(&self) -> &[F];

    fn evaluate(&self, point: &Self::Point) -> F;
}

pub type Point<F, P> = <P as Polynomial<F>>::Point;

pub trait PolynomialCommitmentScheme<F: Field>: Clone + Debug {
    type Param: Debug;
    type ProverParam: Debug;
    type VerifierParam: Debug;
    type Polynomial: Polynomial<F>;
    type Commitment: Clone + Debug + Default;
    type CommitmentWithAux: Clone + Debug + Default + AsRef<Self::Commitment>;

    fn setup(size: usize, rng: impl RngCore) -> Result<Self::Param, Error>;

    fn trim(
        param: &Self::Param,
        size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn commit(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
    ) -> Result<Self::CommitmentWithAux, Error>;

    fn commit_and_write(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        transcript: &mut impl TranscriptWrite<Self::Commitment, F>,
    ) -> Result<Self::CommitmentWithAux, Error> {
        let comm = Self::commit(pp, poly)?;
        transcript.write_commitment(comm.as_ref())?;
        Ok(comm)
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::CommitmentWithAux>, Error>
    where
        Self::Polynomial: 'a;

    fn batch_commit_and_write<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        transcript: &mut impl TranscriptWrite<Self::Commitment, F>,
    ) -> Result<Vec<Self::CommitmentWithAux>, Error>
    where
        Self::Polynomial: 'a,
    {
        let comms = Self::batch_commit(pp, polys)?;
        for comm in comms.iter() {
            transcript.write_commitment(comm.as_ref())?;
        }
        Ok(comms)
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::CommitmentWithAux,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::Commitment, F>,
    ) -> Result<(), Error>;

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::CommitmentWithAux>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::Commitment, F>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a,
        Self::CommitmentWithAux: 'a;

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::Commitment, F>,
    ) -> Result<(), Error>;

    fn batch_verify(
        vp: &Self::VerifierParam,
        comms: &[Self::Commitment],
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::Commitment, F>,
    ) -> Result<(), Error>;
}

#[derive(Clone, Debug)]
pub struct Evaluation<F: Field> {
    poly: usize,
    point: usize,
    value: F,
}

impl<F: Field> Evaluation<F> {
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

pub trait AdditiveCommitment<F: Field>: Debug + Default {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a F> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self
    where
        Self: 'a;
}

impl<C: CurveAffine> AdditiveCommitment<C::Scalar> for C {
    fn sum_with_scalar<'a>(
        scalars: impl IntoIterator<Item = &'a C::Scalar> + 'a,
        bases: impl IntoIterator<Item = &'a Self> + 'a,
    ) -> Self {
        let scalars = scalars.into_iter().collect_vec();
        let bases = bases.into_iter().collect_vec();
        assert_eq!(scalars.len(), bases.len());

        variable_base_msm(scalars, bases).to_affine()
    }
}
