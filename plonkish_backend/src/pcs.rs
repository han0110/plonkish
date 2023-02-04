use crate::{
    util::{
        arithmetic::Field,
        transcript::{TranscriptRead, TranscriptWrite},
    },
    Error,
};
use rand::RngCore;
use std::fmt::Debug;

pub mod multilinear_kzg;
pub mod univariate_kzg;

pub trait PolynomialCommitmentScheme<F: Field>: Clone + Debug {
    type Param: Debug;
    type ProverParam: Debug;
    type VerifierParam: Debug;
    type Polynomial: Debug;
    type Point: Debug;
    type Commitment: Debug;
    type BatchCommitment: Debug;

    fn setup(size: usize, rng: impl RngCore) -> Result<Self::Param, Error>;

    fn trim(
        param: &Self::Param,
        size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error>;

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Self::BatchCommitment, Error>
    where
        Self::Polynomial: 'a;

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        point: &Self::Point,
        eval: &F,
        transcript: &mut impl TranscriptWrite<F, Commitment = Self::Commitment>,
    ) -> Result<(), Error>;

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        points: &[Self::Point],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<F, Commitment = Self::Commitment>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a;

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        transcript: &mut impl TranscriptRead<F, Commitment = Self::Commitment>,
    ) -> Result<(), Error>;

    fn batch_verify(
        vp: &Self::VerifierParam,
        batch_comm: &Self::BatchCommitment,
        points: &[Self::Point],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<F, Commitment = Self::Commitment>,
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
