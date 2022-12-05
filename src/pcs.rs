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
    type Config: Debug + Copy;
    type Param: Debug;
    type ProverParam: Debug;
    type VerifierParam: Debug;
    type Polynomial: Debug;
    type Point: Debug;
    type Commitment: Debug;
    type BatchCommitment: Debug;
    type Proof: Debug;
    type BatchProof: Debug;

    fn setup(config: Self::Config, rng: impl RngCore) -> Result<Self::Param, Error>;

    fn trim(
        param: &Self::Param,
        config: Self::Config,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error>;

    fn batch_commit(
        pp: &Self::ProverParam,
        polys: &[Self::Polynomial],
    ) -> Result<Self::BatchCommitment, Error>;

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(F, Self::Proof), Error>;

    fn batch_open(
        pp: &Self::ProverParam,
        polys: &[Self::Polynomial],
        points: &[Self::Point],
        transcript: &mut impl TranscriptWrite<F>,
    ) -> Result<(Vec<F>, Self::BatchProof), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        proof: &Self::Proof,
    ) -> Result<(), Error>;

    fn batch_verify(
        vp: &Self::VerifierParam,
        batch_comm: &Self::BatchCommitment,
        points: &[Self::Point],
        evals: &[F],
        batch_proof: &Self::BatchProof,
        transcript: &mut impl TranscriptRead<F>,
    ) -> Result<(), Error>;
}
