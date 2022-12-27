use crate::{
    pcs::PolynomialCommitmentScheme,
    util::{
        arithmetic::Field,
        transcript::{TranscriptRead, TranscriptWrite},
    },
    Error,
};
use rand::RngCore;
use std::fmt::Debug;

pub mod hyperplonk;

pub trait UniversalSnark<F, Pcs>: Clone + Debug
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type CircuitInfo: Debug;
    type ProverParam: Debug;
    type VerifierParam: Debug;

    fn setup(size: usize, rng: impl RngCore) -> Result<Pcs::Param, Error>;

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: Self::CircuitInfo,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn prove(
        pp: &Self::ProverParam,
        instances: &[&[F]],
        witness_collector: impl FnMut(&[F]) -> Result<Vec<Vec<F>>, Error>,
        transcript: &mut impl TranscriptWrite<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[&[F]],
        transcript: &mut impl TranscriptRead<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;
}
