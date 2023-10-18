use crate::{
    backend::{PlonkishCircuit, PlonkishCircuitInfo},
    pcs::{CommitmentChunk, PolynomialCommitmentScheme},
    util::{
        arithmetic::Field,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::BorrowMut, fmt::Debug};

pub mod protostar;
pub mod sangria;

pub trait AccumulationScheme<F: Field>: Clone + Debug {
    type Pcs: PolynomialCommitmentScheme<F>;
    type ProverParam: Debug + Serialize + DeserializeOwned;
    type VerifierParam: Debug + Serialize + DeserializeOwned;
    type Accumulator: Debug + AsRef<Self::AccumulatorInstance>;
    type AccumulatorInstance: Clone + Debug + Serialize + DeserializeOwned;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<<Self::Pcs as PolynomialCommitmentScheme<F>>::Param, Error>;

    fn preprocess(
        param: &<Self::Pcs as PolynomialCommitmentScheme<F>>::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn init_accumulator(pp: &Self::ProverParam) -> Result<Self::Accumulator, Error>;

    fn init_accumulator_from_nark(
        pp: &Self::ProverParam,
        nark: PlonkishNark<F, Self::Pcs>,
    ) -> Result<Self::Accumulator, Error>;

    fn prove_nark(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<PlonkishNark<F, Self::Pcs>, Error>;

    fn prove_accumulation<const IS_INCOMING_ABSORBED: bool>(
        pp: &Self::ProverParam,
        accumulator: impl BorrowMut<Self::Accumulator>,
        incoming: &Self::Accumulator,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn prove_accumulation_from_nark(
        pp: &Self::ProverParam,
        accumulator: impl BorrowMut<Self::Accumulator>,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        mut rng: impl RngCore,
    ) -> Result<(), Error> {
        let nark = Self::prove_nark(pp, circuit, transcript, &mut rng)?;
        let incoming = Self::init_accumulator_from_nark(pp, nark)?;
        Self::prove_accumulation::<true>(pp, accumulator, &incoming, transcript, &mut rng)?;
        Ok(())
    }

    fn verify_accumulation_from_nark(
        vp: &Self::VerifierParam,
        accumulator: impl BorrowMut<Self::AccumulatorInstance>,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn prove_decider(
        pp: &Self::ProverParam,
        accumulator: &Self::Accumulator,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn prove_decider_with_last_nark(
        pp: &Self::ProverParam,
        mut accumulator: impl BorrowMut<Self::Accumulator>,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        mut rng: impl RngCore,
    ) -> Result<(), Error> {
        Self::prove_accumulation_from_nark(
            pp,
            accumulator.borrow_mut(),
            circuit,
            transcript,
            &mut rng,
        )?;
        Self::prove_decider(pp, accumulator.borrow(), transcript, &mut rng)?;
        Ok(())
    }

    fn verify_decider(
        vp: &Self::VerifierParam,
        accumulator: &Self::AccumulatorInstance,
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn verify_decider_with_last_nark(
        vp: &Self::VerifierParam,
        mut accumulator: impl BorrowMut<Self::AccumulatorInstance>,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Self::Pcs>, F>,
        mut rng: impl RngCore,
    ) -> Result<(), Error> {
        Self::verify_accumulation_from_nark(
            vp,
            accumulator.borrow_mut(),
            instances,
            transcript,
            &mut rng,
        )?;
        Self::verify_decider(vp, accumulator.borrow(), transcript, &mut rng)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PlonkishNark<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    instance: PlonkishNarkInstance<F, Pcs::Commitment>,
    witness_polys: Vec<Pcs::Polynomial>,
}

impl<F, Pcs> PlonkishNark<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn new(
        instances: Vec<Vec<F>>,
        challenges: Vec<F>,
        witness_comms: Vec<Pcs::Commitment>,
        witness_polys: Vec<Pcs::Polynomial>,
    ) -> Self {
        Self {
            instance: PlonkishNarkInstance::new(instances, challenges, witness_comms),
            witness_polys,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlonkishNarkInstance<F, C> {
    instances: Vec<Vec<F>>,
    challenges: Vec<F>,
    witness_comms: Vec<C>,
}

impl<F, C> PlonkishNarkInstance<F, C> {
    fn new(instances: Vec<Vec<F>>, challenges: Vec<F>, witness_comms: Vec<C>) -> Self {
        Self {
            instances,
            challenges,
            witness_comms,
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        accumulation::AccumulationScheme,
        backend::{PlonkishCircuit, PlonkishCircuitInfo},
        pcs::PolynomialCommitmentScheme,
        util::{
            arithmetic::PrimeField,
            end_timer, start_timer,
            test::seeded_std_rng,
            transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
            DeserializeOwned, Serialize,
        },
    };
    use std::{hash::Hash, ops::Range};

    pub(crate) fn run_accumulation_scheme<F, Fs, T, C>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<F>, Vec<C>),
    ) where
        F: PrimeField + Hash + Serialize + DeserializeOwned,
        Fs: AccumulationScheme<F>,
        T: TranscriptRead<<Fs::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
            + TranscriptWrite<<Fs::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
        C: PlonkishCircuit<F>,
    {
        for num_vars in num_vars_range {
            let (circuit_info, circuits) = circuit_fn(num_vars);
            let last_circuit = circuits.last().unwrap();

            let timer = start_timer(|| format!("setup-{num_vars}"));
            let param = Fs::setup(&circuit_info, seeded_std_rng()).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("preprocess-{num_vars}"));
            let (pp, vp) = Fs::preprocess(&param, &circuit_info).unwrap();
            end_timer(timer);

            let (accumulator_before_last, proof) = {
                let mut accumulator = Fs::init_accumulator(&pp).unwrap();
                for circuit in circuits[..circuits.len() - 1].iter() {
                    let timer = start_timer(|| format!("prove_accumulation_from_nark-{num_vars}"));
                    Fs::prove_accumulation_from_nark(
                        &pp,
                        &mut accumulator,
                        circuit,
                        &mut T::new(()),
                        seeded_std_rng(),
                    )
                    .unwrap();
                    end_timer(timer);
                }

                let accumulator_before_last = accumulator.as_ref().clone();

                let timer = start_timer(|| format!("prove_decider_with_last_nark-{num_vars}"));
                let proof = {
                    let mut transcript = T::new(());
                    Fs::prove_decider_with_last_nark(
                        &pp,
                        &mut accumulator,
                        last_circuit,
                        &mut transcript,
                        seeded_std_rng(),
                    )
                    .unwrap();
                    transcript.into_proof()
                };
                end_timer(timer);

                (accumulator_before_last, proof)
            };

            let timer = start_timer(|| format!("verify_decider_with_last_nark-{num_vars}"));
            let result = {
                let mut transcript = T::from_proof((), proof.as_slice());
                Fs::verify_decider_with_last_nark(
                    &vp,
                    accumulator_before_last,
                    last_circuit.instances(),
                    &mut transcript,
                    seeded_std_rng(),
                )
            };
            assert_eq!(result, Ok(()));
            end_timer(timer);
        }
    }
}
