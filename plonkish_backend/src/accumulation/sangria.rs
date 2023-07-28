use crate::{
    accumulation::protostar::{
        Protostar, ProtostarAccumulator, ProtostarAccumulatorInstance, ProtostarProverParam,
        ProtostarStrategy::NoCompressing, ProtostarVerifierParam,
    },
    pcs::PolynomialCommitmentScheme,
};

mod hyperplonk;

pub type Sangria<Pb> = Protostar<Pb, { NoCompressing as usize }>;

pub type SangriaProverParam<F, Pb> = ProtostarProverParam<F, Pb>;

pub type SangriaVerifierParam<F, Pb> = ProtostarVerifierParam<F, Pb>;

pub type SangriaAccumulator<F, Pcs> = ProtostarAccumulator<F, Pcs>;

pub type SangriaAccumulatorInstance<F, Pcs> =
    ProtostarAccumulatorInstance<F, <Pcs as PolynomialCommitmentScheme<F>>::Commitment>;
