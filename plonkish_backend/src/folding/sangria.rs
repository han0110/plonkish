use crate::{
    folding::protostar::{
        Protostar, ProtostarAccumulator, ProtostarAccumulatorInstance, ProtostarProverParam,
        ProtostarStrategy::NoCompressing, ProtostarVerifierParam,
    },
    pcs::PolynomialCommitmentScheme,
};

mod hyperplonk;

pub type Sangria<Pb> = Protostar<Pb, { NoCompressing as usize }>;

pub type SangriaProverParam<F, Pb> = ProtostarProverParam<F, Pb, { NoCompressing as usize }>;

pub type SangriaVerifierParam<F, Pb> = ProtostarVerifierParam<F, Pb, { NoCompressing as usize }>;

pub type SangriaAccumulator<F, Pcs> = ProtostarAccumulator<F, Pcs, { NoCompressing as usize }>;

pub type SangriaAccumulatorInstance<F, Pcs> = ProtostarAccumulatorInstance<
    F,
    <Pcs as PolynomialCommitmentScheme<F>>::Commitment,
    { NoCompressing as usize },
>;
