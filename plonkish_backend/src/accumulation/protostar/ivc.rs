use crate::{
    accumulation::protostar::{ProtostarAccumulatorInstance, ProtostarStrategy},
    util::{arithmetic::PrimeField, Deserialize, Serialize},
};

#[cfg(feature = "frontend-halo2")]
pub mod halo2;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProtostarAccumulationVerifierParam<F> {
    pub(crate) vp_digest: F,
    pub(crate) strategy: ProtostarStrategy,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<Vec<usize>>,
    pub(crate) num_cross_terms: usize,
}

impl<N: PrimeField> ProtostarAccumulationVerifierParam<N> {
    pub fn num_folding_witness_polys(&self) -> usize {
        self.num_witness_polys.iter().sum()
    }

    pub fn num_folding_challenges(&self) -> usize {
        self.num_challenges.iter().flatten().sum()
    }

    pub fn init_accumulator<F: PrimeField, Comm: Default>(
        &self,
    ) -> ProtostarAccumulatorInstance<F, Comm> {
        ProtostarAccumulatorInstance::init(
            self.strategy,
            &self.num_instances,
            self.num_folding_witness_polys(),
            self.num_folding_challenges(),
        )
    }
}
