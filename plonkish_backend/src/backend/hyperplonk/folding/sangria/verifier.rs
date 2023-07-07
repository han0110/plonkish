use crate::{
    pcs::AdditiveCommitment,
    util::{
        arithmetic::{powers, PrimeField},
        chain, izip_eq, Itertools,
    },
};
use std::{fmt::Debug, iter};

#[derive(Debug)]
pub(crate) struct SangriaInstance<F, C> {
    pub(crate) instances: Vec<Vec<F>>,
    pub(crate) witness_comms: Vec<C>,
    pub(crate) challenges: Vec<F>,
    pub(crate) u: F,
    pub(crate) e_comm: C,
}

impl<F, C> SangriaInstance<F, C>
where
    F: PrimeField,
    C: Default,
{
    pub(crate) fn init(
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        Self {
            instances: num_instances.iter().map(|n| vec![F::ZERO; *n]).collect(),
            witness_comms: iter::repeat_with(C::default)
                .take(num_witness_polys)
                .collect(),
            challenges: vec![F::ZERO; num_challenges],
            u: F::ZERO,
            e_comm: C::default(),
        }
    }

    pub(crate) fn from_committed(
        instances: &[Vec<F>],
        witness_comms: impl IntoIterator<Item = C>,
        challenges: Vec<F>,
    ) -> Self {
        Self {
            instances: instances
                .iter()
                .map(|instances| instances.to_vec())
                .collect(),
            witness_comms: witness_comms.into_iter().collect(),
            challenges,
            u: F::ONE,
            e_comm: C::default(),
        }
    }

    pub(crate) fn instances(&self) -> &[Vec<F>] {
        &self.instances
    }
}

impl<F, C> SangriaInstance<F, C>
where
    F: PrimeField,
    C: AdditiveCommitment<F>,
{
    pub(crate) fn fold(&mut self, rhs: &Self, cross_term_comms: &[C], r: &F) {
        let one = F::ONE;
        let powers_of_r = powers(*r).take(cross_term_comms.len() + 2).collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::sum_with_scalar([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.e_comm = {
            let comms = chain![[&self.e_comm], cross_term_comms, [&rhs.e_comm]];
            C::sum_with_scalar(&powers_of_r, comms)
        };
    }
}
