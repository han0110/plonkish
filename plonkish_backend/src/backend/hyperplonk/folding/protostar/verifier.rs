use crate::{
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    util::{
        arithmetic::{inner_product, powers, PrimeField},
        chain, izip_eq, Itertools,
    },
};
use std::{fmt::Debug, iter};

#[derive(Debug)]
pub(crate) struct ProtostarInstance<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) instances: Vec<Vec<F>>,
    pub(crate) witness_comms: Vec<Pcs::Commitment>,
    pub(crate) challenges: Vec<F>,
    pub(crate) u: F,
    pub(crate) compressed_e_sum: F,
    pub(crate) zeta_e_comm: Pcs::Commitment,
}

impl<F, Pcs> ProtostarInstance<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) fn init(
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        Self {
            instances: num_instances.iter().map(|n| vec![F::ZERO; *n]).collect(),
            witness_comms: iter::repeat_with(Pcs::Commitment::default)
                .take(num_witness_polys)
                .collect(),
            challenges: vec![F::ZERO; num_challenges],
            u: F::ZERO,
            compressed_e_sum: F::ZERO,
            zeta_e_comm: Pcs::Commitment::default(),
        }
    }

    pub(crate) fn from_committed(
        instances: &[&[F]],
        witness_comms: impl IntoIterator<Item = Pcs::Commitment>,
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
            compressed_e_sum: F::ZERO,
            zeta_e_comm: Pcs::Commitment::default(),
        }
    }

    pub(crate) fn instance_slices(&self) -> Vec<&[F]> {
        self.instances.iter().map(Vec::as_slice).collect()
    }
}

impl<F, Pcs> ProtostarInstance<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: AdditiveCommitment<F>,
{
    pub(crate) fn fold(
        &mut self,
        rhs: &Self,
        compressed_cross_term_sums: &[F],
        zeta_cross_term_comm: &Pcs::Commitment,
        r: &F,
    ) {
        let one = F::ONE;
        let powers_of_r = powers(*r)
            .take((compressed_cross_term_sums.len() + 2).max(3))
            .collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = Pcs::Commitment::sum_with_scalar([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.compressed_e_sum += &inner_product(
            &powers_of_r[1..],
            chain![compressed_cross_term_sums, [&rhs.compressed_e_sum]],
        );
        self.zeta_e_comm = {
            let comms = [&self.zeta_e_comm, zeta_cross_term_comm, &rhs.zeta_e_comm];
            Pcs::Commitment::sum_with_scalar(&powers_of_r[..3], comms)
        };
    }
}
