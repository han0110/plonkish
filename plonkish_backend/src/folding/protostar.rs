use crate::{
    backend::PlonkishBackend,
    folding::{
        protostar::ProtostarStrategy::{Compressing, NoCompressing},
        PlonkishNark, PlonkishNarkInstance,
    },
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    poly::Polynomial,
    util::{
        arithmetic::{inner_product, powers, Field},
        chain,
        expression::Expression,
        izip, izip_eq,
        transcript::Transcript,
        Deserialize, Itertools, Serialize,
    },
    Error,
};
use std::{iter, marker::PhantomData};

pub mod hyperplonk;

#[derive(Clone, Debug)]
pub struct Protostar<Pb, const STRATEGY: usize = { Compressing as usize }>(PhantomData<Pb>);

#[derive(Clone, Copy, Debug)]
pub enum ProtostarStrategy {
    // As known as Sangria
    NoCompressing = 0,
    // Compressing verification as described in 2023/620 section 3.5 but without square-root optimization
    Compressing = 1,
    // TODO:
    // Compressing verification with square-root optimization applied as described in 2023/620 section 3.5
    // CompressingWithSqrtPowers = 3,
}

impl ProtostarStrategy {
    const fn from(strategy: usize) -> Self {
        [NoCompressing, Compressing][strategy]
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProtostarProverParam<F, Pb, const STRATEGY: usize = { Compressing as usize }>
where
    F: Field,
    Pb: PlonkishBackend<F>,
{
    pp: Pb::ProverParam,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    cross_term_expressions: Vec<Expression<F>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProtostarVerifierParam<F, Pb, const STRATEGY: usize = { Compressing as usize }>
where
    F: Field,
    Pb: PlonkishBackend<F>,
{
    vp: Pb::VerifierParam,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    num_cross_terms: usize,
}

#[derive(Clone, Debug)]
pub struct ProtostarAccumulator<F, Pcs, const STRATEGY: usize = { Compressing as usize }>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    instance: ProtostarAccumulatorInstance<F, Pcs::Commitment, STRATEGY>,
    witness_polys: Vec<Pcs::Polynomial>,
    e_poly: Pcs::Polynomial,
    _marker: PhantomData<Pcs>,
}

impl<F, Pcs, const STRATEGY: usize>
    AsRef<ProtostarAccumulatorInstance<F, Pcs::Commitment, STRATEGY>>
    for ProtostarAccumulator<F, Pcs, STRATEGY>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn as_ref(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment, STRATEGY> {
        &self.instance
    }
}

impl<F, Pcs, const STRATEGY: usize> ProtostarAccumulator<F, Pcs, STRATEGY>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn init(
        k: usize,
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        let zero_poly = Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]);
        Self {
            instance: ProtostarAccumulatorInstance::init(
                num_instances,
                num_witness_polys,
                num_challenges,
            ),
            witness_polys: iter::repeat_with(|| zero_poly.clone())
                .take(num_witness_polys)
                .collect(),
            e_poly: zero_poly,
            _marker: PhantomData,
        }
    }

    fn from_nark(nark: PlonkishNark<F, Pcs>) -> Self {
        let witness_polys = nark.witness_polys;
        let size = witness_polys[0].evals().len();
        Self {
            instance: ProtostarAccumulatorInstance::from_nark(nark.instance),
            witness_polys,
            e_poly: Pcs::Polynomial::from_evals(vec![F::ZERO; size]),
            _marker: PhantomData,
        }
    }

    fn fold_uncompressed(
        &mut self,
        rhs: &Self,
        cross_term_polys: &[Pcs::Polynomial],
        cross_term_comms: &[Pcs::Commitment],
        r: &F,
    ) where
        Pcs::Commitment: AdditiveCommitment<F>,
    {
        self.instance
            .fold_uncompressed(&rhs.instance, cross_term_comms, r);
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), chain![cross_term_polys, [&rhs.e_poly]])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }

    fn fold_compressed(
        &mut self,
        rhs: &Self,
        compressed_cross_term_sums: &[F],
        zeta_cross_term_poly: &Pcs::Polynomial,
        zeta_cross_term_comm: &Pcs::Commitment,
        r: &F,
    ) where
        Pcs::Commitment: AdditiveCommitment<F>,
    {
        self.instance.fold_compressed(
            &rhs.instance,
            compressed_cross_term_sums,
            zeta_cross_term_comm,
            r,
        );
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), [zeta_cross_term_poly, &rhs.e_poly])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }

    pub fn instance(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment, STRATEGY> {
        &self.instance
    }
}

impl<F, Pcs, const STRATEGY: usize> From<PlonkishNark<F, Pcs>>
    for ProtostarAccumulator<F, Pcs, STRATEGY>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn from(nark: PlonkishNark<F, Pcs>) -> Self {
        Self::from_nark(nark)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarAccumulatorInstance<F, C, const STRATEGY: usize = { Compressing as usize }> {
    instances: Vec<Vec<F>>,
    witness_comms: Vec<C>,
    challenges: Vec<F>,
    u: F,
    e_comm: C,
    compressed_e_sum: Option<F>,
}

impl<F, C, const STRATEGY: usize> ProtostarAccumulatorInstance<F, C, STRATEGY>
where
    F: Field,
    C: Default,
{
    fn init(num_instances: &[usize], num_witness_polys: usize, num_challenges: usize) -> Self {
        Self {
            instances: num_instances.iter().map(|n| vec![F::ZERO; *n]).collect(),
            witness_comms: iter::repeat_with(C::default)
                .take(num_witness_polys)
                .collect(),
            challenges: vec![F::ZERO; num_challenges],
            u: F::ZERO,
            e_comm: C::default(),
            compressed_e_sum: match ProtostarStrategy::from(STRATEGY) {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn absorb_into<CommitmentChunk>(
        &self,
        transcript: &mut impl Transcript<CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        C: AsRef<[CommitmentChunk]>,
    {
        self.instances
            .iter()
            .try_for_each(|instances| transcript.common_field_elements(instances))?;
        self.witness_comms
            .iter()
            .try_for_each(|comm| transcript.common_commitments(comm.as_ref()))?;
        transcript.common_field_elements(&self.challenges)?;
        transcript.common_field_element(&self.u)?;
        transcript.common_commitments(self.e_comm.as_ref())?;
        if let Some(compressed_e_sum) = self.compressed_e_sum.as_ref() {
            transcript.common_field_element(compressed_e_sum)?;
        }
        Ok(())
    }

    fn from_nark(nark: PlonkishNarkInstance<F, C>) -> Self {
        Self {
            instances: nark.instances,
            witness_comms: nark.witness_comms,
            challenges: nark.challenges,
            u: F::ONE,
            e_comm: C::default(),
            compressed_e_sum: match ProtostarStrategy::from(STRATEGY) {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn fold_uncompressed(&mut self, rhs: &Self, cross_term_comms: &[C], r: &F)
    where
        C: AdditiveCommitment<F>,
    {
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

    fn fold_compressed(
        &mut self,
        rhs: &Self,
        compressed_cross_term_sums: &[F],
        zeta_cross_term_comm: &C,
        r: &F,
    ) where
        C: AdditiveCommitment<F>,
    {
        let one = F::ONE;
        let powers_of_r = powers(*r)
            .take((compressed_cross_term_sums.len() + 2).max(3))
            .collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::sum_with_scalar([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        *self.compressed_e_sum.as_mut().unwrap() += &inner_product(
            &powers_of_r[1..],
            chain![
                compressed_cross_term_sums,
                [rhs.compressed_e_sum.as_ref().unwrap()]
            ],
        );
        self.e_comm = {
            let comms = [&self.e_comm, zeta_cross_term_comm, &rhs.e_comm];
            C::sum_with_scalar(&powers_of_r[..3], comms)
        };
    }

    fn claimed_sum(&self) -> F {
        self.compressed_e_sum.unwrap_or(F::ZERO)
    }

    fn instances(&self) -> &[Vec<F>] {
        &self.instances
    }
}
