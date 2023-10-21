use crate::{
    accumulation::{
        protostar::ProtostarStrategy::{Compressing, NoCompressing},
        PlonkishNark, PlonkishNarkInstance,
    },
    backend::PlonkishBackend,
    pcs::{Additive, PolynomialCommitmentScheme},
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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum ProtostarStrategy {
    // As known as Sangria
    NoCompressing = 0,
    // Compressing verification as described in 2023/620 section 3.5 but without square-root optimization
    #[default]
    Compressing = 1,
    // TODO:
    // Compressing verification with square-root optimization applied as described in 2023/620 section 3.5
    // CompressingWithSqrtPowers = 3,
}

impl From<usize> for ProtostarStrategy {
    fn from(strategy: usize) -> Self {
        [NoCompressing, Compressing][strategy]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarProverParam<F, Pb>
where
    F: Field,
    Pb: PlonkishBackend<F>,
{
    pp: Pb::ProverParam,
    strategy: ProtostarStrategy,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    cross_term_expressions: Vec<Expression<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarVerifierParam<F, Pb>
where
    F: Field,
    Pb: PlonkishBackend<F>,
{
    vp: Pb::VerifierParam,
    strategy: ProtostarStrategy,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    num_cross_terms: usize,
}

#[derive(Clone, Debug)]
pub struct ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    instance: ProtostarAccumulatorInstance<F, Pcs::Commitment>,
    witness_polys: Vec<Pcs::Polynomial>,
    e_poly: Pcs::Polynomial,
    _marker: PhantomData<Pcs>,
}

impl<F, Pcs> AsRef<ProtostarAccumulatorInstance<F, Pcs::Commitment>>
    for ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn as_ref(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment> {
        &self.instance
    }
}

impl<F, Pcs> ProtostarAccumulator<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn init(
        strategy: ProtostarStrategy,
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        Self {
            instance: ProtostarAccumulatorInstance::init(
                strategy,
                num_instances,
                num_witness_polys,
                num_challenges,
            ),
            witness_polys: iter::repeat_with(Default::default)
                .take(num_witness_polys)
                .collect(),
            e_poly: Default::default(),
            _marker: PhantomData,
        }
    }

    fn from_nark(strategy: ProtostarStrategy, nark: PlonkishNark<F, Pcs>) -> Self {
        let witness_polys = nark.witness_polys;
        Self {
            instance: ProtostarAccumulatorInstance::from_nark(strategy, nark.instance),
            witness_polys,
            e_poly: Default::default(),
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
        Pcs::Commitment: Additive<F>,
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
        zeta_cross_term_poly: &Pcs::Polynomial,
        zeta_cross_term_comm: &Pcs::Commitment,
        compressed_cross_term_sums: &[F],
        r: &F,
    ) where
        Pcs::Commitment: Additive<F>,
    {
        self.instance.fold_compressed(
            &rhs.instance,
            zeta_cross_term_comm,
            compressed_cross_term_sums,
            r,
        );
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), [zeta_cross_term_poly, &rhs.e_poly])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }

    pub fn instance(&self) -> &ProtostarAccumulatorInstance<F, Pcs::Commitment> {
        &self.instance
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtostarAccumulatorInstance<F, C> {
    instances: Vec<Vec<F>>,
    witness_comms: Vec<C>,
    challenges: Vec<F>,
    u: F,
    e_comm: C,
    compressed_e_sum: Option<F>,
}

impl<F, C> ProtostarAccumulatorInstance<F, C> {
    fn instances(&self) -> &[Vec<F>] {
        &self.instances
    }
}

impl<F, C> ProtostarAccumulatorInstance<F, C>
where
    F: Field,
    C: Default,
{
    fn init(
        strategy: ProtostarStrategy,
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
            compressed_e_sum: match strategy {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn claimed_sum(&self) -> F {
        self.compressed_e_sum.unwrap_or(F::ZERO)
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

    fn from_nark(strategy: ProtostarStrategy, nark: PlonkishNarkInstance<F, C>) -> Self {
        Self {
            instances: nark.instances,
            witness_comms: nark.witness_comms,
            challenges: nark.challenges,
            u: F::ONE,
            e_comm: C::default(),
            compressed_e_sum: match strategy {
                NoCompressing => None,
                Compressing => Some(F::ZERO),
            },
        }
    }

    fn fold_uncompressed(&mut self, rhs: &Self, cross_term_comms: &[C], r: &F)
    where
        C: Additive<F>,
    {
        let one = F::ONE;
        let powers_of_r = powers(*r).take(cross_term_comms.len() + 2).collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::msm([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.e_comm = {
            let comms = chain![[&self.e_comm], cross_term_comms, [&rhs.e_comm]];
            C::msm(&powers_of_r, comms)
        };
    }

    fn fold_compressed(
        &mut self,
        rhs: &Self,
        zeta_cross_term_comm: &C,
        compressed_cross_term_sums: &[F],
        r: &F,
    ) where
        C: Additive<F>,
    {
        let one = F::ONE;
        let powers_of_r = powers(*r)
            .take(compressed_cross_term_sums.len().max(1) + 2)
            .collect_vec();
        izip_eq!(&mut self.instances, &rhs.instances)
            .for_each(|(lhs, rhs)| izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| *lhs += &(*rhs * r)));
        izip_eq!(&mut self.witness_comms, &rhs.witness_comms)
            .for_each(|(lhs, rhs)| *lhs = C::msm([&one, r], [lhs, rhs]));
        izip_eq!(&mut self.challenges, &rhs.challenges).for_each(|(lhs, rhs)| *lhs += &(*rhs * r));
        self.u += &(rhs.u * r);
        self.e_comm = {
            let comms = [&self.e_comm, zeta_cross_term_comm, &rhs.e_comm];
            C::msm(&powers_of_r[..3], comms)
        };
        *self.compressed_e_sum.as_mut().unwrap() += &inner_product(
            &powers_of_r[1..],
            chain![
                compressed_cross_term_sums,
                [rhs.compressed_e_sum.as_ref().unwrap()]
            ],
        );
    }
}
