use crate::{
    backend::{hyperplonk::HyperPlonk, PlonkishBackend, PlonkishCircuit},
    folding::{
        protostar::{
            ivc::ProtostarAccumulationVerifierParam,
            PlonkishNarkInstance, Protostar, ProtostarAccumulator, ProtostarAccumulatorInstance,
            ProtostarProverParam,
            ProtostarStrategy::{Compressing, NoCompressing},
            ProtostarVerifierParam,
        },
        FoldingScheme,
    },
    frontend::halo2::{CircuitExt, Halo2Circuit},
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{
            fe_to_fe, fe_truncated_from_le_bytes, Field, PrimeCurveAffine, PrimeField,
            TwoChainCurve,
        },
        end_timer,
        hash::{Hash as _, Keccak256},
        izip_eq, start_timer,
        transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use rand::RngCore;
use std::{
    borrow::{Borrow, Cow},
    fmt::Debug,
    hash::Hash,
    iter,
    marker::PhantomData,
};

#[cfg(test)]
mod test;

type AssignedPlonkishNarkInstance<AssignedScalar, AssignedEcPoint> =
    PlonkishNarkInstance<AssignedScalar, AssignedEcPoint>;

type AssignedProtostarAccumulatorInstance<AssignedScalar, AssignedEcPoint> =
    ProtostarAccumulatorInstance<AssignedScalar, AssignedEcPoint>;

pub trait TwoChainCurveInstruction<C: TwoChainCurve>: Clone + Debug {
    type Config: Clone + Debug;
    type Assigned: Clone + Debug;
    type AssignedBase: Clone + Debug;
    type AssignedPrimary: Clone + Debug;
    type AssignedSecondary: Clone + Debug;

    fn new(config: Self::Config) -> Self;

    fn to_assigned(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        assigned: &AssignedCell<C::Scalar, C::Scalar>,
    ) -> Result<Self::Assigned, Error>;

    fn constrain_equal(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<(), Error>;

    fn constrain_instance(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::Assigned,
        row: usize,
    ) -> Result<(), Error>;

    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        constant: C::Scalar,
    ) -> Result<Self::Assigned, Error>;

    fn assign_witness(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        witness: Value<C::Scalar>,
    ) -> Result<Self::Assigned, Error>;

    fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C::Scalar) -> bool);

    fn select(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        condition: &Self::Assigned,
        when_true: &Self::Assigned,
        when_false: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn is_equal(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn add(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn mul(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn assign_constant_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        constant: C::Base,
    ) -> Result<Self::AssignedBase, Error>;

    fn assign_witness_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        witness: Value<C::Base>,
    ) -> Result<Self::AssignedBase, Error>;

    fn assert_if_known_base(&self, value: &Self::AssignedBase, f: impl FnOnce(&C::Base) -> bool);

    fn select_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        condition: &Self::Assigned,
        when_true: &Self::AssignedBase,
        when_false: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn fit_base_in_scalar(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::AssignedBase,
    ) -> Result<Self::Assigned, Error>;

    fn add_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn mul_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn powers_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        base: &Self::AssignedBase,
        n: usize,
    ) -> Result<Vec<Self::AssignedBase>, Error> {
        Ok(match n {
            0 => Vec::new(),
            1 => vec![self.assign_constant_base(layouter, C::Base::ONE)?],
            2 => vec![
                self.assign_constant_base(layouter, C::Base::ONE)?,
                base.clone(),
            ],
            _ => {
                let mut powers = Vec::with_capacity(n);
                powers.push(self.assign_constant_base(layouter, C::Base::ONE)?);
                powers.push(base.clone());
                for _ in 0..n - 2 {
                    powers.push(self.mul_base(layouter, powers.last().unwrap(), base)?);
                }
                powers
            }
        })
    }

    fn inner_product_base<'a, 'b>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: impl IntoIterator<Item = &'a Self::AssignedBase>,
        rhs: impl IntoIterator<Item = &'b Self::AssignedBase>,
    ) -> Result<Self::AssignedBase, Error>
    where
        Self::AssignedBase: 'a + 'b,
    {
        let products = izip_eq!(lhs, rhs)
            .map(|(lhs, rhs)| self.mul_base(layouter, lhs, rhs))
            .collect_vec();
        products
            .into_iter()
            .reduce(|acc, output| self.add_base(layouter, &acc?, &output?))
            .unwrap()
    }

    fn assign_constant_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        constant: C,
    ) -> Result<Self::AssignedPrimary, Error>;

    fn assign_witness_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        witness: Value<C>,
    ) -> Result<Self::AssignedPrimary, Error>;

    fn assert_if_known_primary(&self, value: &Self::AssignedPrimary, f: impl FnOnce(&C) -> bool);

    fn select_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        condition: &Self::Assigned,
        when_true: &Self::AssignedPrimary,
        when_false: &Self::AssignedPrimary,
    ) -> Result<Self::AssignedPrimary, Error>;

    fn add_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedPrimary,
        rhs: &Self::AssignedPrimary,
    ) -> Result<Self::AssignedPrimary, Error>;

    fn scalar_mul_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        base: &Self::AssignedPrimary,
        scalar_le_bits: &[Self::Assigned],
    ) -> Result<Self::AssignedPrimary, Error>;

    fn fixed_base_msm_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: &[C],
        scalars: &[Self::Assigned],
    ) -> Result<Self::AssignedPrimary, Error>;

    fn variable_base_msm_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: &[Self::AssignedPrimary],
        scalars: &[Self::Assigned],
    ) -> Result<Self::AssignedPrimary, Error>;

    fn assign_constant_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        constant: C::Secondary,
    ) -> Result<Self::AssignedSecondary, Error>;

    fn assign_witness_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        witness: Value<C::Secondary>,
    ) -> Result<Self::AssignedSecondary, Error>;

    fn assert_if_known_secondary(
        &self,
        value: &Self::AssignedSecondary,
        f: impl FnOnce(&C::Secondary) -> bool,
    );

    fn select_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        condition: &Self::Assigned,
        when_true: &Self::AssignedSecondary,
        when_false: &Self::AssignedSecondary,
    ) -> Result<Self::AssignedSecondary, Error>;

    fn add_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedSecondary,
        rhs: &Self::AssignedSecondary,
    ) -> Result<Self::AssignedSecondary, Error>;

    fn scalar_mul_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        base: &Self::AssignedSecondary,
        scalar_le_bits: &[Self::Assigned],
    ) -> Result<Self::AssignedSecondary, Error>;

    fn fixed_base_msm_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: &[C::Secondary],
        scalars: &[Self::AssignedBase],
    ) -> Result<Self::AssignedSecondary, Error>;

    fn variable_base_msm_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: &[Self::AssignedSecondary],
        scalars: &[Self::AssignedBase],
    ) -> Result<Self::AssignedSecondary, Error>;
}

pub trait HashInstruction<C: TwoChainCurve>: Clone + Debug {
    const NUM_HASH_BITS: usize;

    type Param: Clone + Debug;
    type Config: Clone + Debug + Borrow<Self::Param>;
    type TccChip: TwoChainCurveInstruction<C>;

    fn new(config: Self::Config, chip: Self::TccChip) -> Self;

    fn hash_state<Comm: AsRef<C::Secondary>>(
        param: &Self::Param,
        vp_digest: C::Scalar,
        step_idx: usize,
        initial_input: &[C::Scalar],
        output: &[C::Scalar],
        acc: &ProtostarAccumulatorInstance<C::Base, Comm>,
    ) -> C::Scalar;

    #[allow(clippy::type_complexity)]
    fn hash_assigned_state(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        vp_digest: &<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        step_idx: &<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        initial_input: &[<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned],
        output: &[<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned],
        acc: &AssignedProtostarAccumulatorInstance<
            <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase,
            <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary,
        >,
    ) -> Result<<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned, Error>;
}

pub trait TranscriptInstruction<C: TwoChainCurve>: Debug {
    type Config: Clone + Debug;
    type TccChip: TwoChainCurveInstruction<C>;
    type Challenge: Clone
        + Debug
        + AsRef<<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase>;

    fn new(config: Self::Config, chip: Self::TccChip, proof: Value<Vec<u8>>) -> Self;

    fn dummy_proof(avp: &ProtostarAccumulationVerifierParam<C::Scalar>) -> Vec<u8> {
        let uncompressed_comm_size = C::Scalar::ZERO.to_repr().as_ref().len() * 2;
        let scalar_size = C::Base::ZERO.to_repr().as_ref().len();
        let proof_size = avp.num_folding_witness_polys() * uncompressed_comm_size
            + match avp.strategy {
                NoCompressing => avp.num_cross_terms * uncompressed_comm_size,
                Compressing => uncompressed_comm_size + avp.num_cross_terms * scalar_size,
            };
        vec![0; proof_size]
    }

    fn challenge_to_le_bits(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        scalar: &Self::Challenge,
    ) -> Result<Vec<<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned>, Error>;

    fn squeeze_challenge(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
    ) -> Result<Self::Challenge, Error>;

    fn squeeze_challenges(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
        n: usize,
    ) -> Result<Vec<Self::Challenge>, Error> {
        (0..n).map(|_| self.squeeze_challenge(layouter)).collect()
    }

    fn common_field_element(
        &mut self,
        fe: &<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase,
    ) -> Result<(), Error>;

    fn common_field_elements(
        &mut self,
        fes: &[<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase],
    ) -> Result<(), Error> {
        fes.iter().try_for_each(|fe| self.common_field_element(fe))
    }

    fn read_field_element(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
    ) -> Result<<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase, Error>;

    fn read_field_elements(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
        n: usize,
    ) -> Result<Vec<<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase>, Error> {
        (0..n).map(|_| self.read_field_element(layouter)).collect()
    }

    fn common_commitment(
        &mut self,
        comm: &<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary,
    ) -> Result<(), Error>;

    fn common_commitments(
        &mut self,
        comms: &[<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary],
    ) -> Result<(), Error> {
        comms
            .iter()
            .try_for_each(|comm| self.common_commitment(comm))
    }

    fn read_commitment(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
    ) -> Result<<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary, Error>;

    fn read_commitments(
        &mut self,
        layouter: &mut impl Layouter<C::Scalar>,
        n: usize,
    ) -> Result<Vec<<Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary>, Error> {
        (0..n).map(|_| self.read_commitment(layouter)).collect()
    }

    #[allow(clippy::type_complexity)]
    fn absorb_accumulator(
        &mut self,
        acc: &AssignedProtostarAccumulatorInstance<
            <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase,
            <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary,
        >,
    ) -> Result<(), Error> {
        acc.instances
            .iter()
            .try_for_each(|instances| self.common_field_elements(instances))?;
        self.common_commitments(&acc.witness_comms)?;
        self.common_field_elements(&acc.challenges)?;
        self.common_field_element(&acc.u)?;
        self.common_commitment(&acc.e_comm)?;
        if let Some(compressed_e_sum) = acc.compressed_e_sum.as_ref() {
            self.common_field_element(compressed_e_sum)?;
        }
        Ok(())
    }
}

pub trait StepCircuit<C: TwoChainCurve>: Clone + Debug + CircuitExt<C::Scalar> {
    type TccChip: TwoChainCurveInstruction<C>;
    type HashChip: HashInstruction<C, TccChip = Self::TccChip>;
    type TranscriptChip: TranscriptInstruction<C, TccChip = Self::TccChip>;

    #[allow(clippy::type_complexity)]
    fn configs(
        config: Self::Config,
    ) -> (
        <Self::TccChip as TwoChainCurveInstruction<C>>::Config,
        <Self::HashChip as HashInstruction<C>>::Config,
        <Self::TranscriptChip as TranscriptInstruction<C>>::Config,
    );

    fn arity() -> usize;

    fn initial_input(&self) -> &[C::Scalar];

    fn input(&self) -> &[C::Scalar];

    fn output(&self) -> &[C::Scalar];

    fn step_idx(&self) -> usize;

    fn next(&mut self);

    #[allow(clippy::type_complexity)]
    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<C::Scalar>,
    ) -> Result<
        (
            Vec<AssignedCell<C::Scalar, C::Scalar>>,
            Vec<AssignedCell<C::Scalar, C::Scalar>>,
        ),
        Error,
    >;
}

pub struct ProtostarAccumulationVerifier<C: TwoChainCurve, TccChip> {
    avp: ProtostarAccumulationVerifierParam<C::Scalar>,
    tcc_chip: TccChip,
    _marker: PhantomData<C>,
}

impl<C, TccChip> ProtostarAccumulationVerifier<C, TccChip>
where
    C: TwoChainCurve,
    TccChip: TwoChainCurveInstruction<C>,
{
    pub fn new(avp: ProtostarAccumulationVerifierParam<C::Scalar>, tcc_chip: TccChip) -> Self {
        Self {
            avp,
            tcc_chip,
            _marker: PhantomData,
        }
    }

    pub fn assign_default_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
    ) -> Result<
        AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let ProtostarAccumulationVerifierParam { num_instances, .. } = &self.avp;

        let instances = num_instances
            .iter()
            .map(|num_instances| {
                iter::repeat_with(|| tcc_chip.assign_constant_base(layouter, C::Base::ZERO))
                    .take(*num_instances)
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = iter::repeat_with(|| {
            tcc_chip.assign_constant_secondary(layouter, C::Secondary::identity())
        })
        .take(self.avp.num_folding_witness_polys())
        .try_collect::<_, Vec<_>, _>()?;
        let challenges =
            iter::repeat_with(|| tcc_chip.assign_constant_base(layouter, C::Base::ZERO))
                .take(self.avp.num_folding_challenges())
                .try_collect::<_, Vec<_>, _>()?;
        let u = tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?;
        let e_comm = tcc_chip.assign_constant_secondary(layouter, C::Secondary::identity())?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?),
        };

        Ok(ProtostarAccumulatorInstance {
            instances,
            witness_comms,
            challenges,
            u,
            e_comm,
            compressed_e_sum,
        })
    }

    pub fn assign_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        acc: Value<&ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
    ) -> Result<
        AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let avp = &self.avp;

        let instances = avp
            .num_instances
            .iter()
            .zip(
                acc.map(|acc| &acc.instances)
                    .transpose_vec(avp.num_instances.len()),
            )
            .map(|(num_instances, instances)| {
                instances
                    .transpose_vec(*num_instances)
                    .into_iter()
                    .map(|instance| tcc_chip.assign_witness_base(layouter, instance.copied()))
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = acc
            .map(|acc| &acc.witness_comms)
            .transpose_vec(avp.num_folding_witness_polys())
            .into_iter()
            .map(|witness_comm| tcc_chip.assign_witness_secondary(layouter, witness_comm.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let challenges = acc
            .map(|acc| &acc.challenges)
            .transpose_vec(avp.num_folding_challenges())
            .into_iter()
            .map(|challenge| tcc_chip.assign_witness_base(layouter, challenge.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let u = tcc_chip.assign_witness_base(layouter, acc.map(|acc| &acc.u).copied())?;
        let e_comm = tcc_chip.assign_witness_secondary(layouter, acc.map(|acc| acc.e_comm))?;
        let compressed_e_sum = match avp.strategy {
            NoCompressing => None,
            Compressing => Some(
                tcc_chip
                    .assign_witness_base(layouter, acc.map(|acc| acc.compressed_e_sum.unwrap()))?,
            ),
        };

        Ok(ProtostarAccumulatorInstance {
            instances,
            witness_comms,
            challenges,
            u,
            e_comm,
            compressed_e_sum,
        })
    }

    fn assign_accumulator_from_r_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        r: &TccChip::AssignedBase,
        r_nark: AssignedPlonkishNarkInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
    ) -> Result<
        AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let AssignedPlonkishNarkInstance {
            instances,
            challenges,
            witness_comms,
        } = r_nark;
        let u = r.clone();
        let e_comm = tcc_chip.assign_constant_secondary(layouter, C::Secondary::identity())?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?),
        };

        Ok(ProtostarAccumulatorInstance {
            instances,
            witness_comms,
            challenges,
            u,
            e_comm,
            compressed_e_sum,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn verify_accumulation_from_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        acc: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
        instances: [Value<&C::Base>; 2],
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            AssignedPlonkishNarkInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
            AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
            AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let ProtostarAccumulationVerifierParam {
            strategy,
            num_witness_polys,
            num_challenges,
            num_cross_terms,
            ..
        } = &self.avp;

        let instances = instances
            .into_iter()
            .map(|instance| tcc_chip.assign_witness_base(layouter, instance.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        for instance in instances.iter() {
            transcript.common_field_element(instance)?;
        }

        let mut witness_comms = Vec::with_capacity(self.avp.num_folding_witness_polys());
        let mut challenges = Vec::with_capacity(self.avp.num_folding_challenges());
        for (num_witness_polys, num_powers_of_challenge) in
            num_witness_polys.iter().zip_eq(num_challenges.iter())
        {
            witness_comms.extend(transcript.read_commitments(layouter, *num_witness_polys)?);
            for num_powers in num_powers_of_challenge.iter() {
                let challenge = transcript.squeeze_challenge(layouter)?;
                let powers_of_challenges =
                    tcc_chip.powers_base(layouter, challenge.as_ref(), *num_powers + 1)?;
                challenges.extend(powers_of_challenges.into_iter().skip(1));
            }
        }

        let nark = PlonkishNarkInstance::new(vec![instances], challenges, witness_comms);
        transcript.absorb_accumulator(acc)?;

        let (cross_term_comms, compressed_cross_term_sums) = match strategy {
            NoCompressing => {
                let cross_term_comms = transcript.read_commitments(layouter, *num_cross_terms)?;

                (cross_term_comms, None)
            }
            Compressing => {
                let zeta_cross_term_comm = vec![transcript.read_commitment(layouter)?];
                let compressed_cross_term_sums =
                    transcript.read_field_elements(layouter, *num_cross_terms)?;

                (zeta_cross_term_comm, Some(compressed_cross_term_sums))
            }
        };

        let r = transcript.squeeze_challenge(layouter)?;
        let r_le_bits = transcript.challenge_to_le_bits(layouter, &r)?;

        let (r_nark, acc_prime) = self.fold_accumulator_from_nark(
            layouter,
            acc,
            &nark,
            &cross_term_comms,
            compressed_cross_term_sums.as_deref(),
            r.as_ref(),
            &r_le_bits,
        )?;
        let acc_r_nark = self.assign_accumulator_from_r_nark(layouter, r.as_ref(), r_nark)?;

        Ok((nark, acc_r_nark, acc_prime))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn fold_accumulator_from_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        acc: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
        nark: &AssignedPlonkishNarkInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        cross_term_comms: &[TccChip::AssignedSecondary],
        compressed_cross_term_sums: Option<&[TccChip::AssignedBase]>,
        r: &TccChip::AssignedBase,
        r_le_bits: &[TccChip::Assigned],
    ) -> Result<
        (
            AssignedPlonkishNarkInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
            AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let ProtostarAccumulationVerifierParam {
            strategy,
            num_cross_terms,
            ..
        } = self.avp;

        let powers_of_r = tcc_chip.powers_base(layouter, r, num_cross_terms + 1)?;

        let r_nark = {
            let instances = nark
                .instances
                .iter()
                .map(|instances| {
                    instances
                        .iter()
                        .map(|instance| tcc_chip.mul_base(layouter, r, instance))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = nark
                .witness_comms
                .iter()
                .map(|comm| tcc_chip.scalar_mul_secondary(layouter, comm, r_le_bits))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = nark
                .challenges
                .iter()
                .map(|challenge| tcc_chip.mul_base(layouter, r, challenge))
                .try_collect::<_, Vec<_>, _>()?;
            AssignedPlonkishNarkInstance {
                instances,
                challenges,
                witness_comms,
            }
        };

        let acc_prime = {
            let instances = izip_eq!(&acc.instances, &r_nark.instances)
                .map(|(lhs, rhs)| {
                    izip_eq!(lhs, rhs)
                        .map(|(lhs, rhs)| tcc_chip.add_base(layouter, lhs, rhs))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = izip_eq!(&acc.witness_comms, &r_nark.witness_comms)
                .map(|(lhs, rhs)| tcc_chip.add_secondary(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = izip_eq!(&acc.challenges, &r_nark.challenges)
                .map(|(lhs, rhs)| tcc_chip.add_base(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let u = tcc_chip.add_base(layouter, &acc.u, r)?;
            let e_comm = if cross_term_comms.is_empty() {
                acc.e_comm.clone()
            } else {
                let mut e_comm = cross_term_comms.last().unwrap().clone();
                for item in cross_term_comms.iter().rev().skip(1).chain([&acc.e_comm]) {
                    e_comm = tcc_chip.scalar_mul_secondary(layouter, &e_comm, r_le_bits)?;
                    e_comm = tcc_chip.add_secondary(layouter, &e_comm, item)?;
                }
                e_comm
            };
            let compressed_e_sum = match strategy {
                NoCompressing => None,
                Compressing => {
                    let rhs = tcc_chip.inner_product_base(
                        layouter,
                        &powers_of_r[1..],
                        compressed_cross_term_sums.unwrap(),
                    )?;
                    Some(tcc_chip.add_base(
                        layouter,
                        acc.compressed_e_sum.as_ref().unwrap(),
                        &rhs,
                    )?)
                }
            };

            ProtostarAccumulatorInstance {
                instances,
                witness_comms,
                challenges,
                u,
                e_comm,
                compressed_e_sum,
            }
        };

        Ok((r_nark, acc_prime))
    }

    fn select_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        condition: &TccChip::Assigned,
        when_true: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
        when_false: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
    ) -> Result<
        AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        Error,
    > {
        let tcc_chip = &self.tcc_chip;

        let instances = izip_eq!(&when_true.instances, &when_false.instances)
            .map(|(when_true, when_false)| {
                izip_eq!(when_true, when_false)
                    .map(|(when_true, when_false)| {
                        tcc_chip.select_base(layouter, condition, when_true, when_false)
                    })
                    .try_collect()
            })
            .try_collect()?;
        let witness_comms = izip_eq!(&when_true.witness_comms, &when_false.witness_comms)
            .map(|(when_true, when_false)| {
                tcc_chip.select_secondary(layouter, condition, when_true, when_false)
            })
            .try_collect()?;
        let challenges = izip_eq!(&when_true.challenges, &when_false.challenges)
            .map(|(when_true, when_false)| {
                tcc_chip.select_base(layouter, condition, when_true, when_false)
            })
            .try_collect()?;
        let u = tcc_chip.select_base(layouter, condition, &when_true.u, &when_false.u)?;
        let e_comm = tcc_chip.select_secondary(
            layouter,
            condition,
            &when_true.e_comm,
            &when_false.e_comm,
        )?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(tcc_chip.select_base(
                layouter,
                condition,
                when_true.compressed_e_sum.as_ref().unwrap(),
                when_false.compressed_e_sum.as_ref().unwrap(),
            )?),
        };

        Ok(ProtostarAccumulatorInstance {
            instances,
            witness_comms,
            challenges,
            u,
            e_comm,
            compressed_e_sum,
        })
    }
}

#[derive(Debug)]
pub struct RecursiveCircuit<C, Sc>
where
    C: TwoChainCurve,
    Sc: StepCircuit<C>,
{
    is_primary: bool,
    step_circuit: Sc,
    tcc_chip: Sc::TccChip,
    hash_chip: Sc::HashChip,
    hash_config: <Sc::HashChip as HashInstruction<C>>::Config,
    transcript_config: <Sc::TranscriptChip as TranscriptInstruction<C>>::Config,
    avp: ProtostarAccumulationVerifierParam<C::Scalar>,
    h_prime: Value<C::Scalar>,
    acc: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
    acc_prime: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
    incoming_instances: [Value<C::Base>; 2],
    incoming_proof: Value<Vec<u8>>,
}

impl<C, Sc> RecursiveCircuit<C, Sc>
where
    C: TwoChainCurve,
    Sc: StepCircuit<C>,
{
    pub const DUMMY_H: C::Scalar = C::Scalar::ZERO;

    pub fn new(
        is_primary: bool,
        step_circuit: Sc,
        avp: Option<ProtostarAccumulationVerifierParam<C::Scalar>>,
    ) -> Self {
        let config = Self::configure(&mut Default::default());
        let (tcc_config, hash_config, transcript_config) = Sc::configs(config);
        let tcc_chip = Sc::TccChip::new(tcc_config);
        let hash_chip = Sc::HashChip::new(hash_config.clone(), tcc_chip.clone());
        Self {
            is_primary,
            step_circuit,
            tcc_chip,
            hash_chip,
            hash_config,
            transcript_config,
            avp: avp.unwrap_or_default(),
            h_prime: Value::unknown(),
            acc: Value::unknown(),
            acc_prime: Value::unknown(),
            incoming_instances: [Value::unknown(); 2],
            incoming_proof: Value::unknown(),
        }
    }

    pub fn update<Comm: AsRef<C::Secondary>>(
        &mut self,
        acc: ProtostarAccumulatorInstance<C::Base, Comm>,
        acc_prime: ProtostarAccumulatorInstance<C::Base, Comm>,
        incoming_instances: [C::Base; 2],
        incoming_proof: Vec<u8>,
    ) {
        if (self.is_primary && acc_prime.u != C::Base::ZERO)
            || (!self.is_primary && acc.u != C::Base::ZERO)
        {
            self.step_circuit.next();
        }
        self.h_prime = Value::known(Sc::HashChip::hash_state(
            self.hash_config.borrow(),
            self.avp.vp_digest.unwrap_or_default(),
            self.step_circuit.step_idx() + 1,
            self.step_circuit.initial_input(),
            self.step_circuit.output(),
            &acc_prime,
        ));
        let convert =
            |acc: ProtostarAccumulatorInstance<C::Base, Comm>| ProtostarAccumulatorInstance {
                instances: acc.instances,
                witness_comms: iter::empty()
                    .chain(acc.witness_comms.iter().map(AsRef::as_ref).copied())
                    .collect(),
                challenges: acc.challenges,
                u: acc.u,
                e_comm: *acc.e_comm.as_ref(),
                compressed_e_sum: acc.compressed_e_sum,
            };
        self.acc = Value::known(convert(acc));
        self.acc_prime = Value::known(convert(acc_prime));
        self.incoming_instances = incoming_instances.map(Value::known);
        self.incoming_proof = Value::known(incoming_proof);
    }

    fn init(&mut self, vp_digest: C::Scalar) {
        assert_eq!(&self.avp.num_instances, &[2]);
        self.avp.vp_digest = Some(vp_digest);
        self.update::<Cow<C::Secondary>>(
            self.avp.init_accumulator(),
            self.avp.init_accumulator(),
            [Self::DUMMY_H; 2].map(fe_to_fe),
            Sc::TranscriptChip::dummy_proof(&self.avp),
        );
    }

    fn check_initial_condition(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        is_base_case: &<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        initial_input: &[<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned],
        input: &[<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned],
    ) -> Result<(), Error> {
        let tcc_chip = &self.tcc_chip;
        let zero = tcc_chip.assign_constant(layouter, C::Scalar::ZERO)?;

        for (lhs, rhs) in input.iter().zip(initial_input.iter()) {
            let lhs = tcc_chip.select(layouter, is_base_case, lhs, &zero)?;
            let rhs = tcc_chip.select(layouter, is_base_case, rhs, &zero)?;
            tcc_chip.constrain_equal(layouter, &lhs, &rhs)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn check_state_hash(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        is_base_case: Option<&<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned>,
        h: &<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        vp_digest: &<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        step_idx: &<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned,
        initial_input: &[<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned],
        output: &[<Sc::TccChip as TwoChainCurveInstruction<C>>::Assigned],
        acc: &AssignedProtostarAccumulatorInstance<
            <Sc::TccChip as TwoChainCurveInstruction<C>>::AssignedBase,
            <Sc::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary,
        >,
    ) -> Result<(), Error> {
        let tcc_chip = &self.tcc_chip;
        let hash_chip = &self.hash_chip;
        let lhs = h;
        let rhs = hash_chip.hash_assigned_state(
            layouter,
            vp_digest,
            step_idx,
            initial_input,
            output,
            acc,
        )?;
        let rhs = if let Some(is_base_case) = is_base_case {
            let dummy_h = tcc_chip.assign_constant(layouter, Self::DUMMY_H)?;
            tcc_chip.select(layouter, is_base_case, &dummy_h, &rhs)?
        } else {
            rhs
        };
        tcc_chip.constrain_equal(layouter, lhs, &rhs)?;
        Ok(())
    }

    fn synthesize_accumulation_verifier(
        &self,
        mut layouter: impl Layouter<C::Scalar>,
        input: &[AssignedCell<C::Scalar, C::Scalar>],
        output: &[AssignedCell<C::Scalar, C::Scalar>],
    ) -> Result<(), Error> {
        let Self {
            tcc_chip,
            transcript_config,
            avp,
            ..
        } = &self;

        let layouter = &mut layouter;

        let acc_verifier = ProtostarAccumulationVerifier::new(avp.clone(), tcc_chip.clone());

        let zero = tcc_chip.assign_constant(layouter, C::Scalar::ZERO)?;
        let one = tcc_chip.assign_constant(layouter, C::Scalar::ONE)?;
        let vp_digest = tcc_chip.assign_witness(
            layouter,
            avp.vp_digest.map(Value::known).unwrap_or_default(),
        )?;
        let step_idx = tcc_chip.assign_witness(
            layouter,
            Value::known(C::Scalar::from(self.step_circuit.step_idx() as u64)),
        )?;
        let step_idx_plus_one = tcc_chip.add(layouter, &step_idx, &one)?;
        let initial_input = self
            .step_circuit
            .initial_input()
            .iter()
            .map(|value| tcc_chip.assign_witness(layouter, Value::known(*value)))
            .try_collect::<_, Vec<_>, _>()?;
        let input = input
            .iter()
            .map(|assigned| tcc_chip.to_assigned(layouter, assigned))
            .try_collect::<_, Vec<_>, _>()?;
        let output = output
            .iter()
            .map(|assigned| tcc_chip.to_assigned(layouter, assigned))
            .try_collect::<_, Vec<_>, _>()?;

        let is_base_case = tcc_chip.is_equal(layouter, &step_idx, &zero)?;
        let h_prime = tcc_chip.assign_witness(layouter, self.h_prime)?;

        self.check_initial_condition(layouter, &is_base_case, &initial_input, &input)?;

        let acc = acc_verifier.assign_accumulator(layouter, self.acc.as_ref())?;

        let (nark, acc_r_nark, acc_prime) = {
            let instances =
                [&self.incoming_instances[0], &self.incoming_instances[1]].map(Value::as_ref);
            let proof = self.incoming_proof.clone();
            let transcript =
                &mut Sc::TranscriptChip::new(transcript_config.clone(), tcc_chip.clone(), proof);
            acc_verifier.verify_accumulation_from_nark(layouter, &acc, instances, transcript)?
        };

        let acc_prime = {
            let acc_default = if self.is_primary {
                acc_verifier.assign_default_accumulator(layouter)?
            } else {
                acc_r_nark
            };
            acc_verifier.select_accumulator(layouter, &is_base_case, &acc_default, &acc_prime)?
        };

        let h_from_incoming = tcc_chip.fit_base_in_scalar(layouter, &nark.instances[0][0])?;
        let h_ohs_from_incoming = tcc_chip.fit_base_in_scalar(layouter, &nark.instances[0][1])?;

        self.check_state_hash(
            layouter,
            Some(&is_base_case),
            &h_from_incoming,
            &vp_digest,
            &step_idx,
            &initial_input,
            &input,
            &acc,
        )?;
        self.check_state_hash(
            layouter,
            None,
            &h_prime,
            &vp_digest,
            &step_idx_plus_one,
            &initial_input,
            &output,
            &acc_prime,
        )?;

        tcc_chip.constrain_instance(layouter, &h_ohs_from_incoming, 0)?;
        tcc_chip.constrain_instance(layouter, &h_prime, 1)?;

        Ok(())
    }
}

impl<C, Sc> Circuit<C::Scalar> for RecursiveCircuit<C, Sc>
where
    C: TwoChainCurve,
    Sc: StepCircuit<C>,
{
    type Config = Sc::Config;
    type FloorPlanner = Sc::FloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            is_primary: self.is_primary,
            step_circuit: self.step_circuit.without_witnesses(),
            tcc_chip: self.tcc_chip.clone(),
            hash_chip: self.hash_chip.clone(),
            hash_config: self.hash_config.clone(),
            transcript_config: self.transcript_config.clone(),
            avp: self.avp.clone(),
            h_prime: Value::unknown(),
            acc: Value::unknown(),
            acc_prime: Value::unknown(),
            incoming_instances: [Value::unknown(), Value::unknown()],
            incoming_proof: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<C::Scalar>) -> Self::Config {
        Sc::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<C::Scalar>,
    ) -> Result<(), Error> {
        let (input, output) =
            StepCircuit::synthesize(&self.step_circuit, config, layouter.namespace(|| ""))?;
        self.synthesize_accumulation_verifier(layouter.namespace(|| ""), &input, &output)?;
        Ok(())
    }
}

impl<C, Sc> CircuitExt<C::Scalar> for RecursiveCircuit<C, Sc>
where
    C: TwoChainCurve,
    Sc: StepCircuit<C>,
{
    fn instances(&self) -> Vec<Vec<C::Scalar>> {
        let mut instances = vec![vec![Self::DUMMY_H; 2]];
        self.incoming_instances[1].map(|h_ohs| instances[0][0] = fe_to_fe(h_ohs));
        self.h_prime.map(|h_prime| instances[0][1] = h_prime);
        instances
    }
}

#[derive(Debug)]
pub struct ProtostarIvcProverParam<C, P1, P2, AT1, AT2>
where
    C: TwoChainCurve,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
    AT1: InMemoryTranscript,
    AT2: InMemoryTranscript,
{
    primary_pp: ProtostarProverParam<C::Scalar, HyperPlonk<P1>>,
    primary_atp: AT1::Param,
    secondary_pp: ProtostarProverParam<C::Base, HyperPlonk<P2>>,
    secondary_atp: AT2::Param,
    _marker: PhantomData<(C, AT1, AT2)>,
}

#[derive(Debug)]
pub struct ProtostarIvcVerifierParam<C, P1, P2, HP1, HP2>
where
    C: TwoChainCurve,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
{
    vp_digest: C::Scalar,
    primary_vp: ProtostarVerifierParam<C::Scalar, HyperPlonk<P1>>,
    primary_avp: ProtostarAccumulationVerifierParam<C::Scalar>,
    primary_hp: HP1,
    primary_arity: usize,
    secondary_vp: ProtostarVerifierParam<C::Base, HyperPlonk<P2>>,
    secondary_hp: HP2,
    secondary_arity: usize,
    _marker: PhantomData<C>,
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn preprocess<C, P1, P2, S1, S2, AT1, AT2>(
    primary_num_vars: usize,
    primary_atp: AT1::Param,
    primary_step_circuit: S1,
    secondary_num_vars: usize,
    secondary_atp: AT2::Param,
    secondary_step_circuit: S2,
    mut rng: impl RngCore,
) -> Result<
    (
        Halo2Circuit<C::Scalar, RecursiveCircuit<C, S1>>,
        Halo2Circuit<C::Base, RecursiveCircuit<C::Secondary, S2>>,
        ProtostarIvcProverParam<C, P1, P2, AT1, AT2>,
        ProtostarIvcVerifierParam<
            C,
            P1,
            P2,
            <S1::HashChip as HashInstruction<C>>::Param,
            <S2::HashChip as HashInstruction<C::Secondary>>::Param,
        >,
    ),
    Error,
>
where
    C: TwoChainCurve,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::ScalarExt,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C> + From<C>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    S1: StepCircuit<C>,
    S2: StepCircuit<C::Secondary>,
    AT1: InMemoryTranscript,
    AT2: InMemoryTranscript,
{
    assert_eq!(S1::HashChip::NUM_HASH_BITS, S2::HashChip::NUM_HASH_BITS);

    let primary_param = P1::setup(1 << primary_num_vars, 0, &mut rng).unwrap();
    let secondary_param = P2::setup(1 << secondary_num_vars, 0, &mut rng).unwrap();

    let primary_circuit = RecursiveCircuit::new(true, primary_step_circuit, None);
    let mut primary_circuit =
        Halo2Circuit::new::<HyperPlonk<P1>>(primary_num_vars, primary_circuit);

    let (_, primary_vp) = {
        let primary_circuit_info = primary_circuit.circuit_info_without_preprocess().unwrap();
        Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_circuit_info).unwrap()
    };

    let secondary_circuit = RecursiveCircuit::new(
        false,
        secondary_step_circuit,
        Some(ProtostarAccumulationVerifierParam::from(&primary_vp)),
    );
    let mut secondary_circuit =
        Halo2Circuit::new::<HyperPlonk<P2>>(secondary_num_vars, secondary_circuit);
    let (secondary_pp, secondary_vp) = {
        let secondary_circuit_info = secondary_circuit.circuit_info().unwrap();
        Protostar::<HyperPlonk<P2>>::preprocess(&secondary_param, &secondary_circuit_info).unwrap()
    };

    primary_circuit.update_witness(|circuit| {
        circuit.avp = ProtostarAccumulationVerifierParam::from(&secondary_vp)
    });
    let primary_circuit_info = primary_circuit.circuit_info().unwrap();
    let (primary_pp, primary_vp) =
        Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_circuit_info).unwrap();

    let vp_digest = fe_truncated_from_le_bytes(
        Keccak256::digest(bincode::serialize(&(&primary_vp, &secondary_vp)).unwrap()),
        S1::HashChip::NUM_HASH_BITS,
    );
    primary_circuit.update_witness(|circuit| circuit.init(vp_digest));
    secondary_circuit.update_witness(|circuit| circuit.init(fe_to_fe(vp_digest)));

    let ivc_pp = ProtostarIvcProverParam {
        primary_pp,
        primary_atp,
        secondary_pp,
        secondary_atp,
        _marker: PhantomData,
    };
    let ivc_vp = {
        ProtostarIvcVerifierParam {
            vp_digest,
            primary_vp,
            primary_avp: primary_circuit.circuit().avp.clone(),
            primary_hp: primary_circuit.circuit().hash_config.borrow().clone(),
            primary_arity: S1::arity(),
            secondary_vp,
            secondary_hp: secondary_circuit.circuit().hash_config.borrow().clone(),
            secondary_arity: S2::arity(),
            _marker: PhantomData,
        }
    };

    Ok((primary_circuit, secondary_circuit, ivc_pp, ivc_vp))
}

#[allow(clippy::type_complexity)]
pub fn prove_steps<C, P1, P2, S1, S2, AT1, AT2>(
    ivc_pp: &ProtostarIvcProverParam<C, P1, P2, AT1, AT2>,
    primary_circuit: &mut Halo2Circuit<C::Scalar, RecursiveCircuit<C, S1>>,
    secondary_circuit: &mut Halo2Circuit<C::Base, RecursiveCircuit<C::Secondary, S2>>,
    num_steps: usize,
    mut rng: impl RngCore,
) -> Result<
    (
        ProtostarAccumulator<C::Scalar, P1>,
        ProtostarAccumulator<C::Base, P2>,
        Vec<C::Base>,
    ),
    crate::Error,
>
where
    C: TwoChainCurve,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::ScalarExt,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C> + From<C>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    S1: StepCircuit<C>,
    S2: StepCircuit<C::Secondary>,
    AT1: TranscriptRead<P1::CommitmentChunk, C::Scalar>
        + TranscriptWrite<P1::CommitmentChunk, C::Scalar>
        + InMemoryTranscript,
    AT2: TranscriptRead<P2::CommitmentChunk, C::Base>
        + TranscriptWrite<P2::CommitmentChunk, C::Base>
        + InMemoryTranscript,
{
    let mut primary_acc = Protostar::<HyperPlonk<P1>>::init_accumulator(&ivc_pp.primary_pp)?;
    let mut secondary_acc = Protostar::<HyperPlonk<P2>>::init_accumulator(&ivc_pp.secondary_pp)?;

    for step_idx in 0..num_steps {
        let primary_acc_x = primary_acc.instance.clone();

        let timer = start_timer(|| {
            format!(
                "prove_accumulation_from_nark-primary-{}",
                ivc_pp.primary_pp.pp.num_vars
            )
        });
        let proof = {
            let mut transcript = AT1::new(ivc_pp.primary_atp.clone());
            Protostar::<HyperPlonk<P1>>::prove_accumulation_from_nark(
                &ivc_pp.primary_pp,
                &mut primary_acc,
                primary_circuit as &_,
                &mut transcript,
                &mut rng,
            )?;
            transcript.into_proof()
        };
        end_timer(timer);

        secondary_circuit.update_witness(|circuit| {
            circuit.update(
                primary_acc_x,
                primary_acc.instance.clone(),
                primary_circuit.instances()[0].clone().try_into().unwrap(),
                proof,
            );
        });

        if step_idx != num_steps - 1 {
            let secondary_acc_x = secondary_acc.instance.clone();

            let timer = start_timer(|| {
                format!(
                    "prove_accumulation_from_nark-secondary-{}",
                    ivc_pp.secondary_pp.pp.num_vars
                )
            });
            let proof = {
                let mut transcript = AT2::new(ivc_pp.secondary_atp.clone());
                Protostar::<HyperPlonk<P2>>::prove_accumulation_from_nark(
                    &ivc_pp.secondary_pp,
                    &mut secondary_acc,
                    secondary_circuit as &_,
                    &mut transcript,
                    &mut rng,
                )?;
                transcript.into_proof()
            };
            end_timer(timer);

            primary_circuit.update_witness(|circuit| {
                circuit.update(
                    secondary_acc_x,
                    secondary_acc.instance.clone(),
                    secondary_circuit.instances()[0].clone().try_into().unwrap(),
                    proof,
                );
            });
        } else {
            return Ok((
                primary_acc,
                secondary_acc,
                secondary_circuit.instances()[0].to_vec(),
            ));
        }
    }

    unreachable!()
}

pub fn prove_decider<C, P1, P2, AT1, AT2>(
    ivc_pp: &ProtostarIvcProverParam<C, P1, P2, AT1, AT2>,
    primary_acc: &ProtostarAccumulator<C::Scalar, P1>,
    primary_transcript: &mut impl TranscriptWrite<P1::CommitmentChunk, C::Scalar>,
    secondary_acc: &mut ProtostarAccumulator<C::Base, P2>,
    secondary_circuit: &impl PlonkishCircuit<C::Base>,
    secondary_transcript: &mut impl TranscriptWrite<P2::CommitmentChunk, C::Base>,
    mut rng: impl RngCore,
) -> Result<(), crate::Error>
where
    C: TwoChainCurve,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::ScalarExt,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C> + From<C>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    AT1: InMemoryTranscript,
    AT2: InMemoryTranscript,
{
    let timer = start_timer(|| format!("prove_decider-primary-{}", ivc_pp.primary_pp.pp.num_vars));
    Protostar::<HyperPlonk<P1>>::prove_decider(
        &ivc_pp.primary_pp,
        primary_acc,
        primary_transcript,
        &mut rng,
    )?;
    end_timer(timer);
    let timer = start_timer(|| {
        format!(
            "prove_decider_with_last_nark-secondary-{}",
            ivc_pp.secondary_pp.pp.num_vars
        )
    });
    Protostar::<HyperPlonk<P2>>::prove_decider_with_last_nark(
        &ivc_pp.secondary_pp,
        secondary_acc,
        secondary_circuit,
        secondary_transcript,
        &mut rng,
    )?;
    end_timer(timer);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn verify_decider<C, P1, P2, H1, H2>(
    ivc_vp: &ProtostarIvcVerifierParam<C, P1, P2, H1::Param, H2::Param>,
    primary_initial_input: &[C::Scalar],
    primary_output: &[C::Scalar],
    primary_acc: ProtostarAccumulatorInstance<C::Scalar, P1::Commitment>,
    primary_transcript: &mut impl TranscriptRead<P1::CommitmentChunk, C::Scalar>,
    secondary_initial_input: &[C::Base],
    secondary_output: &[C::Base],
    mut secondary_acc_before_last: ProtostarAccumulatorInstance<C::Base, P2::Commitment>,
    secondary_last_instances: &[Vec<C::Base>],
    secondary_transcript: &mut impl TranscriptRead<P2::CommitmentChunk, C::Base>,
    num_steps: usize,
    mut rng: impl RngCore,
) -> Result<(), crate::Error>
where
    C: TwoChainCurve,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::ScalarExt,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C> + From<C>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    H1: HashInstruction<C>,
    H2: HashInstruction<C::Secondary>,
{
    if H1::hash_state(
        &ivc_vp.primary_hp,
        ivc_vp.vp_digest,
        num_steps,
        primary_initial_input,
        primary_output,
        &secondary_acc_before_last,
    ) != fe_to_fe(secondary_last_instances[0][0])
    {
        return Err(crate::Error::InvalidSnark(
            "Invalid primary state hash".to_string(),
        ));
    }
    if H2::hash_state(
        &ivc_vp.secondary_hp,
        fe_to_fe(ivc_vp.vp_digest),
        num_steps,
        secondary_initial_input,
        secondary_output,
        &primary_acc,
    ) != secondary_last_instances[0][1]
    {
        return Err(crate::Error::InvalidSnark(
            "Invalid secondary state hash".to_string(),
        ));
    }

    Protostar::<HyperPlonk<P1>>::verify_decider(
        &ivc_vp.primary_vp,
        &primary_acc,
        primary_transcript,
        &mut rng,
    )?;
    Protostar::<HyperPlonk<P2>>::verify_decider_with_last_nark(
        &ivc_vp.secondary_vp,
        &mut secondary_acc_before_last,
        secondary_last_instances,
        secondary_transcript,
        &mut rng,
    )?;
    Ok(())
}

#[derive(Debug)]
pub struct ProtostarIvcAggregator<C, P1, P2, TccChip, HashChip, HP2>
where
    C: TwoChainCurve,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
    HashChip: HashInstruction<C>,
{
    ivc_vp: ProtostarIvcVerifierParam<C, P1, P2, HashChip::Param, HP2>,
    tcc_chip: TccChip,
    hash_chip: HashChip,
    _marker: PhantomData<(C, P1, P2)>,
}

impl<C, P1, P2, TccChip, HashChip, HP2> ProtostarIvcAggregator<C, P1, P2, TccChip, HashChip, HP2>
where
    C: TwoChainCurve,
    C::Base: Serialize,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
    TccChip: TwoChainCurveInstruction<C>,
    HashChip: HashInstruction<C, TccChip = TccChip>,
{
    fn assign_accumulator_primary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        acc: Value<&ProtostarAccumulatorInstance<C::Scalar, C>>,
    ) -> Result<
        AssignedProtostarAccumulatorInstance<TccChip::Assigned, TccChip::AssignedPrimary>,
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let avp = &self.ivc_vp.primary_avp;

        let instances = avp
            .num_instances
            .iter()
            .zip(
                acc.map(|acc| &acc.instances)
                    .transpose_vec(avp.num_instances.len()),
            )
            .map(|(num_instances, instances)| {
                instances
                    .transpose_vec(*num_instances)
                    .into_iter()
                    .map(|instance| tcc_chip.assign_witness(layouter, instance.copied()))
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = acc
            .map(|acc| &acc.witness_comms)
            .transpose_vec(avp.num_folding_witness_polys())
            .into_iter()
            .map(|witness_comm| tcc_chip.assign_witness_primary(layouter, witness_comm.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let challenges = acc
            .map(|acc| &acc.challenges)
            .transpose_vec(avp.num_folding_challenges())
            .into_iter()
            .map(|challenge| tcc_chip.assign_witness(layouter, challenge.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let u = tcc_chip.assign_witness(layouter, acc.map(|acc| &acc.u).copied())?;
        let e_comm = tcc_chip.assign_witness_primary(layouter, acc.map(|acc| acc.e_comm))?;
        let compressed_e_sum = match avp.strategy {
            NoCompressing => None,
            Compressing => Some(
                tcc_chip.assign_witness(layouter, acc.map(|acc| acc.compressed_e_sum.unwrap()))?,
            ),
        };

        Ok(ProtostarAccumulatorInstance {
            instances,
            witness_comms,
            challenges,
            u,
            e_comm,
            compressed_e_sum,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_last_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        primary_initial_input: Value<Vec<C::Scalar>>,
        primary_output: Value<Vec<C::Scalar>>,
        primary_acc: Value<ProtostarAccumulatorInstance<C::Scalar, C>>,
        secondary_initial_input: Value<Vec<C::Base>>,
        secondary_output: Value<Vec<C::Base>>,
        secondary_acc_before_last: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
        secondary_last_instance: Value<[C::Base; 2]>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
        num_steps: Value<usize>,
    ) -> Result<
        (
            AssignedProtostarAccumulatorInstance<TccChip::Assigned, TccChip::AssignedPrimary>,
            AssignedProtostarAccumulatorInstance<TccChip::AssignedBase, TccChip::AssignedSecondary>,
        ),
        Error,
    > {
        let primary_avp = &self.ivc_vp.primary_avp;
        let tcc_chip = &self.tcc_chip;
        let hash_chip = &self.hash_chip;

        let acc_verifier =
            ProtostarAccumulationVerifier::new(primary_avp.clone(), tcc_chip.clone());

        let vp_digest = tcc_chip.assign_constant(layouter, primary_avp.vp_digest.unwrap())?;
        let num_steps = tcc_chip.assign_witness(
            layouter,
            num_steps.map(|num_steps| C::Scalar::from(num_steps as u64)),
        )?;

        let primary_acc = self.assign_accumulator_primary(layouter, primary_acc.as_ref())?;
        let secondary_acc_before_last =
            acc_verifier.assign_accumulator(layouter, secondary_acc_before_last.as_ref())?;
        let (secondary_last_nark, _, secondary_acc) = {
            let instances = secondary_last_instance
                .as_ref()
                .map(|instances| [&instances[0], &instances[1]])
                .transpose_array();
            acc_verifier.verify_accumulation_from_nark(
                layouter,
                &secondary_acc_before_last,
                instances,
                transcript,
            )?
        };

        let primary_h = {
            let initial_input = primary_initial_input
                .transpose_vec(self.ivc_vp.primary_arity)
                .into_iter()
                .map(|input| tcc_chip.assign_witness(layouter, input))
                .try_collect::<_, Vec<_>, _>()?;
            let output = primary_output
                .transpose_vec(self.ivc_vp.primary_arity)
                .into_iter()
                .map(|input| tcc_chip.assign_witness(layouter, input))
                .try_collect::<_, Vec<_>, _>()?;
            hash_chip.hash_assigned_state(
                layouter,
                &vp_digest,
                &num_steps,
                &initial_input,
                &output,
                &secondary_acc_before_last,
            )?
        };
        let secondary_h = {
            // TODO: Verify another Hyrax HyperPlonk that proves the stat hash
            let _initial_input = secondary_initial_input
                .transpose_vec(self.ivc_vp.secondary_arity)
                .into_iter()
                .map(|input| tcc_chip.assign_witness_base(layouter, input))
                .try_collect::<_, Vec<_>, _>()?;
            let _output = secondary_output
                .transpose_vec(self.ivc_vp.secondary_arity)
                .into_iter()
                .map(|input| tcc_chip.assign_witness_base(layouter, input))
                .try_collect::<_, Vec<_>, _>()?;
            let secondary_h = tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?;
            tcc_chip.fit_base_in_scalar(layouter, &secondary_h)?
        };

        let primary_h_from_last_nark =
            tcc_chip.fit_base_in_scalar(layouter, &secondary_last_nark.instances[0][0])?;
        let secondary_h_from_last_nark =
            tcc_chip.fit_base_in_scalar(layouter, &secondary_last_nark.instances[0][1])?;
        tcc_chip.constrain_equal(layouter, &primary_h, &primary_h_from_last_nark)?;
        tcc_chip.constrain_equal(layouter, &secondary_h, &secondary_h_from_last_nark)?;

        Ok((primary_acc, secondary_acc))
    }
}
