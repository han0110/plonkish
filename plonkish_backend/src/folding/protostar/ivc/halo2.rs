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
        arithmetic::{fe_to_fe, CurveAffine, CurveCycle, Field, PrimeField},
        end_timer, izip_eq, start_timer,
        transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use rand::RngCore;
use std::{borrow::Cow, fmt::Debug, hash::Hash, iter, marker::PhantomData};

type AssignedPlonkishNarkInstance<C, EccChip, ScalarChip> = PlonkishNarkInstance<
    <ScalarChip as FieldInstruction<
        <C as CurveAffine>::ScalarExt,
        <C as CurveAffine>::Base,
    >>::Assigned,
    <EccChip as NativeEccInstruction<C>>::Assigned,
>;

type AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip> = ProtostarAccumulatorInstance<
    <ScalarChip as FieldInstruction<
        <C as CurveAffine>::ScalarExt,
        <C as CurveAffine>::Base,
    >>::Assigned,
    <EccChip as NativeEccInstruction<C>>::Assigned,
>;

pub trait NativeEccInstruction<C: CurveAffine>: Clone + Debug {
    type AssignedCell: Clone + Debug;
    type Assigned: Clone + Debug + AsRef<[Self::AssignedCell]>;

    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        constant: C,
    ) -> Result<Self::Assigned, Error>;

    fn assign_witness(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        witness: Value<C>,
    ) -> Result<Self::Assigned, Error>;

    fn select(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        condition: &Self::AssignedCell,
        when_true: &Self::Assigned,
        when_false: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C) -> bool);

    fn add(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn mul(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        base: &Self::Assigned,
        le_bits: &[Self::AssignedCell],
    ) -> Result<Self::Assigned, Error>;
}

pub trait NativeFieldInstruction<F: PrimeField>: Clone + Debug {
    type AssignedCell: Clone + Debug;

    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<F>,
        constant: F,
    ) -> Result<Self::AssignedCell, Error>;

    fn assign_witness(
        &self,
        layouter: &mut impl Layouter<F>,
        witness: Value<F>,
    ) -> Result<Self::AssignedCell, Error>;

    fn is_equal(
        &self,
        layouter: &mut impl Layouter<F>,
        lhs: &Self::AssignedCell,
        rhs: &Self::AssignedCell,
    ) -> Result<Self::AssignedCell, Error>;

    fn select(
        &self,
        layouter: &mut impl Layouter<F>,
        condition: &Self::AssignedCell,
        when_true: &Self::AssignedCell,
        when_false: &Self::AssignedCell,
    ) -> Result<Self::AssignedCell, Error>;

    fn assert_if_known(&self, value: &Self::AssignedCell, f: impl FnOnce(&F) -> bool);

    fn add(
        &self,
        layouter: &mut impl Layouter<F>,
        lhs: &Self::AssignedCell,
        rhs: &Self::AssignedCell,
    ) -> Result<Self::AssignedCell, Error>;
}

pub trait FieldInstruction<F: PrimeField, N: PrimeField>: Clone + Debug {
    type AssignedCell: Clone + Debug;
    type Assigned: Clone + Debug + AsRef<[Self::AssignedCell]>;

    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<N>,
        constant: F,
    ) -> Result<Self::Assigned, Error>;

    fn assign_witness(
        &self,
        layouter: &mut impl Layouter<N>,
        witness: Value<F>,
    ) -> Result<Self::Assigned, Error>;

    fn fit_in_native(
        &self,
        layouter: &mut impl Layouter<N>,
        value: &Self::Assigned,
    ) -> Result<Self::AssignedCell, Error>;

    fn select(
        &self,
        layouter: &mut impl Layouter<N>,
        condition: &Self::AssignedCell,
        when_true: &Self::Assigned,
        when_false: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&F) -> bool);

    fn add(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn mul(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
    ) -> Result<Self::Assigned, Error>;

    fn inner_product<'a, 'b>(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: impl IntoIterator<Item = &'a Self::Assigned>,
        rhs: impl IntoIterator<Item = &'b Self::Assigned>,
    ) -> Result<Self::Assigned, Error>
    where
        Self::Assigned: 'a + 'b,
    {
        let products = izip_eq!(lhs, rhs)
            .map(|(lhs, rhs)| self.mul(layouter, lhs, rhs))
            .collect_vec();
        products
            .into_iter()
            .reduce(|acc, output| self.add(layouter, &acc?, &output?))
            .unwrap()
    }

    fn powers(
        &self,
        layouter: &mut impl Layouter<N>,
        base: &Self::Assigned,
        n: usize,
    ) -> Result<Vec<Self::Assigned>, Error> {
        Ok(match n {
            0 => Vec::new(),
            1 => vec![self.assign_constant(layouter, F::ONE)?],
            2 => vec![self.assign_constant(layouter, F::ONE)?, base.clone()],
            _ => {
                let mut powers = Vec::with_capacity(n);
                powers.push(self.assign_constant(layouter, F::ONE)?);
                powers.push(base.clone());
                for _ in 0..n - 2 {
                    powers.push(self.mul(layouter, powers.last().unwrap(), base)?);
                }
                powers
            }
        })
    }
}

pub trait UtilInstruction<N: PrimeField>: Clone + Debug {
    type AssignedCell;

    fn convert(
        &self,
        layouter: &mut impl Layouter<N>,
        value: &AssignedCell<N, N>,
    ) -> Result<Self::AssignedCell, Error>;

    fn constrain_equal(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &Self::AssignedCell,
        rhs: &Self::AssignedCell,
    ) -> Result<(), Error>;

    fn constrain_instance(
        &self,
        layouter: &mut impl Layouter<N>,
        value: &Self::AssignedCell,
        row: usize,
    ) -> Result<(), Error>;
}

pub trait HashInstruction<C: CurveAffine>: Clone + Debug {
    type Param: Clone + Debug;
    type AssignedCell: Clone + Debug;

    fn hash_state<Comm: AsRef<C>>(
        param: &Self::Param,
        vp_digest: C::Base,
        step_idx: usize,
        initial_input: &[C::Base],
        output: &[C::Base],
        acc: &ProtostarAccumulatorInstance<C::Scalar, Comm>,
    ) -> C::Base;

    fn hash_assigned_state<EccChip, ScalarChip>(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        vp_digest: &Self::AssignedCell,
        step_idx: &Self::AssignedCell,
        initial_input: &[Self::AssignedCell],
        output: &[Self::AssignedCell],
        acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
    ) -> Result<Self::AssignedCell, Error>
    where
        EccChip: NativeEccInstruction<C, AssignedCell = Self::AssignedCell>,
        ScalarChip: FieldInstruction<C::Scalar, C::Base, AssignedCell = Self::AssignedCell>;
}

pub trait TranscriptInstruction<C: CurveAffine, EccChip, ScalarChip>: Clone + Debug
where
    EccChip: NativeEccInstruction<C>,
    ScalarChip: FieldInstruction<C::Scalar, C::Base, AssignedCell = EccChip::AssignedCell>,
{
    type Challenge: Clone + Debug + AsRef<ScalarChip::Assigned>;

    fn dummy_proof(avp: &ProtostarAccumulationVerifierParam<C::Base>) -> Vec<u8> {
        let uncompressed_comm_size = C::Base::ZERO.to_repr().as_ref().len() * 2;
        let scalar_size = C::Scalar::ZERO.to_repr().as_ref().len();
        let proof_size = avp.num_folding_witness_polys() * uncompressed_comm_size
            + match avp.strategy {
                NoCompressing => avp.num_cross_terms * uncompressed_comm_size,
                Compressing => uncompressed_comm_size + avp.num_cross_terms * scalar_size,
            };
        vec![0; proof_size]
    }

    #[allow(clippy::type_complexity)]
    fn challenge_to_le_bits(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        scalar: &Self::Challenge,
    ) -> Result<Vec<EccChip::AssignedCell>, Error>;

    fn squeeze_challenge(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
    ) -> Result<Self::Challenge, Error>;

    fn squeeze_challenges(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
        n: usize,
    ) -> Result<Vec<Self::Challenge>, Error> {
        (0..n).map(|_| self.squeeze_challenge(layouter)).collect()
    }

    fn common_field_element(&mut self, fe: &ScalarChip::Assigned) -> Result<(), Error>;

    fn common_field_elements(&mut self, fes: &[ScalarChip::Assigned]) -> Result<(), Error> {
        fes.iter().try_for_each(|fe| self.common_field_element(fe))
    }

    fn read_field_element(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
    ) -> Result<ScalarChip::Assigned, Error>;

    fn read_field_elements(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
        n: usize,
    ) -> Result<Vec<ScalarChip::Assigned>, Error> {
        (0..n).map(|_| self.read_field_element(layouter)).collect()
    }

    fn common_commitment(&mut self, comm: &EccChip::Assigned) -> Result<(), Error>;

    fn common_commitments(&mut self, comms: &[EccChip::Assigned]) -> Result<(), Error> {
        comms
            .iter()
            .try_for_each(|comm| self.common_commitment(comm))
    }

    fn read_commitment(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
    ) -> Result<EccChip::Assigned, Error>;

    fn read_commitments(
        &mut self,
        layouter: &mut impl Layouter<C::Base>,
        n: usize,
    ) -> Result<Vec<EccChip::Assigned>, Error> {
        (0..n).map(|_| self.read_commitment(layouter)).collect()
    }

    fn absorb_accumulator(
        &mut self,
        acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
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

pub trait Chips<C: CurveAffine>: Clone + Debug {
    type AssignedCell: Clone + Debug;

    type EccChip: NativeEccInstruction<C, AssignedCell = Self::AssignedCell>;
    type BaseChip: NativeFieldInstruction<C::Base, AssignedCell = Self::AssignedCell>;
    type ScalarChip: FieldInstruction<C::Scalar, C::Base, AssignedCell = Self::AssignedCell>;
    type UtilChip: UtilInstruction<C::Base, AssignedCell = Self::AssignedCell>;
    type HashChip: HashInstruction<C, AssignedCell = Self::AssignedCell>;
    type TranscriptChip<'a>: TranscriptInstruction<C, Self::EccChip, Self::ScalarChip>
    where
        Self: 'a;

    fn ecc_chip(&self) -> &Self::EccChip;

    fn base_chip(&self) -> &Self::BaseChip;

    fn scalar_chip(&self) -> &Self::ScalarChip;

    fn util_chip(&self) -> &Self::UtilChip;

    fn hash_chip(&self) -> &Self::HashChip;

    fn transcript_chip<'a>(&'a self, proof: Value<&'a [u8]>) -> Self::TranscriptChip<'a>;
}

pub trait StepCircuit<C: CurveAffine>: Clone + Debug + CircuitExt<C::Base> {
    type Chips: Chips<C>;

    fn chips(&self, config: Self::Config) -> Self::Chips;

    fn step_idx(&self) -> usize;

    fn initial_input(&self) -> &[C::Base];

    fn input(&self) -> &[C::Base];

    fn output(&self) -> &[C::Base];

    fn next(&mut self);

    #[allow(clippy::type_complexity)]
    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<C::Base>,
    ) -> Result<
        (
            Vec<AssignedCell<C::Base, C::Base>>,
            Vec<AssignedCell<C::Base, C::Base>>,
        ),
        Error,
    >;
}

pub struct ProtostarAccumulationVerifier<C, EccChip, ScalarChip>
where
    C: CurveAffine,
{
    ecc_chip: EccChip,
    scalar_chip: ScalarChip,
    avp: ProtostarAccumulationVerifierParam<C::Base>,
    _marker: PhantomData<C>,
}

impl<C, EccChip, ScalarChip> ProtostarAccumulationVerifier<C, EccChip, ScalarChip>
where
    C: CurveAffine,
    EccChip: NativeEccInstruction<C>,
    ScalarChip: FieldInstruction<C::Scalar, C::Base, AssignedCell = EccChip::AssignedCell>,
{
    pub fn new(
        ecc_chip: EccChip,
        scalar_chip: ScalarChip,
        avp: ProtostarAccumulationVerifierParam<C::Base>,
    ) -> Result<Self, Error> {
        Ok(Self {
            ecc_chip,
            scalar_chip,
            avp,
            _marker: PhantomData,
        })
    }

    pub fn assign_default_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Base>,
    ) -> Result<AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let ProtostarAccumulationVerifierParam { num_instances, .. } = &self.avp;

        let instances = num_instances
            .iter()
            .map(|num_instances| {
                iter::repeat_with(|| scalar_chip.assign_constant(layouter, C::Scalar::ZERO))
                    .take(*num_instances)
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = iter::repeat_with(|| ecc_chip.assign_constant(layouter, C::identity()))
            .take(self.avp.num_folding_witness_polys())
            .try_collect::<_, Vec<_>, _>()?;
        let challenges =
            iter::repeat_with(|| scalar_chip.assign_constant(layouter, C::Scalar::ZERO))
                .take(self.avp.num_folding_challenges())
                .try_collect::<_, Vec<_>, _>()?;
        let u = scalar_chip.assign_constant(layouter, C::Scalar::ZERO)?;
        let e_comm = ecc_chip.assign_constant(layouter, C::identity())?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(scalar_chip.assign_constant(layouter, C::Scalar::ZERO)?),
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
        layouter: &mut impl Layouter<C::Base>,
        acc: Value<&ProtostarAccumulatorInstance<C::Scalar, C>>,
    ) -> Result<AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let ProtostarAccumulationVerifierParam { num_instances, .. } = &self.avp;

        let instances = num_instances
            .iter()
            .zip(
                acc.map(|acc| &acc.instances)
                    .transpose_vec(num_instances.len()),
            )
            .map(|(num_instances, instances)| {
                instances
                    .transpose_vec(*num_instances)
                    .into_iter()
                    .map(|instance| scalar_chip.assign_witness(layouter, instance.copied()))
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = acc
            .map(|acc| &acc.witness_comms)
            .transpose_vec(self.avp.num_folding_witness_polys())
            .into_iter()
            .map(|witness_comm| ecc_chip.assign_witness(layouter, witness_comm.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let challenges = acc
            .map(|acc| &acc.challenges)
            .transpose_vec(self.avp.num_folding_challenges())
            .into_iter()
            .map(|challenge| scalar_chip.assign_witness(layouter, challenge.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let u = scalar_chip.assign_witness(layouter, acc.map(|acc| &acc.u).copied())?;
        let e_comm = ecc_chip.assign_witness(layouter, acc.map(|acc| acc.e_comm))?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(
                scalar_chip
                    .assign_witness(layouter, acc.map(|acc| acc.compressed_e_sum.unwrap()))?,
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
        layouter: &mut impl Layouter<C::Base>,
        r: &ScalarChip::Assigned,
        r_nark: AssignedPlonkishNarkInstance<C, EccChip, ScalarChip>,
    ) -> Result<AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let AssignedPlonkishNarkInstance::<C, EccChip, ScalarChip> {
            instances,
            challenges,
            witness_comms,
        } = r_nark;
        let u = r.clone();
        let e_comm = ecc_chip.assign_constant(layouter, C::identity())?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(scalar_chip.assign_constant(layouter, C::Scalar::ZERO)?),
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
        layouter: &mut impl Layouter<C::Base>,
        acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        instances: [Value<&C::Scalar>; 2],
        transcript: &mut impl TranscriptInstruction<C, EccChip, ScalarChip>,
    ) -> Result<
        (
            AssignedPlonkishNarkInstance<C, EccChip, ScalarChip>,
            AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
            AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        ),
        Error,
    > {
        let Self { scalar_chip, .. } = self;
        let ProtostarAccumulationVerifierParam {
            strategy,
            num_witness_polys,
            num_challenges,
            num_cross_terms,
            ..
        } = &self.avp;

        let instances = instances
            .into_iter()
            .map(|instance| scalar_chip.assign_witness(layouter, instance.copied()))
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
                    scalar_chip.powers(layouter, challenge.as_ref(), *num_powers + 1)?;
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
        layouter: &mut impl Layouter<C::Base>,
        acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        nark: &AssignedPlonkishNarkInstance<C, EccChip, ScalarChip>,
        cross_term_comms: &[EccChip::Assigned],
        compressed_cross_term_sums: Option<&[ScalarChip::Assigned]>,
        r: &ScalarChip::Assigned,
        r_le_bits: &[EccChip::AssignedCell],
    ) -> Result<
        (
            AssignedPlonkishNarkInstance<C, EccChip, ScalarChip>,
            AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        ),
        Error,
    > {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let ProtostarAccumulationVerifierParam {
            strategy,
            num_cross_terms,
            ..
        } = self.avp;

        let powers_of_r = scalar_chip.powers(layouter, r, num_cross_terms + 1)?;

        let r_nark = {
            let instances = nark
                .instances
                .iter()
                .map(|instances| {
                    instances
                        .iter()
                        .map(|instance| scalar_chip.mul(layouter, r, instance))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = nark
                .witness_comms
                .iter()
                .map(|comm| ecc_chip.mul(layouter, comm, r_le_bits))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = nark
                .challenges
                .iter()
                .map(|challenge| scalar_chip.mul(layouter, r, challenge))
                .try_collect::<_, Vec<_>, _>()?;
            AssignedPlonkishNarkInstance::<C, EccChip, ScalarChip> {
                instances,
                challenges,
                witness_comms,
            }
        };

        let acc_prime = {
            let instances = izip_eq!(&acc.instances, &r_nark.instances)
                .map(|(lhs, rhs)| {
                    izip_eq!(lhs, rhs)
                        .map(|(lhs, rhs)| scalar_chip.add(layouter, lhs, rhs))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = izip_eq!(&acc.witness_comms, &r_nark.witness_comms)
                .map(|(lhs, rhs)| ecc_chip.add(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = izip_eq!(&acc.challenges, &r_nark.challenges)
                .map(|(lhs, rhs)| scalar_chip.add(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let u = scalar_chip.add(layouter, &acc.u, r)?;
            let e_comm = if cross_term_comms.is_empty() {
                acc.e_comm.clone()
            } else {
                let mut e_comm = cross_term_comms.last().unwrap().clone();
                for item in cross_term_comms.iter().rev().skip(1).chain([&acc.e_comm]) {
                    e_comm = ecc_chip.mul(layouter, &e_comm, r_le_bits)?;
                    e_comm = ecc_chip.add(layouter, &e_comm, item)?;
                }
                e_comm
            };
            let compressed_e_sum = match strategy {
                NoCompressing => None,
                Compressing => {
                    let rhs = scalar_chip.inner_product(
                        layouter,
                        &powers_of_r[1..],
                        compressed_cross_term_sums.unwrap(),
                    )?;
                    Some(scalar_chip.add(layouter, acc.compressed_e_sum.as_ref().unwrap(), &rhs)?)
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
        layouter: &mut impl Layouter<C::Base>,
        condition: &EccChip::AssignedCell,
        when_true: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        when_false: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
    ) -> Result<AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;

        let instances = izip_eq!(&when_true.instances, &when_false.instances)
            .map(|(when_true, when_false)| {
                izip_eq!(when_true, when_false)
                    .map(|(when_true, when_false)| {
                        scalar_chip.select(layouter, condition, when_true, when_false)
                    })
                    .try_collect()
            })
            .try_collect()?;
        let witness_comms = izip_eq!(&when_true.witness_comms, &when_false.witness_comms)
            .map(|(when_true, when_false)| {
                ecc_chip.select(layouter, condition, when_true, when_false)
            })
            .try_collect()?;
        let challenges = izip_eq!(&when_true.challenges, &when_false.challenges)
            .map(|(when_true, when_false)| {
                scalar_chip.select(layouter, condition, when_true, when_false)
            })
            .try_collect()?;
        let u = scalar_chip.select(layouter, condition, &when_true.u, &when_false.u)?;
        let e_comm = ecc_chip.select(layouter, condition, &when_true.e_comm, &when_false.e_comm)?;
        let compressed_e_sum = match self.avp.strategy {
            NoCompressing => None,
            Compressing => Some(scalar_chip.select(
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
    C: CurveAffine,
    Sc: StepCircuit<C>,
{
    is_primary: bool,
    step_circuit: Sc,
    config: Sc::Config,
    chips: Sc::Chips,
    hp: <<Sc::Chips as Chips<C>>::HashChip as HashInstruction<C>>::Param,
    avp: ProtostarAccumulationVerifierParam<C::Base>,
    h_prime: Value<C::Base>,
    acc: Value<ProtostarAccumulatorInstance<C::Scalar, C>>,
    acc_prime: Value<ProtostarAccumulatorInstance<C::Scalar, C>>,
    incoming_instances: [Value<C::Scalar>; 2],
    incoming_proof: Value<Vec<u8>>,
}

impl<C, Sc> RecursiveCircuit<C, Sc>
where
    C: CurveAffine,
    Sc: StepCircuit<C>,
{
    pub const DUMMY_H: C::Base = C::Base::ZERO;

    pub fn new(
        is_primary: bool,
        step_circuit: Sc,
        hp: <<Sc::Chips as Chips<C>>::HashChip as HashInstruction<C>>::Param,
        avp: Option<ProtostarAccumulationVerifierParam<C::Base>>,
    ) -> Self {
        let config = Self::configure(&mut Default::default());
        let chips = step_circuit.chips(config.clone());
        let mut circuit = Self {
            is_primary,
            step_circuit,
            config,
            chips,
            hp,
            avp: Default::default(),
            h_prime: Value::unknown(),
            acc: Value::unknown(),
            acc_prime: Value::unknown(),
            incoming_instances: [Value::unknown(); 2],
            incoming_proof: Value::unknown(),
        };
        if let Some(avp) = avp {
            circuit.init(avp);
        }
        circuit
    }

    pub fn update<Comm: AsRef<C>>(
        &mut self,
        acc: ProtostarAccumulatorInstance<C::Scalar, Comm>,
        acc_prime: ProtostarAccumulatorInstance<C::Scalar, Comm>,
        incoming_instances: [C::Scalar; 2],
        incoming_proof: Vec<u8>,
    ) {
        if (self.is_primary && acc_prime.u != C::Scalar::ZERO)
            || (!self.is_primary && acc.u != C::Scalar::ZERO)
        {
            self.step_circuit.next();
        }
        self.h_prime = Value::known(<Sc::Chips as Chips<_>>::HashChip::hash_state(
            &self.hp,
            self.avp.vp_digest,
            self.step_circuit.step_idx() + 1,
            self.step_circuit.initial_input(),
            self.step_circuit.output(),
            &acc_prime,
        ));
        let convert =
            |acc: ProtostarAccumulatorInstance<C::Scalar, Comm>| ProtostarAccumulatorInstance {
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

    fn init(&mut self, avp: ProtostarAccumulationVerifierParam<C::Base>) {
        assert_eq!(&avp.num_instances, &[2]);
        self.avp = avp;
        self.update::<Cow<C>>(
            self.avp.init_accumulator(),
            self.avp.init_accumulator(),
            [Self::DUMMY_H; 2].map(fe_to_fe),
            <Sc::Chips as Chips<_>>::TranscriptChip::dummy_proof(&self.avp),
        );
    }

    fn check_initial_condition(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        chips: &Sc::Chips,
        is_base_case: &<Sc::Chips as Chips<C>>::AssignedCell,
        initial_input: &[<Sc::Chips as Chips<C>>::AssignedCell],
        input: &[<Sc::Chips as Chips<C>>::AssignedCell],
    ) -> Result<(), Error> {
        let base_chip = chips.base_chip();
        let util_chip = chips.util_chip();
        let zero = base_chip.assign_constant(layouter, C::Base::ZERO)?;

        for (lhs, rhs) in input.iter().zip(initial_input.iter()) {
            let lhs = base_chip.select(layouter, is_base_case, lhs, &zero)?;
            let rhs = base_chip.select(layouter, is_base_case, rhs, &zero)?;
            util_chip.constrain_equal(layouter, &lhs, &rhs)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn check_state_hash(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        chips: &Sc::Chips,
        is_base_case: Option<&<Sc::Chips as Chips<C>>::AssignedCell>,
        h: &<Sc::Chips as Chips<C>>::AssignedCell,
        vp_digest: &<Sc::Chips as Chips<C>>::AssignedCell,
        step_idx: &<Sc::Chips as Chips<C>>::AssignedCell,
        initial_input: &[<Sc::Chips as Chips<C>>::AssignedCell],
        output: &[<Sc::Chips as Chips<C>>::AssignedCell],
        acc: &AssignedProtostarAccumulatorInstance<
            C,
            <Sc::Chips as Chips<C>>::EccChip,
            <Sc::Chips as Chips<C>>::ScalarChip,
        >,
    ) -> Result<(), Error> {
        let base_chip = chips.base_chip();
        let hash_chip = chips.hash_chip();
        let util_chip = chips.util_chip();
        let lhs = h;
        let rhs = hash_chip.hash_assigned_state::<<Sc::Chips as Chips<_>>::EccChip, <Sc::Chips as Chips<_>>::ScalarChip>(
            layouter,
            vp_digest,
            step_idx,
            initial_input,
            output,
            acc,
        )?;
        let rhs = if let Some(is_base_case) = is_base_case {
            let dummy_h = base_chip.assign_constant(layouter, Self::DUMMY_H)?;
            base_chip.select(layouter, is_base_case, &dummy_h, &rhs)?
        } else {
            rhs
        };
        util_chip.constrain_equal(layouter, lhs, &rhs)?;
        Ok(())
    }

    fn synthesize_accumulation_verifier(
        &self,
        mut layouter: impl Layouter<C::Base>,
        input: &[AssignedCell<C::Base, C::Base>],
        output: &[AssignedCell<C::Base, C::Base>],
    ) -> Result<(), Error> {
        let layouter = &mut layouter;

        let ecc_chip = self.chips.ecc_chip();
        let base_chip = self.chips.base_chip();
        let scalar_chip = self.chips.scalar_chip();
        let util_chip = self.chips.util_chip();

        let verifier = ProtostarAccumulationVerifier::new(
            ecc_chip.clone(),
            scalar_chip.clone(),
            self.avp.clone(),
        )?;

        let zero = base_chip.assign_constant(layouter, C::Base::ZERO)?;
        let one = base_chip.assign_constant(layouter, C::Base::ONE)?;
        let vp_digest = base_chip.assign_witness(layouter, Value::known(self.avp.vp_digest))?;
        let step_idx = base_chip.assign_witness(
            layouter,
            Value::known(C::Base::from(self.step_circuit.step_idx() as u64)),
        )?;
        let step_idx_plus_one = base_chip.add(layouter, &step_idx, &one)?;
        let initial_input = self
            .step_circuit
            .initial_input()
            .iter()
            .map(|value| base_chip.assign_witness(layouter, Value::known(*value)))
            .try_collect::<_, Vec<_>, _>()?;
        let input = input
            .iter()
            .map(|assigned| util_chip.convert(layouter, assigned))
            .try_collect::<_, Vec<_>, _>()?;
        let output = output
            .iter()
            .map(|assigned| util_chip.convert(layouter, assigned))
            .try_collect::<_, Vec<_>, _>()?;

        let is_base_case = base_chip.is_equal(layouter, &step_idx, &zero)?;
        let h_prime = base_chip.assign_witness(layouter, self.h_prime)?;

        self.check_initial_condition(layouter, &self.chips, &is_base_case, &initial_input, &input)?;

        let acc = verifier.assign_accumulator(layouter, self.acc.as_ref())?;

        let (nark, acc_r_nark, acc_prime) = {
            let instances =
                [&self.incoming_instances[0], &self.incoming_instances[1]].map(Value::as_ref);
            let proof = self.incoming_proof.as_ref().map(Vec::as_slice);
            let mut transcript = self.chips.transcript_chip(proof);
            verifier.verify_accumulation_from_nark(layouter, &acc, instances, &mut transcript)?
        };

        let acc_prime = {
            let acc_default = if self.is_primary {
                verifier.assign_default_accumulator(layouter)?
            } else {
                acc_r_nark
            };
            verifier.select_accumulator(layouter, &is_base_case, &acc_default, &acc_prime)?
        };

        let h_from_incoming = scalar_chip.fit_in_native(layouter, &nark.instances[0][0])?;
        let h_ohs_from_incoming = scalar_chip.fit_in_native(layouter, &nark.instances[0][1])?;

        self.check_state_hash(
            layouter,
            &self.chips,
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
            &self.chips,
            None,
            &h_prime,
            &vp_digest,
            &step_idx_plus_one,
            &initial_input,
            &output,
            &acc_prime,
        )?;

        util_chip.constrain_instance(layouter, &h_ohs_from_incoming, 0)?;
        util_chip.constrain_instance(layouter, &h_prime, 1)?;

        Ok(())
    }
}

impl<C, Sc> Circuit<C::Base> for RecursiveCircuit<C, Sc>
where
    C: CurveAffine,
    Sc: StepCircuit<C>,
{
    type Config = Sc::Config;
    type FloorPlanner = Sc::FloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            is_primary: self.is_primary,
            step_circuit: self.step_circuit.without_witnesses(),
            config: self.config.clone(),
            chips: self.chips.clone(),
            avp: self.avp.clone(),
            hp: self.hp.clone(),
            h_prime: Value::unknown(),
            acc: Value::unknown(),
            acc_prime: Value::unknown(),
            incoming_instances: [Value::unknown(), Value::unknown()],
            incoming_proof: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<C::Base>) -> Self::Config {
        Sc::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<C::Base>,
    ) -> Result<(), Error> {
        let (input, output) =
            StepCircuit::synthesize(&self.step_circuit, config, layouter.namespace(|| ""))?;
        self.synthesize_accumulation_verifier(layouter.namespace(|| ""), &input, &output)?;
        Ok(())
    }
}

impl<C, Sc> CircuitExt<C::Base> for RecursiveCircuit<C, Sc>
where
    C: CurveAffine,
    Sc: StepCircuit<C>,
{
    fn instances(&self) -> Vec<Vec<C::Base>> {
        let mut instances = vec![vec![Self::DUMMY_H; 2]];
        self.incoming_instances[1].map(|h_ohs| instances[0][0] = fe_to_fe(h_ohs));
        self.h_prime.map(|h_prime| instances[0][1] = h_prime);
        instances
    }
}

pub struct ProtostarIvcProverParam<C, P1, P2, AT1, AT2>
where
    C: CurveCycle,
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

pub struct ProtostarIvcVerifierParam<C, P1, P2, H1, H2>
where
    C: CurveCycle,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
    H1: HashInstruction<C::Secondary>,
    H2: HashInstruction<C::Primary>,
{
    primary_vp: ProtostarVerifierParam<C::Scalar, HyperPlonk<P1>>,
    primary_vp_digest: C::Base,
    primary_hp: H1::Param,
    secondary_vp: ProtostarVerifierParam<C::Base, HyperPlonk<P2>>,
    secondary_vp_digest: C::Scalar,
    secondary_hp: H2::Param,
    _marker: PhantomData<(C, H1, H2)>,
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn preprocess<C, P1, P2, S1, S2, AT1, AT2>(
    primary_num_vars: usize,
    primary_atp: AT1::Param,
    primary_hp: <<S1::Chips as Chips<C::Secondary>>::HashChip as HashInstruction<C::Secondary>>::Param,
    primary_step_circuit: S1,
    secondary_num_vars: usize,
    secondary_atp: AT2::Param,
    secondary_hp: <<S2::Chips as Chips<C::Primary>>::HashChip as HashInstruction<C::Primary>>::Param,
    secondary_step_circuit: S2,
    mut rng: impl RngCore,
) -> Result<
    (
        Halo2Circuit<C::Scalar, RecursiveCircuit<C::Secondary, S1>>,
        Halo2Circuit<C::Base, RecursiveCircuit<C::Primary, S2>>,
        ProtostarIvcProverParam<C, P1, P2, AT1, AT2>,
        ProtostarIvcVerifierParam<
            C,
            P1,
            P2,
            <S1::Chips as Chips<C::Secondary>>::HashChip,
            <S2::Chips as Chips<C::Primary>>::HashChip,
        >,
    ),
    Error,
>
where
    C: CurveCycle,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::Scalar,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C::Primary,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    S1: StepCircuit<C::Secondary>,
    S2: StepCircuit<C::Primary>,
    AT1: InMemoryTranscript,
    AT2: InMemoryTranscript,
{
    let primary_param = P1::setup(1 << primary_num_vars, 0, &mut rng).unwrap();
    let secondary_param = P2::setup(1 << secondary_num_vars, 0, &mut rng).unwrap();

    let primary_circuit = RecursiveCircuit::new(true, primary_step_circuit, primary_hp, None);
    let mut primary_circuit =
        Halo2Circuit::new::<HyperPlonk<P1>>(primary_num_vars, primary_circuit);

    let (_, primary_vp) = {
        let primary_circuit_info = primary_circuit.circuit_info_without_preprocess().unwrap();
        Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_circuit_info).unwrap()
    };

    let secondary_circuit = RecursiveCircuit::new(
        false,
        secondary_step_circuit,
        secondary_hp,
        Some(ProtostarAccumulationVerifierParam::from(&primary_vp)),
    );
    let mut secondary_circuit =
        Halo2Circuit::new::<HyperPlonk<P2>>(secondary_num_vars, secondary_circuit);
    let (_, secondary_vp) = {
        let secondary_circuit_info = secondary_circuit.circuit_info().unwrap();
        Protostar::<HyperPlonk<P2>>::preprocess(&secondary_param, &secondary_circuit_info).unwrap()
    };

    primary_circuit.update_witness(|circuit| {
        circuit.init(ProtostarAccumulationVerifierParam::from(&secondary_vp));
    });
    let primary_circuit_info = primary_circuit.circuit_info().unwrap();
    let (primary_pp, primary_vp) =
        Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_circuit_info).unwrap();

    secondary_circuit.update_witness(|circuit| {
        circuit.init(ProtostarAccumulationVerifierParam::from(&primary_vp));
    });
    let secondary_circuit_info = secondary_circuit.circuit_info().unwrap();
    let (secondary_pp, secondary_vp) =
        Protostar::<HyperPlonk<P2>>::preprocess(&secondary_param, &secondary_circuit_info).unwrap();

    let ivc_pp = ProtostarIvcProverParam {
        primary_pp,
        primary_atp,
        secondary_pp,
        secondary_atp,
        _marker: PhantomData,
    };
    let ivc_vp = {
        let primary_vp_digest = primary_vp.digest();
        let secondary_vp_digest = secondary_vp.digest();
        ProtostarIvcVerifierParam {
            primary_vp,
            primary_vp_digest,
            primary_hp: primary_circuit.circuit().hp.clone(),
            secondary_vp,
            secondary_vp_digest,
            secondary_hp: secondary_circuit.circuit().hp.clone(),
            _marker: PhantomData,
        }
    };

    Ok((primary_circuit, secondary_circuit, ivc_pp, ivc_vp))
}

#[allow(clippy::type_complexity)]
pub fn prove_steps<C, P1, P2, S1, S2, AT1, AT2>(
    ivc_pp: &ProtostarIvcProverParam<C, P1, P2, AT1, AT2>,
    primary_circuit: &mut Halo2Circuit<C::Scalar, RecursiveCircuit<C::Secondary, S1>>,
    secondary_circuit: &mut Halo2Circuit<C::Base, RecursiveCircuit<C::Primary, S2>>,
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
    C: CurveCycle,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::Scalar,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C::Primary,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    S1: StepCircuit<C::Secondary>,
    S2: StepCircuit<C::Primary>,
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
    C: CurveCycle,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::Scalar,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C::Primary,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
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
    ivc_vp: &ProtostarIvcVerifierParam<C, P1, P2, H1, H2>,
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
    C: CurveCycle,
    C::Scalar: Hash + Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    P1: PolynomialCommitmentScheme<
        C::Scalar,
        Polynomial = MultilinearPolynomial<C::Scalar>,
        CommitmentChunk = C::Primary,
    >,
    P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
    P2: PolynomialCommitmentScheme<
        C::Base,
        Polynomial = MultilinearPolynomial<C::Base>,
        CommitmentChunk = C::Secondary,
    >,
    P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    H1: HashInstruction<C::Secondary>,
    H2: HashInstruction<C::Primary>,
{
    if H1::hash_state(
        &ivc_vp.primary_hp,
        ivc_vp.secondary_vp_digest,
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
        ivc_vp.primary_vp_digest,
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

#[cfg(test)]
mod strawman {
    use crate::{
        folding::protostar::{
            ivc::halo2::{
                AssignedProtostarAccumulatorInstance, Chips, FieldInstruction, HashInstruction,
                NativeEccInstruction, NativeFieldInstruction, TranscriptInstruction,
                UtilInstruction,
            },
            ProtostarAccumulatorInstance,
        },
        frontend::halo2::chip::halo2_wrong::{
            from_le_bits, integer_to_native, sum_with_coeff, to_le_bits_strict, PoseidonChip,
        },
        util::{
            arithmetic::{
                fe_from_bool, fe_from_le_bytes, fe_to_fe, fe_truncated, BitField, CurveAffine,
                Field, FromUniformBytes, PrimeField, PrimeFieldBits,
            },
            hash::{poseidon::Spec, Poseidon},
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
                Transcript, TranscriptRead, TranscriptWrite,
            },
            Itertools,
        },
    };
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, Value},
        plonk::{Column, ConstraintSystem, Error, Instance},
    };
    use halo2_wrong_v2::{
        integer::{
            chip::{IntegerChip, Range},
            rns::Rns,
            Integer,
        },
        maingate::{config::MainGate, operations::Collector, Gate},
        Composable, Scaled, Witness,
    };
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        fmt::{self, Debug},
        io::{self, Cursor, Read},
        iter,
        marker::PhantomData,
        rc::Rc,
    };

    const NUM_LIMBS: usize = 4;
    const NUM_LIMB_BITS: usize = 65;
    const NUM_SUBLIMBS: usize = 5;
    const NUM_LOOKUPS: usize = 1;

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    const NUM_CHALLENGE_BITS: usize = 128;
    const NUM_CHALLENGE_BYTES: usize = NUM_CHALLENGE_BITS / 8;

    const NUM_HASH_BITS: usize = 250;

    fn fe_to_limbs<F1: PrimeFieldBits, F2: PrimeField>(fe: F1, num_limb_bits: usize) -> Vec<F2> {
        fe.to_le_bits()
            .chunks(num_limb_bits)
            .into_iter()
            .map(|bits| match bits.len() {
                1..=64 => F2::from(bits.load_le()),
                65..=128 => {
                    let lo = bits[..64].load_le::<u64>();
                    let hi = bits[64..].load_le::<u64>();
                    F2::from(hi) * F2::from(2).pow_vartime([64]) + F2::from(lo)
                }
                _ => unimplemented!(),
            })
            .take(NUM_LIMBS)
            .collect()
    }

    fn x_y_is_identity<C: CurveAffine>(ec_point: &C) -> [C::Base; 3] {
        let coords = ec_point.coordinates().unwrap();
        let is_identity = (coords.x().is_zero() & coords.y().is_zero()).into();
        [*coords.x(), *coords.y(), fe_from_bool(is_identity)]
    }

    pub fn accumulation_transcript_param<F: FromUniformBytes<64>>() -> Spec<F, T, RATE> {
        Spec::new(R_F, R_P)
    }

    pub fn hash_param<F: FromUniformBytes<64>>() -> Spec<F, T, RATE> {
        Spec::new(R_F, R_P)
    }

    #[derive(Debug)]
    pub struct PoseidonTranscript<F: PrimeField, S> {
        state: Poseidon<F, T, RATE>,
        stream: S,
    }

    impl<F: FromUniformBytes<64>> InMemoryTranscript for PoseidonTranscript<F, Cursor<Vec<u8>>> {
        type Param = Spec<F, T, RATE>;

        fn new(spec: Self::Param) -> Self {
            Self {
                state: Poseidon::new_with_spec(spec),
                stream: Default::default(),
            }
        }

        fn into_proof(self) -> Vec<u8> {
            self.stream.into_inner()
        }

        fn from_proof(spec: Self::Param, proof: &[u8]) -> Self {
            Self {
                state: Poseidon::new_with_spec(spec),
                stream: Cursor::new(proof.to_vec()),
            }
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, S> FieldTranscript<F>
        for PoseidonTranscript<N, S>
    {
        fn squeeze_challenge(&mut self) -> F {
            let hash = self.state.squeeze();
            self.state.update(&[hash]);

            fe_from_le_bytes(&hash.to_repr().as_ref()[..NUM_CHALLENGE_BYTES])
        }

        fn common_field_element(&mut self, fe: &F) -> Result<(), crate::Error> {
            self.state.update(&fe_to_limbs(*fe, NUM_LIMB_BITS));

            Ok(())
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, R: io::Read> FieldTranscriptRead<F>
        for PoseidonTranscript<N, R>
    {
        fn read_field_element(&mut self) -> Result<F, crate::Error> {
            let mut repr = <F as PrimeField>::Repr::default();
            self.stream
                .read_exact(repr.as_mut())
                .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            let fe = F::from_repr_vartime(repr).ok_or_else(|| {
                crate::Error::Transcript(
                    io::ErrorKind::Other,
                    "Invalid field element encoding in proof".to_string(),
                )
            })?;
            self.common_field_element(&fe)?;
            Ok(fe)
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, W: io::Write> FieldTranscriptWrite<F>
        for PoseidonTranscript<N, W>
    {
        fn write_field_element(&mut self, fe: &F) -> Result<(), crate::Error> {
            self.common_field_element(fe)?;
            let repr = fe.to_repr();
            self.stream
                .write_all(repr.as_ref())
                .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))
        }
    }

    impl<C: CurveAffine, S> Transcript<C, C::Scalar> for PoseidonTranscript<C::Base, S>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn common_commitment(&mut self, ec_point: &C) -> Result<(), crate::Error> {
            self.state.update(&x_y_is_identity(ec_point));
            Ok(())
        }
    }

    impl<C: CurveAffine, R: io::Read> TranscriptRead<C, C::Scalar> for PoseidonTranscript<C::Base, R>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn read_commitment(&mut self) -> Result<C, crate::Error> {
            let mut reprs = [<C::Base as PrimeField>::Repr::default(); 2];
            for repr in &mut reprs {
                self.stream
                    .read_exact(repr.as_mut())
                    .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            }
            let [x, y] = reprs.map(<C::Base as PrimeField>::from_repr_vartime);
            let ec_point = x
                .zip(y)
                .and_then(|(x, y)| CurveAffine::from_xy(x, y).into())
                .ok_or_else(|| {
                    crate::Error::Transcript(
                        io::ErrorKind::Other,
                        "Invalid elliptic curve point encoding in proof".to_string(),
                    )
                })?;
            self.common_commitment(&ec_point)?;
            Ok(ec_point)
        }
    }

    impl<C: CurveAffine, W: io::Write> TranscriptWrite<C, C::Scalar> for PoseidonTranscript<C::Base, W>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn write_commitment(&mut self, ec_point: &C) -> Result<(), crate::Error> {
            self.common_commitment(ec_point)?;
            let coordinates = ec_point.coordinates().unwrap();
            for coordinate in [coordinates.x(), coordinates.y()] {
                let repr = coordinate.to_repr();
                self.stream
                    .write_all(repr.as_ref())
                    .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            }
            Ok(())
        }
    }

    pub type Config<F> = MainGate<F, NUM_LOOKUPS>;

    #[allow(clippy::type_complexity)]
    #[derive(Clone, Debug)]
    pub struct Chip<C: CurveAffine> {
        rns: Rns<C::Scalar, C::Base, NUM_LIMBS, NUM_LIMB_BITS, NUM_SUBLIMBS>,
        maingate: MainGate<C::Base, NUM_LOOKUPS>,
        collector: Rc<RefCell<Collector<C::Base>>>,
        cell_map: Rc<RefCell<BTreeMap<u32, AssignedCell<C::Base, C::Base>>>>,
        instance: Column<Instance>,
        poseidon_spec: Spec<C::Base, T, RATE>,
        _marker: PhantomData<C>,
    }

    impl<C: CurveAffine> Chips<C> for Chip<C>
    where
        C::Base: FromUniformBytes<64> + PrimeFieldBits,
        C::Scalar: PrimeFieldBits,
    {
        type AssignedCell = Witness<C::Base>;

        type EccChip = Self;
        type BaseChip = Self;
        type ScalarChip = Self;
        type UtilChip = Self;
        type HashChip = Self;
        type TranscriptChip<'a> = PoseidonTranscriptChip<'a, C>;

        fn ecc_chip(&self) -> &Self::EccChip {
            self
        }

        fn base_chip(&self) -> &Self::BaseChip {
            self
        }

        fn scalar_chip(&self) -> &Self::ScalarChip {
            self
        }

        fn util_chip(&self) -> &Self::UtilChip {
            self
        }

        fn hash_chip(&self) -> &Self::HashChip {
            self
        }

        fn transcript_chip<'a>(&'a self, proof: Value<&'a [u8]>) -> Self::TranscriptChip<'a> {
            let poseidon_chip = PoseidonChip::from_spec(
                &mut self.collector.borrow_mut(),
                self.poseidon_spec.clone(),
            );
            PoseidonTranscriptChip {
                poseidon_chip,
                chip: self,
                proof,
            }
        }
    }

    impl<C: CurveAffine> Chip<C> {
        pub fn configure(meta: &mut ConstraintSystem<C::Base>) -> Config<C::Base> {
            let rns =
                Rns::<C::Scalar, C::Base, NUM_LIMBS, NUM_LIMB_BITS, NUM_SUBLIMBS>::construct();
            let overflow_bit_lens = rns.overflow_lengths();
            let composition_bit_len = IntegerChip::<
                C::Scalar,
                C::Base,
                NUM_LIMBS,
                NUM_LIMB_BITS,
                NUM_SUBLIMBS,
            >::sublimb_bit_len();
            MainGate::<_, NUM_LOOKUPS>::configure(
                meta,
                vec![composition_bit_len],
                overflow_bit_lens,
            )
        }
    }

    impl<C: CurveAffine> Chip<C>
    where
        C::Base: Ord,
    {
        pub fn new(maingate: MainGate<C::Base, NUM_LOOKUPS>, instance: Column<Instance>) -> Self
        where
            C::Base: FromUniformBytes<64>,
        {
            Chip {
                rns: Rns::construct(),
                maingate,
                collector: Default::default(),
                cell_map: Default::default(),
                instance,
                poseidon_spec: Spec::new(R_F, R_P),
                _marker: PhantomData,
            }
        }

        fn negate_ec_point(&self, value: &AssignedEcPoint<C>) -> AssignedEcPoint<C> {
            let collector = &mut self.collector.borrow_mut();
            let y = collector.sub_from_constant(C::Base::ZERO, value.y());
            let mut out = value.clone();
            out.x_y_is_identity[1] = y;
            out.ec_point = out.ec_point.map(|ec_point| -ec_point);
            out
        }

        fn add_ec_point_incomplete(
            &self,
            lhs: &AssignedEcPoint<C>,
            rhs: &AssignedEcPoint<C>,
        ) -> AssignedEcPoint<C> {
            let collector = &mut self.collector.borrow_mut();
            let x_diff = collector.sub(rhs.x(), lhs.x());
            let y_diff = collector.sub(rhs.y(), lhs.y());
            let (x_diff_inv, _) = collector.inv(&x_diff);
            let lambda = collector.mul(&y_diff, &x_diff_inv);
            let lambda_square = collector.mul(&lambda, &lambda);
            let out_x = sum_with_coeff(
                collector,
                [
                    (&lambda_square, C::Base::ONE),
                    (lhs.x(), -C::Base::ONE),
                    (rhs.x(), -C::Base::ONE),
                ],
            );
            let out_y = {
                let x_diff = collector.sub(lhs.x(), &out_x);
                let lambda_x_diff = collector.mul(&lambda, &x_diff);
                collector.sub(&lambda_x_diff, lhs.y())
            };
            let out_is_identity = collector.register_constant(C::Base::ZERO);

            AssignedEcPoint {
                ec_point: (lhs.ec_point + rhs.ec_point).map(Into::into),
                x_y_is_identity: [out_x, out_y, out_is_identity],
            }
        }

        fn double_ec_point_incomplete(&self, value: &AssignedEcPoint<C>) -> AssignedEcPoint<C> {
            let collector = &mut self.collector.borrow_mut();
            let two = C::Base::ONE.double();
            let three = two + C::Base::ONE;
            let lambda_numer =
                collector.mul_add_constant_scaled(three, value.x(), value.x(), C::a());
            let y_doubled = collector.add(value.y(), value.y());
            let (lambda_denom_inv, _) = collector.inv(&y_doubled);
            let lambda = collector.mul(&lambda_numer, &lambda_denom_inv);
            let lambda_square = collector.mul(&lambda, &lambda);
            let out_x = collector.add_scaled(
                &Scaled::new(&lambda_square, C::Base::ONE),
                &Scaled::new(value.x(), -two),
            );
            let out_y = {
                let x_diff = collector.sub(value.x(), &out_x);
                let lambda_x_diff = collector.mul(&lambda, &x_diff);
                collector.sub(&lambda_x_diff, value.y())
            };
            AssignedEcPoint {
                ec_point: (value.ec_point + value.ec_point).map(Into::into),
                x_y_is_identity: [out_x, out_y, *value.is_identity()],
            }
        }

        fn add_ec_point_inner(
            &self,
            lhs: &AssignedEcPoint<C>,
            rhs: &AssignedEcPoint<C>,
        ) -> (AssignedEcPoint<C>, Witness<C::Base>, Witness<C::Base>) {
            let collector = &mut self.collector.borrow_mut();
            let x_diff = collector.sub(rhs.x(), lhs.x());
            let y_diff = collector.sub(rhs.y(), lhs.y());
            let (x_diff_inv, is_x_equal) = collector.inv(&x_diff);
            let (_, is_y_equal) = collector.inv(&y_diff);
            let lambda = collector.mul(&y_diff, &x_diff_inv);
            let lambda_square = collector.mul(&lambda, &lambda);
            let out_x = sum_with_coeff(
                collector,
                [
                    (&lambda_square, C::Base::ONE),
                    (lhs.x(), -C::Base::ONE),
                    (rhs.x(), -C::Base::ONE),
                ],
            );
            let out_y = {
                let x_diff = collector.sub(lhs.x(), &out_x);
                let lambda_x_diff = collector.mul(&lambda, &x_diff);
                collector.sub(&lambda_x_diff, lhs.y())
            };
            let out_x = collector.select(rhs.is_identity(), lhs.x(), &out_x);
            let out_x = collector.select(lhs.is_identity(), rhs.x(), &out_x);
            let out_y = collector.select(rhs.is_identity(), lhs.y(), &out_y);
            let out_y = collector.select(lhs.is_identity(), rhs.y(), &out_y);
            let out_is_identity = collector.mul(lhs.is_identity(), rhs.is_identity());

            let out = AssignedEcPoint {
                ec_point: (lhs.ec_point + rhs.ec_point).map(Into::into),
                x_y_is_identity: [out_x, out_y, out_is_identity],
            };
            (out, is_x_equal, is_y_equal)
        }

        fn double_ec_point(&self, value: &AssignedEcPoint<C>) -> AssignedEcPoint<C> {
            let doubled = self.double_ec_point_incomplete(value);
            let collector = &mut self.collector.borrow_mut();
            let zero = collector.register_constant(C::Base::ZERO);
            let out_x = collector.select(value.is_identity(), &zero, doubled.x());
            let out_y = collector.select(value.is_identity(), &zero, doubled.y());
            AssignedEcPoint {
                ec_point: (value.ec_point + value.ec_point).map(Into::into),
                x_y_is_identity: [out_x, out_y, *value.is_identity()],
            }
        }
    }

    #[derive(Clone)]
    pub struct AssignedEcPoint<C: CurveAffine> {
        ec_point: Value<C>,
        x_y_is_identity: [Witness<C::Base>; 3],
    }

    impl<C: CurveAffine> AssignedEcPoint<C> {
        fn x(&self) -> &Witness<C::Base> {
            &self.x_y_is_identity[0]
        }

        fn y(&self) -> &Witness<C::Base> {
            &self.x_y_is_identity[1]
        }

        fn is_identity(&self) -> &Witness<C::Base> {
            &self.x_y_is_identity[2]
        }
    }

    impl<C: CurveAffine> AsRef<[Witness<C::Base>]> for AssignedEcPoint<C> {
        fn as_ref(&self) -> &[Witness<C::Base>] {
            &self.x_y_is_identity
        }
    }

    impl<C: CurveAffine> Debug for AssignedEcPoint<C> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedEcPoint");
            let mut value = None;
            self.ec_point.map(|ec_point| value = Some(ec_point));
            if let Some(value) = value {
                s.field("ec_point", &value).finish()
            } else {
                s.finish()
            }
        }
    }

    impl<C: CurveAffine> NativeEccInstruction<C> for Chip<C>
    where
        C::Scalar: Ord + PrimeFieldBits,
        C::Base: Ord,
    {
        type AssignedCell = Witness<C::Base>;
        type Assigned = AssignedEcPoint<C>;

        fn assign_constant(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            constant: C,
        ) -> Result<Self::Assigned, Error> {
            let x_y_is_identity = x_y_is_identity(&constant).map(|value| {
                NativeFieldInstruction::assign_constant(self, layouter, value).unwrap()
            });
            Ok(AssignedEcPoint {
                ec_point: Value::known(constant),
                x_y_is_identity,
            })
        }

        fn assign_witness(
            &self,
            _: &mut impl Layouter<C::Base>,
            witness: Value<C>,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let zero = collector.register_constant(C::Base::ZERO);
            let one = collector.register_constant(C::Base::ONE);
            let [x, y, is_identity] = witness
                .as_ref()
                .map(x_y_is_identity)
                .transpose_array()
                .map(|value| collector.new_witness(value));
            collector.assert_bit(&is_identity);
            let not_identity = collector.sub(&one, &is_identity);
            let lhs = collector.mul(&y, &y);
            let lhs = collector.mul(&lhs, &not_identity);
            let x_square_plus_a = collector.mul_add_constant_scaled(C::Base::ONE, &x, &x, C::a());
            let rhs = collector.mul_add_constant_scaled(C::Base::ONE, &x_square_plus_a, &x, C::b());
            let rhs = collector.mul(&rhs, &not_identity);
            collector.equal(&lhs, &rhs);
            let x = collector.select(&is_identity, &zero, &x);
            let y = collector.select(&is_identity, &zero, &y);
            Ok(AssignedEcPoint {
                ec_point: witness,
                x_y_is_identity: [x, y, is_identity],
            })
        }

        fn select(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            condition: &Self::AssignedCell,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let x_y_is_identity = when_true
                .x_y_is_identity
                .iter()
                .zip(when_false.x_y_is_identity.iter())
                .map(|(when_true, when_false)| {
                    NativeFieldInstruction::select(self, layouter, condition, when_true, when_false)
                })
                .try_collect::<_, Vec<_>, _>()?
                .try_into()
                .unwrap();
            let output = condition
                .value()
                .zip(when_true.ec_point.zip(when_false.ec_point))
                .map(|(condition, (when_true, when_false))| {
                    if condition == C::Base::ONE {
                        when_true
                    } else {
                        when_false
                    }
                });
            Ok(AssignedEcPoint {
                ec_point: output,
                x_y_is_identity,
            })
        }

        fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C) -> bool) {
            value.ec_point.assert_if_known(f)
        }

        fn add(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let (out_added, is_x_equal, is_y_equal) = self.add_ec_point_inner(lhs, rhs);
            let out_doubled = self.double_ec_point(lhs);
            let identity = NativeEccInstruction::assign_constant(self, layouter, C::identity())?;
            let out =
                NativeEccInstruction::select(self, layouter, &is_y_equal, &out_doubled, &identity)?;
            NativeEccInstruction::select(self, layouter, &is_x_equal, &out, &out_added)
        }

        fn mul(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            base: &Self::Assigned,
            le_bits: &[Self::AssignedCell],
        ) -> Result<Self::Assigned, Error> {
            assert!(le_bits.len() < (C::Scalar::NUM_BITS - 2) as usize);

            let base_neg = self.negate_ec_point(base);

            let mut acc = base.clone();
            let mut base = base.clone();

            for bit in le_bits.iter().skip(1) {
                base = self.double_ec_point_incomplete(&base);
                let acc_plus_base = self.add_ec_point_incomplete(&acc, &base);
                let [acc_x, acc_y, _] = &mut acc.x_y_is_identity;
                *acc_x =
                    NativeFieldInstruction::select(self, layouter, bit, acc_plus_base.x(), acc_x)?;
                *acc_y =
                    NativeFieldInstruction::select(self, layouter, bit, acc_plus_base.y(), acc_y)?;
            }

            let acc_minus_base = self.add_ec_point_incomplete(&acc, &base_neg);
            let out =
                NativeEccInstruction::select(self, layouter, &le_bits[0], &acc, &acc_minus_base)?;
            let identity = NativeEccInstruction::assign_constant(self, layouter, C::identity())?;
            NativeEccInstruction::select(self, layouter, base.is_identity(), &identity, &out)
        }
    }

    impl<C: CurveAffine> NativeFieldInstruction<C::Base> for Chip<C>
    where
        C::Scalar: PrimeFieldBits,
    {
        type AssignedCell = Witness<C::Base>;

        fn assign_constant(
            &self,
            _: &mut impl Layouter<C::Base>,
            constant: C::Base,
        ) -> Result<Self::AssignedCell, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.register_constant(constant))
        }

        fn assign_witness(
            &self,
            _: &mut impl Layouter<C::Base>,
            witness: Value<C::Base>,
        ) -> Result<Self::AssignedCell, Error> {
            let collector = &mut self.collector.borrow_mut();
            let value = collector.new_witness(witness);
            Ok(collector.add_constant(&value, C::Base::ZERO))
        }

        fn is_equal(
            &self,
            _: &mut impl Layouter<C::Base>,
            lhs: &Self::AssignedCell,
            rhs: &Self::AssignedCell,
        ) -> Result<Self::AssignedCell, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.is_equal(lhs, rhs))
        }

        fn select(
            &self,
            _: &mut impl Layouter<C::Base>,
            condition: &Self::AssignedCell,
            when_true: &Self::AssignedCell,
            when_false: &Self::AssignedCell,
        ) -> Result<Self::AssignedCell, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.select(condition, when_true, when_false))
        }

        fn assert_if_known(&self, value: &Self::AssignedCell, f: impl FnOnce(&C::Base) -> bool) {
            value.value().assert_if_known(f)
        }

        fn add(
            &self,
            _: &mut impl Layouter<C::Base>,
            lhs: &Self::AssignedCell,
            rhs: &Self::AssignedCell,
        ) -> Result<Self::AssignedCell, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.add(lhs, rhs))
        }
    }

    #[derive(Clone)]
    pub struct AssignedScalar<F: PrimeField, N: PrimeField> {
        scalar: Integer<F, N, NUM_LIMBS, NUM_LIMB_BITS>,
        limbs: Vec<Witness<N>>,
    }

    impl<F: PrimeField, N: PrimeField> AsRef<[Witness<N>]> for AssignedScalar<F, N> {
        fn as_ref(&self) -> &[Witness<N>] {
            &self.limbs
        }
    }

    impl<F: PrimeField, N: PrimeField> Debug for AssignedScalar<F, N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedScalar");
            let mut value = None;
            self.scalar.value().map(|scalar| value = Some(scalar));
            if let Some(value) = value {
                s.field("scalar", &value).finish()
            } else {
                s.finish()
            }
        }
    }

    impl<C: CurveAffine> FieldInstruction<C::Scalar, C::Base> for Chip<C>
    where
        C::Scalar: PrimeFieldBits,
    {
        type AssignedCell = Witness<C::Base>;
        type Assigned = AssignedScalar<C::Scalar, C::Base>;

        fn assign_constant(
            &self,
            _: &mut impl Layouter<C::Base>,
            constant: C::Scalar,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.register_constant(constant);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedScalar { scalar, limbs })
        }

        fn assign_witness(
            &self,
            _: &mut impl Layouter<C::Base>,
            witness: Value<C::Scalar>,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.range(self.rns.from_fe(witness), Range::Remainder);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedScalar { scalar, limbs })
        }

        fn fit_in_native(
            &self,
            _: &mut impl Layouter<C::Base>,
            value: &Self::Assigned,
        ) -> Result<Self::AssignedCell, Error> {
            Ok(integer_to_native(
                &self.rns,
                &mut self.collector.borrow_mut(),
                &value.scalar,
                NUM_HASH_BITS,
            ))
        }

        fn select(
            &self,
            _: &mut impl Layouter<C::Base>,
            condition: &Self::AssignedCell,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.select(&when_true.scalar, &when_false.scalar, condition);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedScalar { scalar, limbs })
        }

        fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C::Scalar) -> bool) {
            value.scalar.value().assert_if_known(f)
        }

        fn add(
            &self,
            _: &mut impl Layouter<C::Base>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.add(&lhs.scalar, &rhs.scalar);
            let scalar = integer_chip.reduce(&scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedScalar { scalar, limbs })
        }

        fn mul(
            &self,
            _: &mut impl Layouter<C::Base>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.mul(&lhs.scalar, &rhs.scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedScalar { scalar, limbs })
        }
    }

    impl<C: CurveAffine> UtilInstruction<C::Base> for Chip<C>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        type AssignedCell = Witness<C::Base>;

        fn convert(
            &self,
            _: &mut impl Layouter<C::Base>,
            value: &AssignedCell<C::Base, C::Base>,
        ) -> Result<Self::AssignedCell, Error> {
            Ok(self.collector.borrow_mut().new_external(value))
        }

        fn constrain_equal(
            &self,
            _: &mut impl Layouter<C::Base>,
            lhs: &Self::AssignedCell,
            rhs: &Self::AssignedCell,
        ) -> Result<(), Error> {
            self.collector.borrow_mut().equal(lhs, rhs);
            Ok(())
        }

        fn constrain_instance(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            assigned: &Self::AssignedCell,
            row: usize,
        ) -> Result<(), Error> {
            if self.cell_map.borrow().is_empty() {
                *self.cell_map.borrow_mut() =
                    self.maingate.layout(layouter, &self.collector.borrow())?;
            }
            let cell = self.cell_map.borrow()[&assigned.id()].cell();
            layouter.constrain_instance(cell, self.instance, row)?;

            if row == 1 {
                *self.collector.borrow_mut() = Default::default();
                *self.cell_map.borrow_mut() = Default::default();
            }

            Ok(())
        }
    }

    impl<C: CurveAffine> HashInstruction<C> for Chip<C>
    where
        C::Base: FromUniformBytes<64> + PrimeFieldBits,
        C::ScalarExt: PrimeFieldBits,
    {
        type Param = Spec<C::Base, T, RATE>;
        type AssignedCell = Witness<C::Base>;

        fn hash_state<Comm: AsRef<C>>(
            spec: &Self::Param,
            vp_digest: C::Base,
            step_idx: usize,
            initial_input: &[C::Base],
            output: &[C::Base],
            acc: &ProtostarAccumulatorInstance<C::Scalar, Comm>,
        ) -> C::Base {
            let mut poseidon = Poseidon::new_with_spec(spec.clone());
            let fe_to_limbs = |fe| fe_to_limbs(fe, NUM_LIMB_BITS);
            let inputs = iter::empty()
                .chain([vp_digest, C::Base::from(step_idx as u64)])
                .chain(initial_input.iter().copied())
                .chain(output.iter().copied())
                .chain(fe_to_limbs(acc.instances[0][0]))
                .chain(fe_to_limbs(acc.instances[0][1]))
                .chain(
                    acc.witness_comms
                        .iter()
                        .map(AsRef::as_ref)
                        .flat_map(x_y_is_identity),
                )
                .chain(acc.challenges.iter().copied().flat_map(fe_to_limbs))
                .chain(fe_to_limbs(acc.u))
                .chain(x_y_is_identity(acc.e_comm.as_ref()))
                .chain(acc.compressed_e_sum.map(fe_to_limbs).into_iter().flatten())
                .collect_vec();
            poseidon.update(&inputs);
            fe_truncated(poseidon.squeeze(), NUM_HASH_BITS)
        }

        fn hash_assigned_state<EccChip, ScalarChip>(
            &self,
            _: &mut impl Layouter<C::Base>,
            vp_digest: &Witness<C::Base>,
            step_idx: &Witness<C::Base>,
            initial_input: &[Witness<C::Base>],
            output: &[Witness<C::Base>],
            acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        ) -> Result<Witness<C::Base>, Error>
        where
            EccChip: NativeEccInstruction<C, AssignedCell = Self::AssignedCell>,
            ScalarChip: FieldInstruction<C::Scalar, C::Base, AssignedCell = Self::AssignedCell>,
        {
            let collector = &mut self.collector.borrow_mut();
            let inputs = iter::empty()
                .chain([vp_digest, step_idx])
                .chain(initial_input)
                .chain(output)
                .chain(AsRef::as_ref(&acc.instances[0][0]))
                .chain(AsRef::as_ref(&acc.instances[0][1]))
                .chain(acc.witness_comms.iter().flat_map(AsRef::as_ref))
                .chain(acc.challenges.iter().flat_map(AsRef::<[_]>::as_ref))
                .chain(AsRef::as_ref(&acc.u))
                .chain(acc.e_comm.as_ref())
                .chain(
                    acc.compressed_e_sum
                        .as_ref()
                        .map(AsRef::as_ref)
                        .into_iter()
                        .flatten(),
                )
                .copied()
                .collect_vec();
            let mut poseidon_chip = PoseidonChip::from_spec(collector, self.poseidon_spec.clone());
            poseidon_chip.update(&inputs);
            let hash = poseidon_chip.squeeze(collector);
            let hash_le_bits = to_le_bits_strict(collector, &hash);
            Ok(from_le_bits(collector, &hash_le_bits[..NUM_HASH_BITS]))
        }
    }

    #[derive(Clone, Debug)]
    pub struct PoseidonTranscriptChip<'a, C: CurveAffine> {
        poseidon_chip: PoseidonChip<C::Base, T, RATE>,
        chip: &'a Chip<C>,
        proof: Value<&'a [u8]>,
    }

    #[derive(Clone, Debug)]
    pub struct Challenge<F: PrimeField, N: PrimeField> {
        le_bits: Vec<Witness<N>>,
        scalar: AssignedScalar<F, N>,
    }

    impl<F: PrimeField, N: PrimeField> AsRef<AssignedScalar<F, N>> for Challenge<F, N> {
        fn as_ref(&self) -> &AssignedScalar<F, N> {
            &self.scalar
        }
    }

    impl<'a, C> TranscriptInstruction<C, Chip<C>, Chip<C>> for PoseidonTranscriptChip<'a, C>
    where
        C: CurveAffine,
        C::Base: FromUniformBytes<64> + PrimeFieldBits,
        C::Scalar: PrimeFieldBits,
    {
        type Challenge = Challenge<C::Scalar, C::Base>;

        fn challenge_to_le_bits(
            &self,
            _: &mut impl Layouter<<C as CurveAffine>::Base>,
            challenge: &Self::Challenge,
        ) -> Result<Vec<Witness<C::Base>>, Error> {
            Ok(challenge.le_bits.clone())
        }

        fn common_field_element(
            &mut self,
            value: &AssignedScalar<C::Scalar, C::Base>,
        ) -> Result<(), Error> {
            AsRef::<[_]>::as_ref(value)
                .iter()
                .for_each(|value| self.poseidon_chip.update(&[*value]));
            Ok(())
        }

        fn common_commitment(&mut self, value: &AssignedEcPoint<C>) -> Result<(), Error> {
            value
                .as_ref()
                .iter()
                .for_each(|value| self.poseidon_chip.update(&[*value]));
            Ok(())
        }

        fn read_field_element(
            &mut self,
            layouter: &mut impl Layouter<C::Base>,
        ) -> Result<AssignedScalar<C::Scalar, C::Base>, Error> {
            let fe = self.proof.as_mut().and_then(|proof| {
                let mut repr = <C::Scalar as PrimeField>::Repr::default();
                if proof.read_exact(repr.as_mut()).is_err() {
                    return Value::unknown();
                }
                C::Scalar::from_repr_vartime(repr)
                    .map(Value::known)
                    .unwrap_or_else(Value::unknown)
            });
            let fe = FieldInstruction::assign_witness(self.chip, layouter, fe)?;
            self.common_field_element(&fe)?;
            Ok(fe)
        }

        fn read_commitment(
            &mut self,
            layouter: &mut impl Layouter<C::Base>,
        ) -> Result<AssignedEcPoint<C>, Error> {
            let comm = self.proof.as_mut().and_then(|proof| {
                let mut reprs = [<C::Base as PrimeField>::Repr::default(); 2];
                for repr in &mut reprs {
                    if proof.read_exact(repr.as_mut()).is_err() {
                        return Value::unknown();
                    }
                }
                let [x, y] = reprs.map(|repr| {
                    C::Base::from_repr_vartime(repr)
                        .map(Value::known)
                        .unwrap_or_else(Value::unknown)
                });
                x.zip(y).and_then(|(x, y)| {
                    Option::from(C::from_xy(x, y))
                        .map(Value::known)
                        .unwrap_or_else(Value::unknown)
                })
            });
            let comm = NativeEccInstruction::assign_witness(self.chip, layouter, comm)?;
            self.common_commitment(&comm)?;
            Ok(comm)
        }

        fn squeeze_challenge(
            &mut self,
            _: &mut impl Layouter<C::Base>,
        ) -> Result<Challenge<C::Scalar, C::Base>, Error> {
            let collector = &mut self.chip.collector.borrow_mut();
            let (challenge_le_bits, challenge) = {
                let hash = self.poseidon_chip.squeeze(collector);
                self.poseidon_chip.update(&[hash]);

                let challenge_le_bits = to_le_bits_strict(collector, &hash)
                    .into_iter()
                    .take(NUM_CHALLENGE_BITS)
                    .collect_vec();
                let challenge = from_le_bits(collector, &challenge_le_bits);

                (challenge_le_bits, challenge)
            };

            let mut integer_chip = IntegerChip::new(collector, &self.chip.rns);
            let limbs = self.chip.rns.from_fe(challenge.value().map(fe_to_fe));
            let scalar = integer_chip.range(limbs, Range::Remainder);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();

            let scalar_in_base =
                integer_to_native(&self.chip.rns, collector, &scalar, NUM_CHALLENGE_BITS);
            collector.equal(&challenge, &scalar_in_base);

            Ok(Challenge {
                le_bits: challenge_le_bits,
                scalar: AssignedScalar { scalar, limbs },
            })
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        folding::protostar::ivc::halo2::{
            preprocess, prove_decider, prove_steps, strawman, verify_decider, StepCircuit,
        },
        frontend::halo2::CircuitExt,
        pcs::{
            multilinear::{MultilinearIpa, MultilinearSimulator},
            univariate::UnivariateKzg,
            AdditiveCommitment, PolynomialCommitmentScheme,
        },
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::{
                Bn254Grumpkin, CurveAffine, CurveCycle, FromUniformBytes, PrimeFieldBits,
            },
            test::seeded_std_rng,
            transcript::{
                InMemoryTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite,
            },
            DeserializeOwned, Serialize,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, SimpleFloorPlanner},
        plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
    };
    use std::{fmt::Debug, hash::Hash, io::Cursor, marker::PhantomData};

    #[derive(Clone, Debug, Default)]
    struct TrivialCircuit<C> {
        step_idx: usize,
        _marker: PhantomData<C>,
    }

    impl<C: CurveAffine> Circuit<C::Base> for TrivialCircuit<C> {
        type Config = (strawman::Config<C::Base>, Column<Instance>);
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<C::Base>) -> Self::Config {
            let main_gate = strawman::Chip::<C>::configure(meta);
            let instance = meta.instance_column();
            meta.enable_equality(instance);
            (main_gate, instance)
        }

        fn synthesize(&self, _: Self::Config, _: impl Layouter<C::Base>) -> Result<(), Error> {
            Ok(())
        }
    }

    impl<C: CurveAffine> CircuitExt<C::Base> for TrivialCircuit<C> {
        fn instances(&self) -> Vec<Vec<C::Base>> {
            Vec::new()
        }
    }

    impl<C: CurveAffine> StepCircuit<C> for TrivialCircuit<C>
    where
        C::Base: FromUniformBytes<64> + PrimeFieldBits,
        C::Scalar: PrimeFieldBits,
    {
        type Chips = strawman::Chip<C>;

        fn chips(&self, (main_gate, instance): Self::Config) -> Self::Chips {
            strawman::Chip::new(main_gate, instance)
        }

        fn step_idx(&self) -> usize {
            self.step_idx
        }

        fn initial_input(&self) -> &[C::Base] {
            &[]
        }

        fn input(&self) -> &[C::Base] {
            &[]
        }

        fn output(&self) -> &[C::Base] {
            &[]
        }

        fn next(&mut self) {
            self.step_idx += 1;
        }

        fn synthesize(
            &self,
            _: Self::Config,
            _: impl Layouter<C::Base>,
        ) -> Result<
            (
                Vec<AssignedCell<C::Base, C::Base>>,
                Vec<AssignedCell<C::Base, C::Base>>,
            ),
            Error,
        > {
            Ok((Vec::new(), Vec::new()))
        }
    }

    fn run_protostar_hyperplonk_ivc<C, P1, P2>(num_vars: usize, num_steps: usize)
    where
        C: CurveCycle,
        C::Scalar: Hash + Serialize + DeserializeOwned,
        C::Base: Hash + Serialize + DeserializeOwned,
        P1: PolynomialCommitmentScheme<
            C::Scalar,
            Polynomial = MultilinearPolynomial<C::Scalar>,
            CommitmentChunk = C::Primary,
        >,
        P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
        P2: PolynomialCommitmentScheme<
            C::Base,
            Polynomial = MultilinearPolynomial<C::Base>,
            CommitmentChunk = C::Secondary,
        >,
        P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
        Keccak256Transcript<Cursor<Vec<u8>>>: TranscriptRead<C::Primary, C::Scalar>
            + TranscriptRead<C::Secondary, C::Base>
            + TranscriptWrite<C::Primary, C::Scalar>
            + TranscriptWrite<C::Secondary, C::Base>,
    {
        let primary_num_vars = num_vars;
        let primary_atp = strawman::accumulation_transcript_param();
        let primary_hp = strawman::hash_param();
        let secondary_num_vars = num_vars;
        let secondary_atp = strawman::accumulation_transcript_param();
        let secondary_hp = strawman::hash_param();

        let (mut primary_circuit, mut secondary_circuit, ivc_pp, ivc_vp) = preprocess::<
            C,
            P1,
            P2,
            _,
            _,
            strawman::PoseidonTranscript<_, _>,
            strawman::PoseidonTranscript<_, _>,
        >(
            primary_num_vars,
            primary_atp,
            primary_hp,
            TrivialCircuit::default(),
            secondary_num_vars,
            secondary_atp,
            secondary_hp,
            TrivialCircuit::default(),
            seeded_std_rng(),
        )
        .unwrap();

        let (primary_acc, mut secondary_acc, secondary_last_instances) = prove_steps(
            &ivc_pp,
            &mut primary_circuit,
            &mut secondary_circuit,
            num_steps,
            seeded_std_rng(),
        )
        .unwrap();

        let (
            primary_acc,
            primary_initial_input,
            primary_output,
            primary_proof,
            secondary_acc_before_last,
            secondary_initial_input,
            secondary_output,
            secondary_proof,
        ) = {
            let secondary_acc_before_last = secondary_acc.instance.clone();

            let mut primary_transcript = Keccak256Transcript::default();
            let mut secondary_transcript = Keccak256Transcript::default();
            prove_decider(
                &ivc_pp,
                &primary_acc,
                &mut primary_transcript,
                &mut secondary_acc,
                &secondary_circuit,
                &mut secondary_transcript,
                seeded_std_rng(),
            )
            .unwrap();

            (
                primary_acc.instance,
                StepCircuit::<C::Secondary>::initial_input(&primary_circuit.circuit().step_circuit),
                StepCircuit::<C::Secondary>::output(&primary_circuit.circuit().step_circuit),
                primary_transcript.into_proof(),
                secondary_acc_before_last,
                StepCircuit::<C::Primary>::initial_input(&secondary_circuit.circuit().step_circuit),
                StepCircuit::<C::Primary>::output(&secondary_circuit.circuit().step_circuit),
                secondary_transcript.into_proof(),
            )
        };

        let result = {
            let mut primary_transcript =
                Keccak256Transcript::from_proof((), primary_proof.as_slice());
            let mut secondary_transcript =
                Keccak256Transcript::from_proof((), secondary_proof.as_slice());
            verify_decider(
                &ivc_vp,
                primary_initial_input,
                primary_output,
                primary_acc,
                &mut primary_transcript,
                secondary_initial_input,
                secondary_output,
                secondary_acc_before_last,
                &[secondary_last_instances],
                &mut secondary_transcript,
                num_steps,
                seeded_std_rng(),
            )
        };
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn kzg_protostar_folding_verifier() {
        const NUM_VARS: usize = 16;
        const NUM_STEPS: usize = 3;
        run_protostar_hyperplonk_ivc::<
            Bn254Grumpkin,
            MultilinearSimulator<UnivariateKzg<Bn256>>,
            MultilinearIpa<grumpkin::G1Affine>,
        >(NUM_VARS, NUM_STEPS);
    }
}
