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
        izip_eq,
        transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
};
use halo2_proofs::{
    circuit::{AssignedCell, Cell, Layouter, Value},
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
    type Assigned: Clone + Debug + AsRef<[AssignedCell<C::Base, C::Base>]>;

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
        condition: &AssignedCell<C::Base, C::Base>,
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
        le_bits: &[AssignedCell<C::Base, C::Base>],
    ) -> Result<Self::Assigned, Error>;
}

pub trait FieldInstruction<F: PrimeField, N: PrimeField>: Clone + Debug {
    type Assigned: Clone + Debug + AsRef<[AssignedCell<N, N>]>;

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
    ) -> Result<AssignedCell<N, N>, Error>;

    fn select(
        &self,
        layouter: &mut impl Layouter<N>,
        condition: &AssignedCell<N, N>,
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

pub trait TranscriptInstruction<C: CurveAffine, EccChip, ScalarChip>: Clone + Debug
where
    EccChip: NativeEccInstruction<C>,
    ScalarChip: FieldInstruction<C::Scalar, C::Base>,
{
    type Challenge: Clone + Debug + AsRef<ScalarChip::Assigned>;

    fn dummy_proof(avp: &ProtostarAccumulationVerifierParam<C::Base>) -> Vec<u8> {
        let g = C::generator().coordinates().unwrap();
        let g_x = g.x().to_repr();
        let g_y = g.y().to_repr();
        let zero = C::Scalar::ZERO.to_repr();
        iter::empty()
            .chain(
                iter::repeat_with(|| [g_x.as_ref(), g_y.as_ref()])
                    .take(avp.num_folding_witness_polys())
                    .flatten()
                    .flatten(),
            )
            .chain(match avp.strategy {
                NoCompressing => iter::empty()
                    .chain(iter::repeat([g_x.as_ref(), g_y.as_ref()]).take(avp.num_cross_terms))
                    .flatten()
                    .flatten()
                    .collect_vec(),
                Compressing => iter::empty()
                    .chain([g_x.as_ref(), g_y.as_ref()])
                    .chain(iter::repeat(zero.as_ref()).take(avp.num_cross_terms))
                    .flatten()
                    .collect_vec(),
            })
            .copied()
            .collect()
    }

    fn init(&self, proof: Value<&[u8]>) -> Self;

    #[allow(clippy::type_complexity)]
    fn challenge_to_le_bits(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        scalar: &Self::Challenge,
    ) -> Result<Vec<AssignedCell<C::Base, C::Base>>, Error>;

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

pub trait HashInstruction<C: CurveAffine>: Clone + Debug {
    type Param: Clone + Debug;

    fn param(&self) -> Self::Param;

    fn hash_state<Comm: AsRef<C>>(
        param: Self::Param,
        vp_digest: C::Base,
        step_idx: usize,
        initial_input: &[C::Base],
        output: &[C::Base],
        acc: &ProtostarAccumulatorInstance<C::Scalar, Comm>,
    ) -> C::Base;

    fn hash_assigned_state<EccChip, ScalarChip>(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        vp_digest: &AssignedCell<C::Base, C::Base>,
        step_idx: &AssignedCell<C::Base, C::Base>,
        initial_input: &[AssignedCell<C::Base, C::Base>],
        output: &[AssignedCell<C::Base, C::Base>],
        acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
    ) -> Result<AssignedCell<C::Base, C::Base>, Error>
    where
        EccChip: NativeEccInstruction<C>,
        ScalarChip: FieldInstruction<C::Scalar, C::Base>;
}

pub trait UtilInstruction<N: PrimeField>: Clone + Debug {
    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<N>,
        constant: N,
    ) -> Result<AssignedCell<N, N>, Error>;

    fn assign_witness(
        &self,
        layouter: &mut impl Layouter<N>,
        witness: Value<N>,
    ) -> Result<AssignedCell<N, N>, Error>;

    fn is_equal(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &AssignedCell<N, N>,
        rhs: &AssignedCell<N, N>,
    ) -> Result<AssignedCell<N, N>, Error>;

    fn select(
        &self,
        layouter: &mut impl Layouter<N>,
        condition: &AssignedCell<N, N>,
        when_true: &AssignedCell<N, N>,
        when_false: &AssignedCell<N, N>,
    ) -> Result<AssignedCell<N, N>, Error>;

    fn add(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &AssignedCell<N, N>,
        rhs: &AssignedCell<N, N>,
    ) -> Result<AssignedCell<N, N>, Error>;

    fn constrain_equal(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &AssignedCell<N, N>,
        rhs: &AssignedCell<N, N>,
    ) -> Result<(), Error> {
        lhs.value().zip(rhs.value()).assert_if_known(|(lhs, rhs)| {
            assert_eq!(lhs, rhs);
            lhs == rhs
        });
        layouter.assign_region(
            || "",
            |mut region| region.constrain_equal(lhs.cell(), rhs.cell()),
        )
    }

    fn constrain_constant(
        &self,
        layouter: &mut impl Layouter<N>,
        lhs: &AssignedCell<N, N>,
        constant: N,
    ) -> Result<(), Error> {
        let constant = self.assign_constant(layouter, constant)?;
        self.constrain_equal(layouter, lhs, &constant)
    }

    fn constrain_instance(
        &self,
        layouter: &mut impl Layouter<N>,
        cell: Cell,
        row: usize,
    ) -> Result<(), Error>;
}

pub trait Chips<C: CurveAffine>: Clone + Debug {
    type EccChip: NativeEccInstruction<C>;
    type ScalarChip: FieldInstruction<C::Scalar, C::Base>;
    type TranscriptChip: TranscriptInstruction<C, Self::EccChip, Self::ScalarChip>;
    type HashChip: HashInstruction<C>;
    type UtilChip: UtilInstruction<C::Base>;

    fn ecc_chip(&self) -> &Self::EccChip;

    fn scalar_chip(&self) -> &Self::ScalarChip;

    fn transcript_chip(&self) -> &Self::TranscriptChip;

    fn hash_chip(&self) -> &Self::HashChip;

    fn util_chip(&self) -> &Self::UtilChip;
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
    ScalarChip: FieldInstruction<C::Scalar, C::Base>,
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
        r_le_bits: &[AssignedCell<C::Base, C::Base>],
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
        condition: &AssignedCell<C::Base, C::Base>,
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
        avp: Option<ProtostarAccumulationVerifierParam<C::Base>>,
    ) -> Self {
        let config = Self::configure(&mut Default::default());
        let chips = step_circuit.chips(config.clone());
        let hp = chips.hash_chip().param();
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
            self.hp.clone(),
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
        is_base_case: &AssignedCell<C::Base, C::Base>,
        initial_input: &[AssignedCell<C::Base, C::Base>],
        input: &[AssignedCell<C::Base, C::Base>],
    ) -> Result<(), Error> {
        let util_chip = self.chips.util_chip();
        let zero = util_chip.assign_constant(layouter, C::Base::ZERO)?;

        for (lhs, rhs) in input.iter().zip(initial_input.iter()) {
            let lhs = util_chip.select(layouter, is_base_case, lhs, &zero)?;
            let rhs = util_chip.select(layouter, is_base_case, rhs, &zero)?;
            util_chip.constrain_equal(layouter, &lhs, &rhs)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn check_state_hash(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        is_base_case: Option<&AssignedCell<C::Base, C::Base>>,
        h: &AssignedCell<C::Base, C::Base>,
        vp_digest: &AssignedCell<C::Base, C::Base>,
        step_idx: &AssignedCell<C::Base, C::Base>,
        initial_input: &[AssignedCell<C::Base, C::Base>],
        output: &[AssignedCell<C::Base, C::Base>],
        acc: &AssignedProtostarAccumulatorInstance<
            C,
            <Sc::Chips as Chips<C>>::EccChip,
            <Sc::Chips as Chips<C>>::ScalarChip,
        >,
    ) -> Result<(), Error> {
        let hash_chip = self.chips.hash_chip();
        let util_chip = self.chips.util_chip();
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
            let dummy_h = util_chip.assign_constant(layouter, Self::DUMMY_H)?;
            util_chip.select(layouter, is_base_case, &dummy_h, &rhs)?
        } else {
            rhs
        };
        util_chip.constrain_equal(layouter, lhs, &rhs)?;
        Ok(())
    }

    fn synthesize_folding(
        &self,
        mut layouter: impl Layouter<C::Base>,
        input: &[AssignedCell<C::Base, C::Base>],
        output: &[AssignedCell<C::Base, C::Base>],
    ) -> Result<(), Error> {
        let layouter = &mut layouter;

        let ecc_chip = self.chips.ecc_chip();
        let scalar_chip = self.chips.scalar_chip();
        let transcript_chip = self.chips.transcript_chip();
        let util_chip = self.chips.util_chip();

        let verifier = ProtostarAccumulationVerifier::new(
            ecc_chip.clone(),
            scalar_chip.clone(),
            self.avp.clone(),
        )?;

        let zero = util_chip.assign_constant(layouter, C::Base::ZERO)?;
        let one = util_chip.assign_constant(layouter, C::Base::ONE)?;
        let vp_digest = util_chip.assign_witness(layouter, Value::known(self.avp.vp_digest))?;
        let step_idx = util_chip.assign_witness(
            layouter,
            Value::known(C::Base::from(self.step_circuit.step_idx() as u64)),
        )?;
        let step_idx_plus_one = util_chip.add(layouter, &step_idx, &one)?;
        let initial_input = self
            .step_circuit
            .initial_input()
            .iter()
            .map(|value| util_chip.assign_witness(layouter, Value::known(*value)))
            .try_collect::<_, Vec<_>, _>()?;

        let is_base_case = util_chip.is_equal(layouter, &step_idx, &zero)?;
        let h_prime = util_chip.assign_witness(layouter, self.h_prime)?;

        self.check_initial_condition(layouter, &is_base_case, &initial_input, input)?;

        let acc = verifier.assign_accumulator(layouter, self.acc.as_ref())?;

        let (nark, acc_r_nark, acc_prime) = verifier.verify_accumulation_from_nark(
            layouter,
            &acc,
            [&self.incoming_instances[0], &self.incoming_instances[1]].map(Value::as_ref),
            &mut transcript_chip.init(self.incoming_proof.as_ref().map(Vec::as_slice)),
        )?;

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
            Some(&is_base_case),
            &h_from_incoming,
            &vp_digest,
            &step_idx,
            &initial_input,
            input,
            &acc,
        )?;
        self.check_state_hash(
            layouter,
            None,
            &h_prime,
            &vp_digest,
            &step_idx_plus_one,
            &initial_input,
            output,
            &acc_prime,
        )?;

        util_chip.constrain_instance(layouter, h_ohs_from_incoming.cell(), 0)?;
        util_chip.constrain_instance(layouter, h_prime.cell(), 1)?;

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
        self.synthesize_folding(layouter.namespace(|| ""), &input, &output)?;
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
            primary_hp: primary_circuit.circuit().chips.hash_chip().param(),
            secondary_vp,
            secondary_vp_digest,
            secondary_hp: secondary_circuit.circuit().chips.hash_chip().param(),
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
    Protostar::<HyperPlonk<P1>>::prove_decider(
        &ivc_pp.primary_pp,
        primary_acc,
        primary_transcript,
        &mut rng,
    )?;
    Protostar::<HyperPlonk<P2>>::prove_decider_with_last_nark(
        &ivc_pp.secondary_pp,
        secondary_acc,
        secondary_circuit,
        secondary_transcript,
        &mut rng,
    )?;
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
        ivc_vp.primary_hp.clone(),
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
        ivc_vp.secondary_hp.clone(),
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
mod test {
    use crate::{
        folding::protostar::{
            ivc::halo2::{
                preprocess, prove_decider, prove_steps, verify_decider,
                AssignedProtostarAccumulatorInstance, Chips, FieldInstruction, HashInstruction,
                NativeEccInstruction, StepCircuit, TranscriptInstruction, UtilInstruction,
            },
            ProtostarAccumulatorInstance,
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
                div_ceil, fe_from_bool, fe_from_le_bytes, fe_truncated, BitField, Bn254Grumpkin,
                Coordinates, CurveAffine, CurveCycle, Field, FromUniformBytes, PrimeField,
                PrimeFieldBits,
            },
            hash::Poseidon,
            test::seeded_std_rng,
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
                Keccak256Transcript, Transcript, TranscriptRead, TranscriptWrite,
            },
            DeserializeOwned, Itertools, Serialize,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use halo2_proofs::{
        circuit::{AssignedCell, Cell, Layouter, SimpleFloorPlanner, Value},
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
        poly::Rotation,
    };
    use std::{
        fmt::{self, Debug},
        hash::Hash,
        io::{self, Cursor, Read},
        iter,
        marker::PhantomData,
    };

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    const NUM_CHALLENGE_BITS: usize = 128;
    const NUM_CHALLENGE_BYTES: usize = NUM_CHALLENGE_BITS / 8;

    fn num_limbs<F: PrimeField>(num_limb_bits: usize) -> usize {
        div_ceil(F::NUM_BITS as usize, num_limb_bits)
    }

    fn fe_to_limbs<F1: PrimeFieldBits, F2: PrimeField>(fe: F1, num_limb_bits: usize) -> Vec<F2> {
        fe.to_le_bits()
            .chunks(num_limb_bits)
            .into_iter()
            .map(|bits| match bits.len() {
                1..=64 => F2::from(bits.load_le()),
                65..=128 => {
                    let lo = bits.load_le::<u64>();
                    let hi = bits[64..].load_le::<u64>();
                    F2::from(hi) * F2::from(2).pow_vartime([64]) + F2::from(lo)
                }
                _ => unimplemented!(),
            })
            .chain(iter::repeat(F2::ZERO))
            .take(num_limbs::<F1>(num_limb_bits))
            .collect()
    }

    #[derive(Debug)]
    pub struct PoseidonTranscript<F: PrimeField, S> {
        num_limb_bits: usize,
        state: Poseidon<F, T, RATE>,
        stream: S,
    }

    impl<F: FromUniformBytes<64>> InMemoryTranscript for PoseidonTranscript<F, Cursor<Vec<u8>>> {
        type Param = usize;

        fn new(num_limb_bits: usize) -> Self {
            Self {
                num_limb_bits,
                state: Poseidon::new(R_F, R_P),
                stream: Default::default(),
            }
        }

        fn into_proof(self) -> Vec<u8> {
            self.stream.into_inner()
        }

        fn from_proof(num_limb_bits: usize, proof: &[u8]) -> Self {
            Self {
                num_limb_bits,
                state: Poseidon::new(R_F, R_P),
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
            self.state.update(&fe_to_limbs(*fe, self.num_limb_bits));

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
            let x_y_is_identity = Option::<Coordinates<_>>::from(ec_point.coordinates())
                .map(|coords| [*coords.x(), *coords.y(), C::Base::ZERO])
                .unwrap_or_else(|| [C::Base::ZERO, C::Base::ZERO, C::Base::ONE]);
            self.state.update(&x_y_is_identity);
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

    #[derive(Clone, Debug)]
    struct DummyChip<C: CurveAffine> {
        num_hash_bits: usize,
        num_limb_bits: usize,
        advice: Column<Advice>,
        instance: Column<Instance>,
        poseidon: Poseidon<C::Base, T, RATE>,
        proof: Value<Cursor<Vec<u8>>>,
        _marker: PhantomData<C>,
    }

    impl<C: CurveAffine> DummyChip<C>
    where
        C::Base: FromUniformBytes<64>,
    {
        fn new(
            num_hash_bits: usize,
            num_limb_bits: usize,
            advice: Column<Advice>,
            instance: Column<Instance>,
        ) -> Self {
            DummyChip {
                num_hash_bits,
                num_limb_bits,
                advice,
                instance,
                poseidon: Poseidon::new(R_F, R_P),
                proof: Value::unknown(),
                _marker: PhantomData,
            }
        }
    }

    #[derive(Clone)]
    struct AssignedEcPoint<C: CurveAffine> {
        ec_point: Value<C>,
        x_y_is_identity: [AssignedCell<C::Base, C::Base>; 3],
    }

    impl<C: CurveAffine> AsRef<[AssignedCell<C::Base, C::Base>]> for AssignedEcPoint<C> {
        fn as_ref(&self) -> &[AssignedCell<C::Base, C::Base>] {
            &self.x_y_is_identity
        }
    }

    impl<C: CurveAffine> Debug for AssignedEcPoint<C> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedEcPoint");
            let mut value = None;
            self.ec_point.map(|ec_point| value = Some(ec_point));
            s.field("ec_point", &value).finish()
        }
    }

    impl<C: CurveAffine> NativeEccInstruction<C> for DummyChip<C> {
        type Assigned = AssignedEcPoint<C>;

        fn assign_constant(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            constant: C,
        ) -> Result<Self::Assigned, Error> {
            <Self as NativeEccInstruction<C>>::assign_witness(
                self,
                layouter,
                Value::known(constant),
            )
        }

        fn assign_witness(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            witness: Value<C>,
        ) -> Result<Self::Assigned, Error> {
            let x_y_is_identity = layouter.assign_region(
                || "",
                |mut region| {
                    Ok(witness
                        .map(|witness| {
                            Option::<Coordinates<_>>::from(witness.coordinates())
                                .map(|coords| [*coords.x(), *coords.y(), C::Base::ZERO])
                                .unwrap_or_else(|| [C::Base::ZERO, C::Base::ZERO, C::Base::ONE])
                        })
                        .transpose_array()
                        .into_iter()
                        .enumerate()
                        .map(|(offset, value)| {
                            region.assign_advice(|| "", self.advice, offset, || value)
                        })
                        .try_collect::<_, Vec<_>, _>()?
                        .try_into()
                        .unwrap())
                },
            )?;

            Ok(AssignedEcPoint {
                ec_point: witness,
                x_y_is_identity,
            })
        }

        fn select(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            condition: &AssignedCell<C::Base, C::Base>,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let mut out = when_false.clone();
            condition.value().map(|condition| {
                if condition == &C::Base::ONE {
                    out = when_true.clone();
                }
            });
            <Self as NativeEccInstruction<C>>::assign_witness(self, layouter, out.ec_point)
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
            let output = (lhs.ec_point + rhs.ec_point).map(Into::into);
            <Self as NativeEccInstruction<C>>::assign_witness(self, layouter, output)
        }

        fn mul(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            base: &Self::Assigned,
            le_bits: &[AssignedCell<C::Base, C::Base>],
        ) -> Result<Self::Assigned, Error> {
            let scalar = le_bits.iter().rev().map(AssignedCell::value).fold(
                Value::known(C::Scalar::ZERO),
                |acc, bit| {
                    acc.zip(bit).map(|(acc, bit)| {
                        acc.double()
                            + if *bit == C::Base::ONE {
                                C::Scalar::ONE
                            } else {
                                C::Scalar::ZERO
                            }
                    })
                },
            );
            let output = base
                .ec_point
                .zip(scalar)
                .map(|(base, scalar)| (base * scalar).into());
            <Self as NativeEccInstruction<C>>::assign_witness(self, layouter, output)
        }
    }

    #[derive(Clone)]
    struct AssignedScalar<F: Field, N: Field> {
        scalar: Value<F>,
        limbs: Vec<AssignedCell<N, N>>,
    }

    impl<F: Field, N: Field> AsRef<Self> for AssignedScalar<F, N> {
        fn as_ref(&self) -> &Self {
            self
        }
    }

    impl<F: Field, N: Field> AsRef<[AssignedCell<N, N>]> for AssignedScalar<F, N> {
        fn as_ref(&self) -> &[AssignedCell<N, N>] {
            &self.limbs
        }
    }

    impl<F: Field, N: Field> Debug for AssignedScalar<F, N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedScalar");
            let mut value = None;
            self.scalar.map(|scalar| value = Some(scalar));
            s.field("scalar", &value).finish()
        }
    }

    impl<C: CurveAffine> FieldInstruction<C::Scalar, C::Base> for DummyChip<C>
    where
        C::Scalar: PrimeFieldBits,
    {
        type Assigned = AssignedScalar<C::Scalar, C::Base>;

        fn assign_constant(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            constant: C::Scalar,
        ) -> Result<Self::Assigned, Error> {
            let limbs = layouter.assign_region(
                || "",
                |mut region| {
                    fe_to_limbs(constant, self.num_limb_bits)
                        .into_iter()
                        .enumerate()
                        .map(|(offset, limb)| {
                            region.assign_advice(|| "", self.advice, offset, || Value::known(limb))
                        })
                        .try_collect()
                },
            )?;
            Ok(AssignedScalar {
                scalar: Value::known(constant),
                limbs,
            })
        }

        fn assign_witness(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            witness: Value<C::Scalar>,
        ) -> Result<Self::Assigned, Error> {
            let limbs = layouter.assign_region(
                || "",
                |mut region| {
                    witness
                        .map(|fe| fe_to_limbs(fe, self.num_limb_bits))
                        .transpose_vec(num_limbs::<C::Scalar>(self.num_limb_bits))
                        .into_iter()
                        .enumerate()
                        .map(|(offset, limb)| {
                            region.assign_advice(|| "", self.advice, offset, || limb)
                        })
                        .try_collect()
                },
            )?;
            Ok(AssignedScalar {
                scalar: witness,
                limbs,
            })
        }

        fn fit_in_native(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            value: &Self::Assigned,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            let base = Value::known(C::Base::ONE.double().pow([self.num_limb_bits as u64]));
            let native = value
                .limbs
                .iter()
                .rev()
                .fold(Value::known(C::Base::ZERO), |acc, limb| {
                    acc * base + limb.value()
                });
            layouter.assign_region(
                || "",
                |mut region| region.assign_advice(|| "", self.advice, 0, || native),
            )
        }

        fn select(
            &self,
            _: &mut impl Layouter<C::Base>,
            condition: &AssignedCell<C::Base, C::Base>,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let mut out = when_false.clone();
            condition.value().map(|condition| {
                if condition == &C::Base::ONE {
                    out = when_true.clone();
                }
            });
            Ok(out)
        }

        fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C::Scalar) -> bool) {
            value.scalar.assert_if_known(f)
        }

        fn add(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let scalar = lhs.scalar + rhs.scalar;
            <Self as FieldInstruction<C::Scalar, C::Base>>::assign_witness(self, layouter, scalar)
        }

        fn mul(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let scalar = lhs.scalar * rhs.scalar;
            <Self as FieldInstruction<C::Scalar, C::Base>>::assign_witness(self, layouter, scalar)
        }
    }

    impl<C> TranscriptInstruction<C, Self, Self> for DummyChip<C>
    where
        C: CurveAffine,
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        type Challenge = <Self as FieldInstruction<C::Scalar, C::Base>>::Assigned;

        fn init(&self, proof: Value<&[u8]>) -> Self {
            let mut chip = self.clone();
            chip.proof = proof.map(|proof| Cursor::new(proof.to_vec()));
            chip
        }

        fn challenge_to_le_bits(
            &self,
            layouter: &mut impl Layouter<<C as CurveAffine>::Base>,
            scalar: &Self::Challenge,
        ) -> Result<Vec<AssignedCell<C::Base, C::Base>>, Error> {
            layouter.assign_region(
                || "",
                |mut region| {
                    scalar
                        .scalar
                        .map(|scalar| {
                            scalar
                                .to_le_bits()
                                .into_iter()
                                .take(NUM_CHALLENGE_BITS)
                                .collect_vec()
                        })
                        .transpose_vec(NUM_CHALLENGE_BITS)
                        .into_iter()
                        .enumerate()
                        .map(|(offset, bit)| {
                            region.assign_advice(
                                || "",
                                self.advice,
                                offset,
                                || bit.map(fe_from_bool),
                            )
                        })
                        .try_collect()
                },
            )
        }

        fn common_field_element(
            &mut self,
            value: &<Self as FieldInstruction<C::Scalar, C::Base>>::Assigned,
        ) -> Result<(), Error> {
            AsRef::<[_]>::as_ref(value).iter().for_each(|value| {
                value.value().map(|value| self.poseidon.update(&[*value]));
            });
            Ok(())
        }

        fn common_commitment(
            &mut self,
            value: &<Self as NativeEccInstruction<C>>::Assigned,
        ) -> Result<(), Error> {
            value.as_ref().iter().for_each(|value| {
                value.value().map(|value| self.poseidon.update(&[*value]));
            });
            Ok(())
        }

        fn read_field_element(
            &mut self,
            layouter: &mut impl Layouter<C::Base>,
        ) -> Result<<Self as FieldInstruction<C::Scalar, C::Base>>::Assigned, Error> {
            let fe = self.proof.as_mut().and_then(|proof| {
                let mut repr = <C::Scalar as PrimeField>::Repr::default();
                if proof.read_exact(repr.as_mut()).is_err() {
                    return Value::unknown();
                }
                C::Scalar::from_repr_vartime(repr)
                    .map(Value::known)
                    .unwrap_or_else(Value::unknown)
            });
            let fe = FieldInstruction::assign_witness(self, layouter, fe)?;
            self.common_field_element(&fe)?;
            Ok(fe)
        }

        fn read_commitment(
            &mut self,
            layouter: &mut impl Layouter<C::Base>,
        ) -> Result<<Self as NativeEccInstruction<C>>::Assigned, Error> {
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
            let comm = NativeEccInstruction::assign_witness(self, layouter, comm)?;
            self.common_commitment(&comm)?;
            Ok(comm)
        }

        fn squeeze_challenge(
            &mut self,
            layouter: &mut impl Layouter<C::Base>,
        ) -> Result<<Self as FieldInstruction<C::Scalar, C::Base>>::Assigned, Error> {
            let hash = self.poseidon.squeeze();
            self.poseidon.update(&[hash]);

            let challenge = fe_from_le_bytes(&hash.to_repr().as_ref()[..NUM_CHALLENGE_BYTES]);
            FieldInstruction::assign_witness(self, layouter, Value::known(challenge))
        }
    }

    impl<C: CurveAffine> HashInstruction<C> for DummyChip<C>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        type Param = (usize, usize, Poseidon<C::Base, T, RATE>);

        fn param(&self) -> Self::Param {
            (
                self.num_hash_bits,
                self.num_limb_bits,
                self.poseidon.clone(),
            )
        }

        fn hash_state<Comm: AsRef<C>>(
            (num_hash_bits, num_limb_bits, mut poseidon): Self::Param,
            vp_digest: C::Base,
            step_idx: usize,
            initial_input: &[C::Base],
            output: &[C::Base],
            acc: &ProtostarAccumulatorInstance<C::Scalar, Comm>,
        ) -> C::Base {
            let x_y_is_identity = |comm: &Comm| {
                Option::<Coordinates<_>>::from(comm.as_ref().coordinates())
                    .map(|coords| [*coords.x(), *coords.y(), C::Base::ZERO])
                    .unwrap_or_else(|| [C::Base::ZERO, C::Base::ZERO, C::Base::ONE])
            };
            let fe_to_limbs = |fe| fe_to_limbs(fe, num_limb_bits);
            let inputs = iter::empty()
                .chain([vp_digest, C::Base::from(step_idx as u64)])
                .chain(initial_input.iter().copied())
                .chain(output.iter().copied())
                .chain(fe_to_limbs(acc.instances[0][0]))
                .chain(fe_to_limbs(acc.instances[0][1]))
                .chain(acc.witness_comms.iter().flat_map(x_y_is_identity))
                .chain(acc.challenges.iter().copied().flat_map(fe_to_limbs))
                .chain(fe_to_limbs(acc.u))
                .chain(x_y_is_identity(&acc.e_comm))
                .chain(acc.compressed_e_sum.map(fe_to_limbs).into_iter().flatten())
                .collect_vec();
            poseidon.update(&inputs);
            fe_truncated(poseidon.squeeze(), num_hash_bits)
        }

        fn hash_assigned_state<EccChip, ScalarChip>(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            vp_digest: &AssignedCell<C::Base, C::Base>,
            step_idx: &AssignedCell<C::Base, C::Base>,
            initial_input: &[AssignedCell<C::Base, C::Base>],
            output: &[AssignedCell<C::Base, C::Base>],
            acc: &AssignedProtostarAccumulatorInstance<C, EccChip, ScalarChip>,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error>
        where
            EccChip: NativeEccInstruction<C>,
            ScalarChip: FieldInstruction<C::Scalar, C::Base>,
        {
            let mut poseidon = self.poseidon.clone();
            iter::empty()
                .chain([vp_digest, step_idx])
                .chain(initial_input)
                .chain(output)
                .chain(AsRef::as_ref(&acc.instances[0][0]))
                .chain(AsRef::as_ref(&acc.instances[0][1]))
                .chain(acc.witness_comms.iter().flat_map(|comm| comm.as_ref()))
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
                .for_each(|value| {
                    value.value().map(|value| poseidon.update(&[*value]));
                });
            let hash = fe_truncated(poseidon.squeeze(), self.num_hash_bits);
            layouter.assign_region(
                || "",
                |mut region| region.assign_advice(|| "", self.advice, 0, || Value::known(hash)),
            )
        }
    }

    impl<C: CurveAffine> UtilInstruction<C::Base> for DummyChip<C>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn assign_constant(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            witness: C::Base,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            UtilInstruction::<C::Base>::assign_witness(self, layouter, Value::known(witness))
        }

        fn assign_witness(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            witness: Value<C::Base>,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            layouter.assign_region(
                || "",
                |mut region| region.assign_advice(|| "", self.advice, 0, || witness),
            )
        }

        fn is_equal(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            lhs: &AssignedCell<C::Base, C::Base>,
            rhs: &AssignedCell<C::Base, C::Base>,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            let is_equal = lhs
                .value()
                .zip(rhs.value())
                .map(|(lhs, rhs)| lhs == rhs)
                .map(fe_from_bool);
            UtilInstruction::<C::Base>::assign_witness(self, layouter, is_equal)
        }

        fn select(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            condition: &AssignedCell<C::Base, C::Base>,
            when_true: &AssignedCell<C::Base, C::Base>,
            when_false: &AssignedCell<C::Base, C::Base>,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            condition.value().assert_if_known(|condition| {
                *condition == &C::Base::ZERO || *condition == &C::Base::ONE
            });
            let value = condition.value().copied() * when_true.value()
                + condition.value().map(|condition| C::Base::ONE - condition) * when_false.value();
            UtilInstruction::<C::Base>::assign_witness(self, layouter, value)
        }

        fn add(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            lhs: &AssignedCell<C::Base, C::Base>,
            rhs: &AssignedCell<C::Base, C::Base>,
        ) -> Result<AssignedCell<C::Base, C::Base>, Error> {
            let output = lhs.value().copied() + rhs.value();
            UtilInstruction::<C::Base>::assign_witness(self, layouter, output)
        }

        fn constrain_instance(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            cell: Cell,
            row: usize,
        ) -> Result<(), Error> {
            layouter.constrain_instance(cell, self.instance, row)
        }
    }

    impl<C: CurveAffine> Chips<C> for DummyChip<C>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        type EccChip = Self;
        type ScalarChip = Self;
        type TranscriptChip = Self;
        type HashChip = Self;
        type UtilChip = Self;

        fn ecc_chip(&self) -> &Self::EccChip {
            self
        }

        fn scalar_chip(&self) -> &Self::ScalarChip {
            self
        }

        fn transcript_chip(&self) -> &Self::TranscriptChip {
            self
        }

        fn hash_chip(&self) -> &Self::HashChip {
            self
        }

        fn util_chip(&self) -> &Self::UtilChip {
            self
        }
    }

    #[derive(Clone, Debug)]
    struct TrivialCircuit {
        num_hash_bits: usize,
        num_limb_bits: usize,
        step_idx: usize,
    }

    impl TrivialCircuit {
        fn new(num_hash_bits: usize, num_limb_bits: usize) -> Self {
            Self {
                num_hash_bits,
                num_limb_bits,
                step_idx: 0,
            }
        }
    }

    impl<F: Field> Circuit<F> for TrivialCircuit {
        type Config = (Column<Advice>, Column<Instance>);
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let selector = meta.selector();
            let advice = meta.advice_column();
            let instance = meta.instance_column();
            meta.create_gate("", |meta| {
                let selector = meta.query_selector(selector);
                let advice = meta.query_advice(advice, Rotation::cur());
                Some(selector * advice)
            });
            meta.enable_equality(advice);
            meta.enable_equality(instance);
            (advice, instance)
        }

        fn synthesize(&self, _: Self::Config, _: impl Layouter<F>) -> Result<(), Error> {
            Ok(())
        }
    }

    impl<F: Field> CircuitExt<F> for TrivialCircuit {
        fn instances(&self) -> Vec<Vec<F>> {
            Vec::new()
        }
    }

    impl<C: CurveAffine> StepCircuit<C> for TrivialCircuit
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        type Chips = DummyChip<C>;

        fn chips(&self, (advice, instance): Self::Config) -> Self::Chips {
            DummyChip::new(self.num_hash_bits, self.num_limb_bits, advice, instance)
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

    fn run_protostar_hyperplonk_ivc<C, P1, P2>(
        num_hash_bits: usize,
        num_limb_bits: usize,
        num_vars: usize,
        num_steps: usize,
    ) where
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
        let secondary_num_vars = num_vars;
        let primary_atp = num_limb_bits;
        let secondary_atp = num_limb_bits;

        let (mut primary_circuit, mut secondary_circuit, ivc_pp, ivc_vp) =
            preprocess::<C, P1, P2, _, _, PoseidonTranscript<_, _>, PoseidonTranscript<_, _>>(
                primary_num_vars,
                primary_atp,
                TrivialCircuit::new(num_hash_bits, num_limb_bits),
                secondary_num_vars,
                secondary_atp,
                TrivialCircuit::new(num_hash_bits, num_limb_bits),
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
        const NUM_HASH_BITS: usize = 250;
        const NUM_LIMB_BITS: usize = 64;
        const NUM_VARS: usize = 9;
        const NUM_STEPS: usize = 3;
        run_protostar_hyperplonk_ivc::<
            Bn254Grumpkin,
            MultilinearSimulator<UnivariateKzg<Bn256>>,
            MultilinearIpa<grumpkin::G1Affine>,
        >(NUM_HASH_BITS, NUM_LIMB_BITS, NUM_VARS, NUM_STEPS);
    }
}
