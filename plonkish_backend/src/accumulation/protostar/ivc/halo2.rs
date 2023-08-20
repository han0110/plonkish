use crate::{
    accumulation::{
        protostar::{
            ivc::ProtostarAccumulationVerifierParam,
            PlonkishNarkInstance, Protostar, ProtostarAccumulator, ProtostarAccumulatorInstance,
            ProtostarProverParam,
            ProtostarStrategy::{Compressing, NoCompressing, CompressingWithSqrtPowers},
            ProtostarVerifierParam,
        },
        AccumulationScheme,
    },
    backend::{
        hyperplonk::{verifier::point_offset, HyperPlonk, HyperPlonkVerifierParam},
        PlonkishBackend, PlonkishCircuit,
    },
    frontend::halo2::{CircuitExt, Halo2Circuit},
    pcs::{
        multilinear::{
            Gemini, MultilinearHyrax, MultilinearHyraxParams, MultilinearIpa, MultilinearIpaParams,
        },
        univariate::{kzg::eval_sets, UnivariateKzg},
        AdditiveCommitment, Evaluation, PolynomialCommitmentScheme,
    },
    poly::multilinear::{
        rotation_eval_coeff_pattern, rotation_eval_point_pattern, zip_self, MultilinearPolynomial,
    },
    util::{
        arithmetic::{
            barycentric_weights, fe_to_fe, fe_truncated_from_le_bytes, powers, steps,
            BooleanHypercube, Field, MultiMillerLoop, PrimeCurveAffine, PrimeField, TwoChainCurve,
        },
        chain, end_timer,
        expression::{CommonPolynomial, Expression, Query, Rotation},
        hash::{Hash as _, Keccak256},
        izip, izip_eq, start_timer,
        transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
        BitIndex, DeserializeOwned, Itertools, Serialize,
    },
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use rand::RngCore;
use std::{
    borrow::{Borrow, BorrowMut, Cow},
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
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

    fn constrain_instance(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::Assigned,
        row: usize,
    ) -> Result<(), Error>;

    fn constrain_equal(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::Assigned,
        rhs: &Self::Assigned,
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

    fn sub(
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

    fn constrain_equal_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<(), Error>;

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

    fn to_repr_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::AssignedBase,
    ) -> Result<Vec<Self::Assigned>, Error>;

    fn add_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn sub_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn neg_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error> {
        let zero = self.assign_constant_base(layouter, C::Base::ZERO)?;
        self.sub_base(layouter, &zero, value)
    }

    fn sum_base<'a>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        values: impl IntoIterator<Item = &'a Self::AssignedBase>,
    ) -> Result<Self::AssignedBase, Error>
    where
        Self::AssignedBase: 'a,
    {
        values.into_iter().fold(
            self.assign_constant_base(layouter, C::Base::ZERO),
            |acc, value| self.add_base(layouter, &acc?, value),
        )
    }

    fn product_base<'a>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        values: impl IntoIterator<Item = &'a Self::AssignedBase>,
    ) -> Result<Self::AssignedBase, Error>
    where
        Self::AssignedBase: 'a,
    {
        values.into_iter().fold(
            self.assign_constant_base(layouter, C::Base::ONE),
            |acc, value| self.mul_base(layouter, &acc?, value),
        )
    }

    fn mul_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn div_incomplete_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedBase,
        rhs: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error>;

    fn invert_incomplete_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        value: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error> {
        let one = self.assign_constant_base(layouter, C::Base::ONE)?;
        self.div_incomplete_base(layouter, &one, value)
    }

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

    fn squares_base(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        base: &Self::AssignedBase,
        n: usize,
    ) -> Result<Vec<Self::AssignedBase>, Error> {
        Ok(match n {
            0 => Vec::new(),
            1 => vec![base.clone()],
            _ => {
                let mut squares = Vec::with_capacity(n);
                squares.push(base.clone());
                for _ in 0..n - 1 {
                    squares.push(self.mul_base(
                        layouter,
                        squares.last().unwrap(),
                        squares.last().unwrap(),
                    )?);
                }
                squares
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

    fn constrain_equal_secondary(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        lhs: &Self::AssignedSecondary,
        rhs: &Self::AssignedSecondary,
    ) -> Result<(), Error>;

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

    fn fixed_base_msm_secondary<'a, 'b>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: impl IntoIterator<Item = &'a C::Secondary>,
        scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
    ) -> Result<Self::AssignedSecondary, Error>
    where
        Self::AssignedBase: 'b;

    fn variable_base_msm_secondary<'a, 'b>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        bases: impl IntoIterator<Item = &'a Self::AssignedSecondary>,
        scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
    ) -> Result<Self::AssignedSecondary, Error>
    where
        Self::AssignedSecondary: 'a,
        Self::AssignedBase: 'b;
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
                CompressingWithSqrtPowers => uncompressed_comm_size + avp.num_cross_terms * scalar_size,
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
            CompressingWithSqrtPowers => Some(tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?),
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
            CompressingWithSqrtPowers => Some(
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
            CompressingWithSqrtPowers => Some(tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?),
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
            CompressingWithSqrtPowers => {
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
                CompressingWithSqrtPowers => {
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
            CompressingWithSqrtPowers => Some(tcc_chip.select_base(
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
            self.avp.vp_digest,
            self.step_circuit.step_idx() + 1,
            self.step_circuit.initial_input(),
            self.step_circuit.output(),
            &acc_prime,
        ));
        self.acc = Value::known(acc.unwrap_comm());
        self.acc_prime = Value::known(acc_prime.unwrap_comm());
        self.incoming_instances = incoming_instances.map(Value::known);
        self.incoming_proof = Value::known(incoming_proof);
    }

    fn init(&mut self, vp_digest: C::Scalar) {
        assert_eq!(&self.avp.num_instances, &[2]);
        self.avp.vp_digest = vp_digest;
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
        let vp_digest = tcc_chip.assign_witness(layouter, Value::known(avp.vp_digest))?;
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
    primary_hp: HP1,
    primary_arity: usize,
    secondary_vp: ProtostarVerifierParam<C::Base, HyperPlonk<P2>>,
    secondary_hp: HP2,
    secondary_arity: usize,
    _marker: PhantomData<C>,
}

impl<C, P1, P2, HP1, HP2> ProtostarIvcVerifierParam<C, P1, P2, HP1, HP2>
where
    C: TwoChainCurve,
    HyperPlonk<P1>: PlonkishBackend<C::Scalar>,
    HyperPlonk<P2>: PlonkishBackend<C::Base>,
{
    pub fn primary_arity(&self) -> usize {
        self.primary_arity
    }

    pub fn secondary_arity(&self) -> usize {
        self.secondary_arity
    }
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
    num_steps: usize,
    primary_initial_input: &[C::Scalar],
    primary_output: &[C::Scalar],
    primary_acc: &ProtostarAccumulatorInstance<C::Scalar, P1::Commitment>,
    primary_transcript: &mut impl TranscriptRead<P1::CommitmentChunk, C::Scalar>,
    secondary_initial_input: &[C::Base],
    secondary_output: &[C::Base],
    secondary_acc_before_last: impl BorrowMut<ProtostarAccumulatorInstance<C::Base, P2::Commitment>>,
    secondary_last_instances: &[Vec<C::Base>],
    secondary_transcript: &mut impl TranscriptRead<P2::CommitmentChunk, C::Base>,
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
        secondary_acc_before_last.borrow(),
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
        primary_acc,
    ) != secondary_last_instances[0][1]
    {
        return Err(crate::Error::InvalidSnark(
            "Invalid secondary state hash".to_string(),
        ));
    }

    Protostar::<HyperPlonk<P1>>::verify_decider(
        &ivc_vp.primary_vp,
        primary_acc,
        primary_transcript,
        &mut rng,
    )?;
    Protostar::<HyperPlonk<P2>>::verify_decider_with_last_nark(
        &ivc_vp.secondary_vp,
        secondary_acc_before_last,
        secondary_last_instances,
        secondary_transcript,
        &mut rng,
    )?;
    Ok(())
}

trait ProtostarHyperPlonkUtil<C: TwoChainCurve>: TwoChainCurveInstruction<C> {
    fn hornor(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        coeffs: &[Self::AssignedBase],
        x: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error> {
        let powers_of_x = self.powers_base(layouter, x, coeffs.len())?;
        self.inner_product_base(layouter, coeffs, &powers_of_x)
    }

    fn barycentric_weights(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        points: &[Self::AssignedBase],
    ) -> Result<Vec<Self::AssignedBase>, Error> {
        if points.len() == 1 {
            return Ok(vec![self.assign_constant_base(layouter, C::Base::ONE)?]);
        }
        points
            .iter()
            .enumerate()
            .map(|(j, point_j)| {
                let diffs = points
                    .iter()
                    .enumerate()
                    .filter_map(|(i, point_i)| {
                        (i != j).then(|| self.sub_base(layouter, point_j, point_i))
                    })
                    .try_collect::<_, Vec<_>, _>()?;
                let weight_inv = self.product_base(layouter, &diffs)?;
                self.invert_incomplete_base(layouter, &weight_inv)
            })
            .collect()
    }

    fn barycentric_interpolate(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        weights: &[Self::AssignedBase],
        points: &[Self::AssignedBase],
        evals: &[Self::AssignedBase],
        x: &Self::AssignedBase,
    ) -> Result<Self::AssignedBase, Error> {
        let (coeffs, sum_inv) = {
            let coeffs = izip_eq!(weights, points)
                .map(|(weight, point)| {
                    let coeff = self.sub_base(layouter, x, point)?;
                    self.div_incomplete_base(layouter, weight, &coeff)
                })
                .try_collect::<_, Vec<_>, _>()?;
            let sum = self.sum_base(layouter, &coeffs)?;
            let sum_inv = self.invert_incomplete_base(layouter, &sum)?;
            (coeffs, sum_inv)
        };
        let numer = self.inner_product_base(layouter, &coeffs, evals)?;
        self.mul_base(layouter, &numer, &sum_inv)
    }

    fn rotation_eval_points(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        x: &[Self::AssignedBase],
        one_minus_x: &[Self::AssignedBase],
        rotation: Rotation,
    ) -> Result<Vec<Vec<Self::AssignedBase>>, Error> {
        if rotation == Rotation::cur() {
            return Ok(vec![x.to_vec()]);
        }

        let zero = self.assign_constant_base(layouter, C::Base::ZERO)?;
        let one = self.assign_constant_base(layouter, C::Base::ONE)?;
        let distance = rotation.distance();
        let num_x = x.len() - distance;
        let points = if rotation < Rotation::cur() {
            let pattern = rotation_eval_point_pattern::<false>(x.len(), distance);
            let x = &x[distance..];
            let one_minus_x = &one_minus_x[distance..];
            pattern
                .iter()
                .map(|pat| {
                    iter::empty()
                        .chain((0..num_x).map(|idx| {
                            if pat.nth_bit(idx) {
                                &one_minus_x[idx]
                            } else {
                                &x[idx]
                            }
                        }))
                        .chain((0..distance).map(|idx| {
                            if pat.nth_bit(idx + num_x) {
                                &one
                            } else {
                                &zero
                            }
                        }))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec()
        } else {
            let pattern = rotation_eval_point_pattern::<true>(x.len(), distance);
            let x = &x[..num_x];
            let one_minus_x = &one_minus_x[..num_x];
            pattern
                .iter()
                .map(|pat| {
                    iter::empty()
                        .chain((0..distance).map(|idx| if pat.nth_bit(idx) { &one } else { &zero }))
                        .chain((0..num_x).map(|idx| {
                            if pat.nth_bit(idx + distance) {
                                &one_minus_x[idx]
                            } else {
                                &x[idx]
                            }
                        }))
                        .cloned()
                        .collect_vec()
                })
                .collect()
        };

        Ok(points)
    }

    fn rotation_eval(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        x: &[Self::AssignedBase],
        rotation: Rotation,
        evals_for_rotation: &[Self::AssignedBase],
    ) -> Result<Self::AssignedBase, Error> {
        if rotation == Rotation::cur() {
            assert!(evals_for_rotation.len() == 1);
            return Ok(evals_for_rotation[0].clone());
        }

        let num_vars = x.len();
        let distance = rotation.distance();
        assert!(evals_for_rotation.len() == 1 << distance);
        assert!(distance <= num_vars);

        let (pattern, nths, x) = if rotation < Rotation::cur() {
            (
                rotation_eval_coeff_pattern::<false>(num_vars, distance),
                (1..=distance).rev().collect_vec(),
                x[0..distance].iter().rev().collect_vec(),
            )
        } else {
            (
                rotation_eval_coeff_pattern::<true>(num_vars, distance),
                (num_vars - 1..).take(distance).collect(),
                x[num_vars - distance..].iter().collect(),
            )
        };
        x.into_iter()
            .zip(nths)
            .enumerate()
            .fold(
                Ok(Cow::Borrowed(evals_for_rotation)),
                |evals, (idx, (x_i, nth))| {
                    evals.and_then(|evals| {
                        pattern
                            .iter()
                            .step_by(1 << idx)
                            .map(|pat| pat.nth_bit(nth))
                            .zip(zip_self!(evals.iter()))
                            .map(|(bit, (mut eval_0, mut eval_1))| {
                                if bit {
                                    std::mem::swap(&mut eval_0, &mut eval_1);
                                }
                                let diff = self.sub_base(layouter, eval_1, eval_0)?;
                                let diff_x_i = self.mul_base(layouter, &diff, x_i)?;
                                self.add_base(layouter, &diff_x_i, eval_0)
                            })
                            .try_collect::<_, Vec<_>, _>()
                            .map(Into::into)
                    })
                },
            )
            .map(|evals| evals[0].clone())
    }

    fn eq_xy_coeffs(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        y: &[Self::AssignedBase],
    ) -> Result<Vec<Self::AssignedBase>, Error> {
        let mut evals = vec![self.assign_constant_base(layouter, C::Base::ONE)?];

        for y_i in y.iter().rev() {
            evals = evals
                .iter()
                .map(|eval| {
                    let hi = self.mul_base(layouter, eval, y_i)?;
                    let lo = self.sub_base(layouter, eval, &hi)?;
                    Ok([lo, hi])
                })
                .try_collect::<_, Vec<_>, Error>()?
                .into_iter()
                .flatten()
                .collect();
        }

        Ok(evals)
    }

    fn eq_xy_eval(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        x: &[Self::AssignedBase],
        y: &[Self::AssignedBase],
    ) -> Result<Self::AssignedBase, Error> {
        let terms = izip_eq!(x, y)
            .map(|(x, y)| {
                let one = self.assign_constant_base(layouter, C::Base::ONE)?;
                let xy = self.mul_base(layouter, x, y)?;
                let two_xy = self.add_base(layouter, &xy, &xy)?;
                let two_xy_plus_one = self.add_base(layouter, &two_xy, &one)?;
                let x_plus_y = self.add_base(layouter, x, y)?;
                self.sub_base(layouter, &two_xy_plus_one, &x_plus_y)
            })
            .try_collect::<_, Vec<_>, _>()?;
        self.product_base(layouter, &terms)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        expression: &Expression<C::Base>,
        identity_eval: &Self::AssignedBase,
        lagrange_evals: &BTreeMap<i32, Self::AssignedBase>,
        eq_xy_eval: &Self::AssignedBase,
        query_evals: &BTreeMap<Query, Self::AssignedBase>,
        challenges: &[Self::AssignedBase],
    ) -> Result<Self::AssignedBase, Error> {
        let mut evaluate = |expression| {
            self.evaluate(
                layouter,
                expression,
                identity_eval,
                lagrange_evals,
                eq_xy_eval,
                query_evals,
                challenges,
            )
        };
        match expression {
            Expression::Constant(scalar) => self.assign_constant_base(layouter, *scalar),
            Expression::CommonPolynomial(poly) => match poly {
                CommonPolynomial::Identity => Ok(identity_eval.clone()),
                CommonPolynomial::Lagrange(i) => Ok(lagrange_evals[i].clone()),
                CommonPolynomial::EqXY(idx) => {
                    assert_eq!(*idx, 0);
                    Ok(eq_xy_eval.clone())
                }
            },
            Expression::Polynomial(query) => Ok(query_evals[query].clone()),
            Expression::Challenge(index) => Ok(challenges[*index].clone()),
            Expression::Negated(a) => {
                let a = evaluate(a)?;
                self.neg_base(layouter, &a)
            }
            Expression::Sum(a, b) => {
                let a = evaluate(a)?;
                let b = evaluate(b)?;
                self.add_base(layouter, &a, &b)
            }
            Expression::Product(a, b) => {
                let a = evaluate(a)?;
                let b = evaluate(b)?;
                self.mul_base(layouter, &a, &b)
            }
            Expression::Scaled(a, scalar) => {
                let a = evaluate(a)?;
                let scalar = self.assign_constant_base(layouter, *scalar)?;
                self.mul_base(layouter, &a, &scalar)
            }
            Expression::DistributePowers(exprs, scalar) => {
                assert!(!exprs.is_empty());
                if exprs.len() == 1 {
                    return evaluate(&exprs[0]);
                }
                let scalar = evaluate(scalar)?;
                let exprs = exprs.iter().map(evaluate).try_collect::<_, Vec<_>, _>()?;
                let mut scalars = Vec::with_capacity(exprs.len());
                scalars.push(self.assign_constant_base(layouter, C::Base::ONE)?);
                scalars.push(scalar);
                for _ in 2..exprs.len() {
                    scalars.push(self.mul_base(layouter, &scalars[1], scalars.last().unwrap())?);
                }
                self.inner_product_base(layouter, &scalars, &exprs)
            }
        }
    }

    fn verify_sum_check<const IS_MSG_EVALS: bool>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_vars: usize,
        degree: usize,
        sum: &Self::AssignedBase,
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<(Self::AssignedBase, Vec<Self::AssignedBase>), Error> {
        let points = steps(C::Base::ZERO).take(degree + 1).collect_vec();
        let weights = barycentric_weights(&points)
            .into_iter()
            .map(|weight| self.assign_constant_base(layouter, weight))
            .try_collect::<_, Vec<_>, _>()?;
        let points = points
            .into_iter()
            .map(|point| self.assign_constant_base(layouter, point))
            .try_collect::<_, Vec<_>, _>()?;

        let mut sum = Cow::Borrowed(sum);
        let mut x = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let msg = transcript.read_field_elements(layouter, degree + 1)?;
            x.push(transcript.squeeze_challenge(layouter)?.as_ref().clone());

            let sum_from_evals = if IS_MSG_EVALS {
                self.add_base(layouter, &msg[0], &msg[1])?
            } else {
                self.sum_base(layouter, chain![[&msg[0], &msg[0]], &msg[1..]])?
            };
            self.constrain_equal_base(layouter, &sum, &sum_from_evals)?;

            if IS_MSG_EVALS {
                sum = Cow::Owned(self.barycentric_interpolate(
                    layouter,
                    &weights,
                    &points,
                    &msg,
                    x.last().unwrap(),
                )?);
            } else {
                sum = Cow::Owned(self.hornor(layouter, &msg, x.last().unwrap())?);
            }
        }

        Ok((sum.into_owned(), x))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn verify_sum_check_and_query(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_vars: usize,
        expression: &Expression<C::Base>,
        sum: &Self::AssignedBase,
        instances: &[Vec<Self::AssignedBase>],
        challenges: &[Self::AssignedBase],
        y: &[Self::AssignedBase],
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<
        (
            Vec<Vec<Self::AssignedBase>>,
            Vec<Evaluation<Self::AssignedBase>>,
        ),
        Error,
    > {
        let degree = expression.degree();

        let (x_eval, x) =
            self.verify_sum_check::<true>(layouter, num_vars, degree, sum, transcript)?;

        let pcs_query = {
            let mut used_query = expression.used_query();
            used_query.retain(|query| query.poly() >= instances.len());
            used_query
        };
        let (evals_for_rotation, query_evals) = pcs_query
            .iter()
            .map(|query| {
                let evals_for_rotation =
                    transcript.read_field_elements(layouter, 1 << query.rotation().distance())?;
                let eval = self.rotation_eval(
                    layouter,
                    x.as_ref(),
                    query.rotation(),
                    &evals_for_rotation,
                )?;
                Ok((evals_for_rotation, (*query, eval)))
            })
            .try_collect::<_, Vec<_>, Error>()?
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let one = self.assign_constant_base(layouter, C::Base::ONE)?;
        let one_minus_x = x
            .iter()
            .map(|x_i| self.sub_base(layouter, &one, x_i))
            .try_collect::<_, Vec<_>, _>()?;

        let (lagrange_evals, query_evals) = {
            let mut instance_query = expression.used_query();
            instance_query.retain(|query| query.poly() < instances.len());

            let lagranges = {
                let mut lagranges = instance_query.iter().fold(0..0, |range, query| {
                    let i = -query.rotation().0;
                    range.start.min(i)..range.end.max(i + instances[query.poly()].len() as i32)
                });
                if lagranges.start < 0 {
                    lagranges.start -= 1;
                }
                if lagranges.end > 0 {
                    lagranges.end += 1;
                }
                chain![lagranges, expression.used_langrange()].collect::<BTreeSet<_>>()
            };

            let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
            let lagrange_evals = lagranges
                .into_iter()
                .map(|i| {
                    let b = bh[i.rem_euclid(1 << num_vars as i32) as usize];
                    let eval = self.product_base(
                        layouter,
                        (0..num_vars).map(|idx| {
                            if b.nth_bit(idx) {
                                &x[idx]
                            } else {
                                &one_minus_x[idx]
                            }
                        }),
                    )?;
                    Ok((i, eval))
                })
                .try_collect::<_, BTreeMap<_, _>, Error>()?;

            let instance_evals = instance_query
                .into_iter()
                .map(|query| {
                    let is = if query.rotation() > Rotation::cur() {
                        (-query.rotation().0..0)
                            .chain(1..)
                            .take(instances[query.poly()].len())
                            .collect_vec()
                    } else {
                        (1 - query.rotation().0..)
                            .take(instances[query.poly()].len())
                            .collect_vec()
                    };
                    let eval = self.inner_product_base(
                        layouter,
                        &instances[query.poly()],
                        is.iter().map(|i| lagrange_evals.get(i).unwrap()),
                    )?;
                    Ok((query, eval))
                })
                .try_collect::<_, BTreeMap<_, _>, Error>()?;

            (
                lagrange_evals,
                chain![query_evals, instance_evals].collect(),
            )
        };
        let identity_eval = {
            let powers_of_two = powers(C::Base::ONE.double())
                .take(x.len())
                .map(|power_of_two| self.assign_constant_base(layouter, power_of_two))
                .try_collect::<_, Vec<_>, Error>()?;
            self.inner_product_base(layouter, &powers_of_two, &x)?
        };
        let eq_xy_eval = self.eq_xy_eval(layouter, &x, y)?;

        let eval = self.evaluate(
            layouter,
            expression,
            &identity_eval,
            &lagrange_evals,
            &eq_xy_eval,
            &query_evals,
            challenges,
        )?;

        self.constrain_equal_base(layouter, &x_eval, &eval)?;

        let points = pcs_query
            .iter()
            .map(Query::rotation)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|rotation| self.rotation_eval_points(layouter, &x, &one_minus_x, rotation))
            .try_collect::<_, Vec<_>, _>()?
            .into_iter()
            .flatten()
            .collect_vec();

        let point_offset = point_offset(&pcs_query);
        let evals = pcs_query
            .iter()
            .zip(evals_for_rotation)
            .flat_map(|(query, evals_for_rotation)| {
                (point_offset[&query.rotation()]..)
                    .zip(evals_for_rotation)
                    .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
            })
            .collect();
        Ok((points, evals))
    }

    #[allow(clippy::type_complexity)]
    fn multilinear_pcs_batch_verify<'a, Comm>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        comms: &'a [Comm],
        points: &[Vec<Self::AssignedBase>],
        evals: &[Evaluation<Self::AssignedBase>],
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<
        (
            Vec<(&'a Comm, Self::AssignedBase)>,
            Vec<Self::AssignedBase>,
            Self::AssignedBase,
        ),
        Error,
    > {
        let num_vars = points[0].len();

        let ell = evals.len().next_power_of_two().ilog2() as usize;
        let t = transcript
            .squeeze_challenges(layouter, ell)?
            .iter()
            .map(AsRef::as_ref)
            .cloned()
            .collect_vec();

        let eq_xt = self.eq_xy_coeffs(layouter, &t)?;
        let tilde_gs_sum = self.inner_product_base(
            layouter,
            &eq_xt[..evals.len()],
            evals.iter().map(Evaluation::value),
        )?;
        let (g_prime_eval, x) =
            self.verify_sum_check::<false>(layouter, num_vars, 2, &tilde_gs_sum, transcript)?;
        let eq_xy_evals = points
            .iter()
            .map(|point| self.eq_xy_eval(layouter, &x, point))
            .try_collect::<_, Vec<_>, _>()?;

        let g_prime_comm = {
            let scalars = evals.iter().zip(&eq_xt).fold(
                Ok::<_, Error>(BTreeMap::<_, _>::new()),
                |scalars, (eval, eq_xt_i)| {
                    let mut scalars = scalars?;
                    let scalar = self.mul_base(layouter, &eq_xy_evals[eval.point()], eq_xt_i)?;
                    match scalars.entry(eval.poly()) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() = self.add_base(layouter, entry.get(), &scalar)?;
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(scalar);
                        }
                    }
                    Ok(scalars)
                },
            )?;
            scalars
                .into_iter()
                .map(|(poly, scalar)| (&comms[poly], scalar))
                .collect_vec()
        };

        Ok((g_prime_comm, x, g_prime_eval))
    }

    fn verify_ipa<'a>(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        vp: &MultilinearIpaParams<C::Secondary>,
        comm: impl IntoIterator<Item = (&'a Self::AssignedSecondary, &'a Self::AssignedBase)>,
        point: &[Self::AssignedBase],
        eval: &Self::AssignedBase,
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<(), Error>
    where
        Self::AssignedSecondary: 'a,
        Self::AssignedBase: 'a,
    {
        let xi_0 = transcript.squeeze_challenge(layouter)?.as_ref().clone();

        let (ls, rs, xis) = iter::repeat_with(|| {
            Ok::<_, Error>((
                transcript.read_commitment(layouter)?,
                transcript.read_commitment(layouter)?,
                transcript.squeeze_challenge(layouter)?.as_ref().clone(),
            ))
        })
        .take(point.len())
        .try_collect::<_, Vec<_>, _>()?
        .into_iter()
        .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
        let g_k = transcript.read_commitment(layouter)?;
        let c = transcript.read_field_element(layouter)?;

        let xi_invs = xis
            .iter()
            .map(|xi| self.invert_incomplete_base(layouter, xi))
            .try_collect::<_, Vec<_>, _>()?;
        let eval_prime = self.mul_base(layouter, &xi_0, eval)?;

        let h_eval = {
            let one = self.assign_constant_base(layouter, C::Base::ONE)?;
            let terms = izip_eq!(point, xis.iter().rev())
                .map(|(point, xi)| {
                    let point_xi = self.mul_base(layouter, point, xi)?;
                    let neg_point = self.neg_base(layouter, point)?;
                    self.sum_base(layouter, [&one, &neg_point, &point_xi])
                })
                .try_collect::<_, Vec<_>, _>()?;
            self.product_base(layouter, &terms)?
        };
        let h_coeffs = {
            let one = self.assign_constant_base(layouter, C::Base::ONE)?;
            let mut coeff = vec![one];

            for xi in xis.iter().rev() {
                let extended = coeff
                    .iter()
                    .map(|coeff| self.mul_base(layouter, coeff, xi))
                    .try_collect::<_, Vec<_>, _>()?;
                coeff.extend(extended);
            }

            coeff
        };

        let neg_c = self.neg_base(layouter, &c)?;
        let h_scalar = {
            let mut tmp = self.mul_base(layouter, &neg_c, &h_eval)?;
            tmp = self.mul_base(layouter, &tmp, &xi_0)?;
            self.add_base(layouter, &tmp, &eval_prime)?
        };
        let identity = self.assign_constant_secondary(layouter, C::Secondary::identity())?;
        let out = {
            let h = self.assign_constant_secondary(layouter, *vp.h())?;
            let (mut bases, mut scalars) = comm.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            bases.extend(chain![&ls, &rs, [&h, &g_k]]);
            scalars.extend(chain![&xi_invs, &xis, [&h_scalar, &neg_c]]);
            self.variable_base_msm_secondary(layouter, bases, scalars)?
        };
        self.constrain_equal_secondary(layouter, &out, &identity)?;

        let out = {
            let bases = vp.g();
            let scalars = h_coeffs;
            self.fixed_base_msm_secondary(layouter, bases, &scalars)?
        };
        self.constrain_equal_secondary(layouter, &out, &g_k)?;

        Ok(())
    }

    fn verify_hyrax(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        vp: &MultilinearHyraxParams<C::Secondary>,
        comm: &[(&Vec<Self::AssignedSecondary>, Self::AssignedBase)],
        point: &[Self::AssignedBase],
        eval: &Self::AssignedBase,
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<(), Error> {
        let (lo, hi) = point.split_at(vp.row_num_vars());
        let scalars = self.eq_xy_coeffs(layouter, hi)?;

        let comm = comm
            .iter()
            .map(|(comm, rhs)| {
                let scalars = scalars
                    .iter()
                    .map(|lhs| self.mul_base(layouter, lhs, rhs))
                    .try_collect::<_, Vec<_>, _>()?;
                Ok::<_, Error>(izip_eq!(*comm, scalars))
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_iter()
            .flatten()
            .collect_vec();
        let comm = comm.iter().map(|(comm, scalar)| (*comm, scalar));

        self.verify_ipa(layouter, vp.ipa(), comm, lo, eval, transcript)
    }

    fn verify_hyrax_hyperplonk(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        vp: &HyperPlonkVerifierParam<C::Base, MultilinearHyrax<C::Secondary>>,
        instances: Value<&[C::Base]>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = Self>,
    ) -> Result<Vec<Self::AssignedBase>, Error>
    where
        C::Base: Serialize + DeserializeOwned,
        C::Secondary: Serialize + DeserializeOwned,
    {
        assert_eq!(vp.num_instances.len(), 1);
        let instances = vec![instances
            .transpose_vec(vp.num_instances[0])
            .into_iter()
            .map(|instance| self.assign_witness_base(layouter, instance.copied()))
            .try_collect::<_, Vec<_>, _>()?];

        transcript.common_field_elements(&instances[0])?;

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 3);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(
                iter::repeat_with(|| transcript.read_commitments(layouter, vp.pcs.num_chunks()))
                    .take(*num_polys)
                    .try_collect::<_, Vec<_>, _>()?,
            );
            challenges.extend(
                transcript
                    .squeeze_challenges(layouter, *num_challenges)?
                    .iter()
                    .map(AsRef::as_ref)
                    .cloned(),
            );
        }

        let beta = transcript.squeeze_challenge(layouter)?.as_ref().clone();

        let lookup_m_comms =
            iter::repeat_with(|| transcript.read_commitments(layouter, vp.pcs.num_chunks()))
                .take(vp.num_lookups)
                .try_collect::<_, Vec<_>, _>()?;

        let gamma = transcript.squeeze_challenge(layouter)?.as_ref().clone();

        let lookup_h_permutation_z_comms =
            iter::repeat_with(|| transcript.read_commitments(layouter, vp.pcs.num_chunks()))
                .take(vp.num_lookups + vp.num_permutation_z_polys)
                .try_collect::<_, Vec<_>, _>()?;

        let alpha = transcript.squeeze_challenge(layouter)?.as_ref().clone();
        let y = transcript
            .squeeze_challenges(layouter, vp.num_vars)?
            .iter()
            .map(AsRef::as_ref)
            .cloned()
            .collect_vec();

        challenges.extend([beta, gamma, alpha]);

        let zero = self.assign_constant_base(layouter, C::Base::ZERO)?;
        let (points, evals) = self.verify_sum_check_and_query(
            layouter,
            vp.num_vars,
            &vp.expression,
            &zero,
            &instances,
            &challenges,
            &y,
            transcript,
        )?;

        let dummy_comm = vec![
            self.assign_constant_secondary(layouter, C::Secondary::identity())?;
            vp.pcs.num_chunks()
        ];
        let preprocess_comms = vp
            .preprocess_comms
            .iter()
            .map(|comm| {
                comm.0
                    .iter()
                    .map(|c| self.assign_constant_secondary(layouter, *c))
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let permutation_comms = vp
            .permutation_comms
            .iter()
            .map(|comm| {
                comm.1
                     .0
                    .iter()
                    .map(|c| self.assign_constant_secondary(layouter, *c))
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let comms = iter::empty()
            .chain(iter::repeat(dummy_comm).take(vp.num_instances.len()))
            .chain(preprocess_comms)
            .chain(witness_comms)
            .chain(permutation_comms)
            .chain(lookup_m_comms)
            .chain(lookup_h_permutation_z_comms)
            .collect_vec();

        let (comm, point, eval) =
            self.multilinear_pcs_batch_verify(layouter, &comms, &points, &evals, transcript)?;

        self.verify_hyrax(layouter, &vp.pcs, &comm, &point, &eval, transcript)?;

        Ok(instances.into_iter().next().unwrap())
    }
}

impl<C, I> ProtostarHyperPlonkUtil<C> for I
where
    C: TwoChainCurve,
    I: TwoChainCurveInstruction<C>,
{
}

#[derive(Debug)]
pub struct ProtostarIvcAggregator<C, Pcs, TccChip, HashChip>
where
    C: TwoChainCurve,
    HyperPlonk<Pcs>: PlonkishBackend<C::Base>,
    HashChip: HashInstruction<C>,
{
    vp_digest: C::Scalar,
    vp: ProtostarVerifierParam<C::Base, HyperPlonk<Pcs>>,
    arity: usize,
    tcc_chip: TccChip,
    hash_chip: HashChip,
    _marker: PhantomData<(C, Pcs)>,
}

impl<C, Pcs, TccChip, HashChip> ProtostarIvcAggregator<C, Pcs, TccChip, HashChip>
where
    C: TwoChainCurve,
    Pcs: PolynomialCommitmentScheme<C::Base>,
    Pcs::Commitment: AsRef<C::Secondary>,
    HyperPlonk<Pcs>:
        PlonkishBackend<C::Base, VerifierParam = HyperPlonkVerifierParam<C::Base, Pcs>>,
    TccChip: TwoChainCurveInstruction<C>,
    HashChip: HashInstruction<C, TccChip = TccChip>,
{
    pub fn new(
        vp_digest: C::Scalar,
        vp: ProtostarVerifierParam<C::Base, HyperPlonk<Pcs>>,
        arity: usize,
        tcc_chip: TccChip,
        hash_chip: HashChip,
    ) -> Self {
        Self {
            vp_digest,
            vp,
            arity,
            tcc_chip,
            hash_chip,
            _marker: PhantomData,
        }
    }

    #[allow(clippy::type_complexity)]
    fn hash_state(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_steps: Value<usize>,
        initial_input: Value<Vec<C::Scalar>>,
        output: Value<Vec<C::Scalar>>,
        acc: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
    ) -> Result<
        (
            TccChip::Assigned,
            Vec<TccChip::Assigned>,
            Vec<TccChip::Assigned>,
            TccChip::Assigned,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let hash_chip = &self.hash_chip;

        let vp_digest = tcc_chip.assign_constant(layouter, self.vp_digest)?;
        let num_steps = tcc_chip.assign_witness(
            layouter,
            num_steps.map(|num_steps| C::Scalar::from(num_steps as u64)),
        )?;
        let initial_input = initial_input
            .transpose_vec(self.arity)
            .into_iter()
            .map(|input| tcc_chip.assign_witness(layouter, input))
            .try_collect::<_, Vec<_>, _>()?;
        let output = output
            .transpose_vec(self.arity)
            .into_iter()
            .map(|input| tcc_chip.assign_witness(layouter, input))
            .try_collect::<_, Vec<_>, _>()?;
        let h = hash_chip.hash_assigned_state(
            layouter,
            &vp_digest,
            &num_steps,
            &initial_input,
            &output,
            acc,
        )?;

        Ok((num_steps, initial_input, output, h))
    }

    #[allow(clippy::type_complexity)]
    fn reduce_decider_inner(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        acc: &AssignedProtostarAccumulatorInstance<
            TccChip::AssignedBase,
            TccChip::AssignedSecondary,
        >,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            Vec<TccChip::AssignedSecondary>,
            Vec<Vec<TccChip::AssignedBase>>,
            Vec<Evaluation<TccChip::AssignedBase>>,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let vp = &self.vp.vp;

        transcript.absorb_accumulator(acc)?;

        let beta = transcript.squeeze_challenge(layouter)?;
        let gamma = transcript.squeeze_challenge(layouter)?;

        let permutation_z_comms =
            transcript.read_commitments(layouter, vp.num_permutation_z_polys)?;

        let alpha = transcript.squeeze_challenge(layouter)?;
        let y = transcript.squeeze_challenges(layouter, vp.num_vars)?;

        let challenges = chain![
            &acc.challenges,
            [&acc.u],
            [&beta, &gamma, &alpha].map(AsRef::as_ref)
        ]
        .cloned()
        .collect_vec();
        let y = y.iter().map(AsRef::as_ref).cloned().collect_vec();

        let claimed_sum = if let Some(compressed_e_sum) = acc.compressed_e_sum.as_ref() {
            Cow::Borrowed(compressed_e_sum)
        } else {
            Cow::Owned(tcc_chip.assign_constant_base(layouter, C::Base::ZERO)?)
        };
        let (points, evals) = tcc_chip.verify_sum_check_and_query(
            layouter,
            self.vp.vp.num_vars,
            &self.vp.vp.expression,
            &claimed_sum,
            acc.instances(),
            &challenges,
            &y,
            transcript,
        )?;

        let builtin_witness_poly_offset = vp.num_witness_polys.iter().sum::<usize>();
        let dummy_comm = tcc_chip.assign_constant_secondary(layouter, C::Secondary::identity())?;
        let preprocess_comms = vp
            .preprocess_comms
            .iter()
            .map(|comm| tcc_chip.assign_constant_secondary(layouter, *comm.as_ref()))
            .try_collect::<_, Vec<_>, _>()?;
        let permutation_comms = vp
            .permutation_comms
            .iter()
            .map(|(_, comm)| tcc_chip.assign_constant_secondary(layouter, *comm.as_ref()))
            .try_collect::<_, Vec<_>, _>()?;
        let comms = chain![
            iter::repeat(&dummy_comm)
                .take(vp.num_instances.len())
                .cloned(),
            preprocess_comms,
            acc.witness_comms[..builtin_witness_poly_offset]
                .iter()
                .cloned(),
            permutation_comms,
            acc.witness_comms[builtin_witness_poly_offset..]
                .iter()
                .cloned(),
            permutation_z_comms,
            [acc.e_comm.clone()]
        ]
        .collect_vec();
        Ok((comms, points, evals))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn reduce_decider(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_steps: Value<usize>,
        initial_input: Value<Vec<C::Scalar>>,
        output: Value<Vec<C::Scalar>>,
        acc: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            TccChip::Assigned,
            Vec<TccChip::Assigned>,
            Vec<TccChip::Assigned>,
            TccChip::Assigned,
            Vec<TccChip::AssignedSecondary>,
            Vec<Vec<TccChip::AssignedBase>>,
            Vec<Evaluation<TccChip::AssignedBase>>,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let avp = ProtostarAccumulationVerifierParam::from(&self.vp);
        let acc_verifier = ProtostarAccumulationVerifier::new(avp, tcc_chip.clone());

        let acc = acc_verifier.assign_accumulator(layouter, acc.as_ref())?;

        let (num_steps, initial_input, output, h) =
            self.hash_state(layouter, num_steps, initial_input, output, &acc)?;

        let (comms, points, evals) = self.reduce_decider_inner(layouter, &acc, transcript)?;

        Ok((num_steps, initial_input, output, h, comms, points, evals))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn reduce_decider_with_last_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_steps: Value<usize>,
        initial_input: Value<Vec<C::Scalar>>,
        output: Value<Vec<C::Scalar>>,
        acc_before_last: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
        last_instance: Value<[C::Base; 2]>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            TccChip::Assigned,
            Vec<TccChip::Assigned>,
            Vec<TccChip::Assigned>,
            Vec<TccChip::AssignedSecondary>,
            Vec<Vec<TccChip::AssignedBase>>,
            Vec<Evaluation<TccChip::AssignedBase>>,
            TccChip::Assigned,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let avp = ProtostarAccumulationVerifierParam::from(&self.vp);
        let acc_verifier = ProtostarAccumulationVerifier::new(avp, tcc_chip.clone());

        let acc_before_last =
            acc_verifier.assign_accumulator(layouter, acc_before_last.as_ref())?;
        let (last_nark, _, acc) = {
            let instances = last_instance
                .as_ref()
                .map(|instances| [&instances[0], &instances[1]])
                .transpose_array();
            acc_verifier.verify_accumulation_from_nark(
                layouter,
                &acc_before_last,
                instances,
                transcript,
            )?
        };

        let (num_steps, initial_input, output, h) =
            self.hash_state(layouter, num_steps, initial_input, output, &acc_before_last)?;

        let h_from_last_nark = tcc_chip.fit_base_in_scalar(layouter, &last_nark.instances[0][0])?;
        let h_ohs_from_last_nark =
            tcc_chip.fit_base_in_scalar(layouter, &last_nark.instances[0][1])?;
        tcc_chip.constrain_equal(layouter, &h, &h_from_last_nark)?;

        let (comms, points, evals) = self.reduce_decider_inner(layouter, &acc, transcript)?;

        Ok((
            num_steps,
            initial_input,
            output,
            comms,
            points,
            evals,
            h_ohs_from_last_nark,
        ))
    }
}

impl<C, M, TccChip, HashChip> ProtostarIvcAggregator<C, Gemini<UnivariateKzg<M>>, TccChip, HashChip>
where
    C: TwoChainCurve,
    M: MultiMillerLoop<Scalar = C::Base, G1Affine = C::Secondary>,
    M::G1Affine: Serialize + DeserializeOwned,
    M::G2Affine: Serialize + DeserializeOwned,
    M::Scalar: Hash + Serialize + DeserializeOwned,
    TccChip: TwoChainCurveInstruction<C>,
    HashChip: HashInstruction<C, TccChip = TccChip>,
{
    #[allow(clippy::type_complexity)]
    pub fn aggregate_gemini_kzg_ivc(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_steps: Value<usize>,
        initial_input: Value<Vec<C::Scalar>>,
        output: Value<Vec<C::Scalar>>,
        acc: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            TccChip::Assigned,
            Vec<TccChip::Assigned>,
            Vec<TccChip::Assigned>,
            TccChip::Assigned,
            TccChip::AssignedSecondary,
            TccChip::AssignedSecondary,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let (num_steps, initial_input, output, h, comms, points, evals) =
            self.reduce_decider(layouter, num_steps, initial_input, output, acc, transcript)?;
        let (comm, point, eval) =
            tcc_chip.multilinear_pcs_batch_verify(layouter, &comms, &points, &evals, transcript)?;

        let (fs, points, evals) = {
            let num_vars = point.len();
            let fs = transcript.read_commitments(layouter, num_vars - 1)?;

            let beta = transcript.squeeze_challenge(layouter)?;
            let squares_of_beta = tcc_chip.squares_base(layouter, beta.as_ref(), num_vars)?;

            let evals = transcript.read_field_elements(layouter, num_vars)?;

            let one = tcc_chip.assign_constant_base(layouter, C::Base::ONE)?;
            let two = tcc_chip.assign_constant_base(layouter, C::Base::ONE.double())?;
            let eval_0 = evals.iter().zip(&squares_of_beta).zip(&point).rev().fold(
                Ok::<_, Error>(eval),
                |eval_pos, ((eval_neg, sqaure_of_beta), x_i)| {
                    let eval_pos = eval_pos?;
                    let mut tmp = tcc_chip.sub_base(layouter, &one, x_i)?;
                    tmp = tcc_chip.mul_base(layouter, &tmp, sqaure_of_beta)?;
                    let numer = {
                        let mut numer_lhs = tcc_chip.mul_base(layouter, &two, sqaure_of_beta)?;
                        numer_lhs = tcc_chip.mul_base(layouter, &numer_lhs, &eval_pos)?;
                        let mut numer_rhs = tcc_chip.sub_base(layouter, &tmp, x_i)?;
                        numer_rhs = tcc_chip.mul_base(layouter, &numer_rhs, eval_neg)?;
                        tcc_chip.sub_base(layouter, &numer_lhs, &numer_rhs)?
                    };
                    let denom = tcc_chip.add_base(layouter, &tmp, x_i)?;
                    tcc_chip.div_incomplete_base(layouter, &numer, &denom)
                },
            )?;

            let evals = chain!([(0, 0), (0, 1)], (1..num_vars).zip(2..))
                .zip(chain![[eval_0], evals])
                .map(|((idx, point), eval)| Evaluation::new(idx, point, eval))
                .collect_vec();
            let points = chain![
                [squares_of_beta[0].clone()],
                squares_of_beta
                    .iter()
                    .map(|sqaure_of_beta| tcc_chip.neg_base(layouter, sqaure_of_beta))
                    .try_collect::<_, Vec<_>, _>()?
            ]
            .collect_vec();

            (fs, points, evals)
        };

        let (sets, superset) = eval_sets(&evals);

        let beta = transcript.squeeze_challenge(layouter)?;
        let gamma = transcript.squeeze_challenge(layouter)?;

        let q = transcript.read_commitment(layouter)?;

        let z = transcript.squeeze_challenge(layouter)?;

        let max_set_len = sets.iter().map(|set| set.polys.len()).max().unwrap();
        let powers_of_beta = tcc_chip.powers_base(layouter, beta.as_ref(), max_set_len)?;
        let powers_of_gamma = tcc_chip.powers_base(layouter, gamma.as_ref(), sets.len())?;

        let vanishing_diff_evals = sets
            .iter()
            .map(|set| {
                let diffs = set
                    .diffs
                    .iter()
                    .map(|idx| tcc_chip.sub_base(layouter, z.as_ref(), &points[*idx]))
                    .try_collect::<_, Vec<_>, _>()?;
                tcc_chip.product_base(layouter, &diffs)
            })
            .try_collect::<_, Vec<_>, _>()?;
        let normalizer = tcc_chip.invert_incomplete_base(layouter, &vanishing_diff_evals[0])?;
        let normalized_scalars = izip_eq!(&powers_of_gamma, &vanishing_diff_evals)
            .map(|(power_of_gamma, vanishing_diff_eval)| {
                tcc_chip.product_base(layouter, [&normalizer, vanishing_diff_eval, power_of_gamma])
            })
            .try_collect::<_, Vec<_>, _>()?;
        let superset_eval = {
            let diffs = superset
                .iter()
                .map(|idx| tcc_chip.sub_base(layouter, z.as_ref(), &points[*idx]))
                .try_collect::<_, Vec<_>, _>()?;
            tcc_chip.product_base(layouter, &diffs)?
        };
        let q_scalar = {
            let neg_superset_eval = tcc_chip.neg_base(layouter, &superset_eval)?;
            tcc_chip.mul_base(layouter, &neg_superset_eval, &normalizer)?
        };
        let comm_scalars = sets.iter().zip(&normalized_scalars).fold(
            Ok::<_, Error>(vec![
                tcc_chip
                    .assign_constant_base(layouter, C::Base::ZERO)?;
                fs.len() + 1
            ]),
            |scalars, (set, coeff)| {
                let mut scalars = scalars?;
                for (poly, power_of_beta) in izip!(&set.polys, &powers_of_beta) {
                    let scalar = tcc_chip.mul_base(layouter, coeff, power_of_beta)?;
                    scalars[*poly] = tcc_chip.add_base(layouter, &scalars[*poly], &scalar)?;
                }
                Ok(scalars)
            },
        )?;
        let r_evals = sets
            .iter()
            .map(|set| {
                let points = set
                    .points
                    .iter()
                    .map(|idx| points[*idx].clone())
                    .collect_vec();
                let weights = tcc_chip.barycentric_weights(layouter, &points)?;
                let r_evals = set
                    .evals
                    .iter()
                    .map(|evals| {
                        tcc_chip.barycentric_interpolate(
                            layouter,
                            &weights,
                            &points,
                            evals,
                            z.as_ref(),
                        )
                    })
                    .try_collect::<_, Vec<_>, _>()?;
                tcc_chip.inner_product_base(layouter, &powers_of_beta[..r_evals.len()], &r_evals)
            })
            .try_collect::<_, Vec<_>, _>()?;
        let eval = tcc_chip.inner_product_base(layouter, &normalized_scalars, &r_evals)?;

        let pi = transcript.read_commitment(layouter)?;

        let pi_scalar = z.as_ref().clone();
        let g_scalar = tcc_chip.neg_base(layouter, &eval)?;

        let g = tcc_chip.assign_constant_secondary(layouter, self.vp.vp.pcs.g1())?;

        let (mut bases, mut scalars) = comm.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
        bases.extend(chain![&fs, [&q, &pi, &g]]);
        scalars.extend(chain![
            comm_scalars.into_iter().skip(1),
            [q_scalar, pi_scalar, g_scalar]
        ]);

        let lhs = tcc_chip.variable_base_msm_secondary(layouter, bases, &scalars)?;
        let rhs = pi;

        Ok((num_steps, initial_input, output, h, lhs, rhs))
    }
}

impl<C, TccChip, HashChip>
    ProtostarIvcAggregator<C, MultilinearIpa<C::Secondary>, TccChip, HashChip>
where
    C: TwoChainCurve,
    C::Secondary: Serialize + DeserializeOwned,
    C::Base: Hash + Serialize + DeserializeOwned,
    TccChip: TwoChainCurveInstruction<C>,
    HashChip: HashInstruction<C, TccChip = TccChip>,
{
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_ipa_grumpkin_ivc_with_last_nark(
        &self,
        layouter: &mut impl Layouter<C::Scalar>,
        num_steps: Value<usize>,
        initial_input: Value<Vec<C::Scalar>>,
        output: Value<Vec<C::Scalar>>,
        acc_before_last: Value<ProtostarAccumulatorInstance<C::Base, C::Secondary>>,
        last_instance: Value<[C::Base; 2]>,
        transcript: &mut impl TranscriptInstruction<C, TccChip = TccChip>,
    ) -> Result<
        (
            TccChip::Assigned,
            Vec<TccChip::Assigned>,
            Vec<TccChip::Assigned>,
            TccChip::Assigned,
        ),
        Error,
    > {
        let tcc_chip = &self.tcc_chip;
        let (num_steps, initial_input, output, comms, points, evals, h_ohs_from_last_nark) = self
            .reduce_decider_with_last_nark(
            layouter,
            num_steps,
            initial_input,
            output,
            acc_before_last,
            last_instance,
            transcript,
        )?;
        let (comm, point, eval) =
            tcc_chip.multilinear_pcs_batch_verify(layouter, &comms, &points, &evals, transcript)?;
        let comm = comm.iter().map(|(comm, scalar)| (*comm, scalar));

        tcc_chip.verify_ipa(layouter, &self.vp.vp.pcs, comm, &point, &eval, transcript)?;

        Ok((num_steps, initial_input, output, h_ohs_from_last_nark))
    }
}
