use crate::{
    backend::hyperplonk::folding::{protostar::ProtostarInstance, ProtostarVerifierParam},
    frontend::halo2::CircuitExt,
    pcs::PolynomialCommitmentScheme,
    util::{
        arithmetic::{fe_mod_from_le_bytes, fe_to_fe, CurveAffine, Field, PrimeField},
        hash::{Hash, Keccak256},
        izip_eq, Itertools, Serialize,
    },
};
use halo2_proofs::{
    circuit::{AssignedCell, Cell, Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use std::{fmt::Debug, iter, marker::PhantomData};

type AssignedProtostarInstance<C, EccChip, ScalarChip> = ProtostarInstance<
    <ScalarChip as FieldInstruction<
        <C as CurveAffine>::ScalarExt,
        <C as CurveAffine>::Base,
    >>::Assigned,
    <EccChip as NativeEccInstruction<C>>::Assigned,
>;

type AssignedCommittedPlonkish<C, EccChip, ScalarChip> = CommittedPlonkish<
    <ScalarChip as FieldInstruction<
        <C as CurveAffine>::ScalarExt,
        <C as CurveAffine>::Base,
    >>::Assigned,
    <EccChip as NativeEccInstruction<C>>::Assigned,
>;

type AssignedFoldingProof<C, EccChip, ScalarChip> = FoldingProof<
    <ScalarChip as FieldInstruction<
        <C as CurveAffine>::ScalarExt,
        <C as CurveAffine>::Base,
    >>::Assigned,
    <EccChip as NativeEccInstruction<C>>::Assigned,
>;

#[derive(Clone, Debug, Default)]
pub struct ProtostarFoldingVerifierParam<F> {
    vp_digest: F,
    num_instances: Vec<usize>,
    num_witness_polys: Vec<usize>,
    num_challenges: Vec<usize>,
    num_lookups: usize,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    num_compressed_cross_terms: usize,
}

impl<F, N, Pcs> From<&ProtostarVerifierParam<F, Pcs>> for ProtostarFoldingVerifierParam<N>
where
    F: PrimeField + Serialize,
    N: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    fn from(vp: &ProtostarVerifierParam<F, Pcs>) -> Self {
        let vp_digest = fe_mod_from_le_bytes(Keccak256::digest(bincode::serialize(vp).unwrap()));
        Self {
            vp_digest,
            num_instances: vp.vp.num_instances.clone(),
            num_witness_polys: vp.vp.num_witness_polys.clone(),
            num_challenges: vp.vp.num_challenges.clone(),
            num_lookups: vp.vp.num_lookups,
            num_theta_primes: vp.num_theta_primes,
            num_alpha_primes: vp.num_alpha_primes,
            num_folding_witness_polys: vp.num_folding_witness_polys,
            num_folding_challenges: vp.num_folding_challenges,
            num_compressed_cross_terms: vp.num_compressed_cross_terms,
        }
    }
}

impl<N: PrimeField> ProtostarFoldingVerifierParam<N> {
    pub fn dummy_h(&self) -> N {
        N::ZERO
    }

    pub fn dummy_protostar_instance<F: PrimeField, Comm: Default>(
        &self,
    ) -> ProtostarInstance<F, Comm> {
        ProtostarInstance::init(
            &self.num_instances,
            self.num_folding_witness_polys,
            self.num_folding_challenges,
        )
    }
}

#[derive(Clone, Debug)]
pub struct CommittedPlonkish<F, C> {
    instances: Vec<Vec<F>>,
    witness_comms: Vec<C>,
    challenges: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct FoldingProof<F, C> {
    compressed_cross_term_sums: Vec<F>,
    zeta_cross_term_comm: C,
}

pub struct ProtostarFoldingVerifier<C: CurveAffine, EccChip, ScalarChip, TranscriptChip> {
    ecc_chip: EccChip,
    scalar_chip: ScalarChip,
    transcript_chip: TranscriptChip,
    fvp: ProtostarFoldingVerifierParam<C::Base>,
    _marker: PhantomData<C>,
}

impl<C, EccChip, ScalarChip, TranscriptChip>
    ProtostarFoldingVerifier<C, EccChip, ScalarChip, TranscriptChip>
where
    C: CurveAffine,
    EccChip: NativeEccInstruction<C>,
    ScalarChip: FieldInstruction<C::Scalar, C::Base>,
    TranscriptChip: TranscriptInstruction<C, EccChip, ScalarChip>,
{
    pub fn new(
        ecc_chip: EccChip,
        scalar_chip: ScalarChip,
        transcript_chip: TranscriptChip,
        fvp: ProtostarFoldingVerifierParam<C::Base>,
    ) -> Result<Self, Error> {
        Ok(Self {
            ecc_chip,
            scalar_chip,
            transcript_chip,
            fvp,
            _marker: PhantomData,
        })
    }

    pub fn assign_default_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Base>,
    ) -> Result<AssignedProtostarInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let ProtostarFoldingVerifierParam {
            num_instances,
            num_folding_witness_polys,
            num_folding_challenges,
            ..
        } = &self.fvp;

        let instances = num_instances
            .iter()
            .map(|num_instances| {
                iter::repeat_with(|| scalar_chip.assign_constant(layouter, C::Scalar::ZERO))
                    .take(*num_instances)
                    .try_collect::<_, Vec<_>, _>()
            })
            .try_collect::<_, Vec<_>, _>()?;
        let witness_comms = iter::repeat_with(|| ecc_chip.assign_constant(layouter, C::identity()))
            .take(*num_folding_witness_polys)
            .try_collect::<_, Vec<_>, _>()?;
        let challenges =
            iter::repeat_with(|| scalar_chip.assign_constant(layouter, C::Scalar::ZERO))
                .take(*num_folding_challenges)
                .try_collect::<_, Vec<_>, _>()?;
        let u = scalar_chip.assign_constant(layouter, C::Scalar::ZERO)?;
        let compressed_e_sum = scalar_chip.assign_constant(layouter, C::Scalar::ZERO)?;
        let zeta_e_comm = ecc_chip.assign_constant(layouter, C::identity())?;
        Ok(ProtostarInstance {
            instances,
            witness_comms,
            challenges,
            u,
            compressed_e_sum,
            zeta_e_comm,
        })
    }

    pub fn assign_accumulator<Comm: AsRef<C>>(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        acc: Value<&ProtostarInstance<C::Scalar, Comm>>,
    ) -> Result<AssignedProtostarInstance<C, EccChip, ScalarChip>, Error> {
        let Self {
            ecc_chip,
            scalar_chip,
            ..
        } = self;
        let ProtostarFoldingVerifierParam {
            num_instances,
            num_folding_witness_polys,
            num_folding_challenges,
            ..
        } = &self.fvp;

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
            .transpose_vec(*num_folding_witness_polys)
            .into_iter()
            .map(|witness_comm| {
                ecc_chip.assign_witness(layouter, witness_comm.map(|comm| *comm.as_ref()))
            })
            .try_collect::<_, Vec<_>, _>()?;
        let challenges = acc
            .map(|acc| &acc.challenges)
            .transpose_vec(*num_folding_challenges)
            .into_iter()
            .map(|challenge| scalar_chip.assign_witness(layouter, challenge.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        let u = scalar_chip.assign_witness(layouter, acc.map(|acc| &acc.u).copied())?;
        let compressed_e_sum =
            scalar_chip.assign_witness(layouter, acc.map(|acc| &acc.compressed_e_sum).copied())?;
        let zeta_e_comm =
            ecc_chip.assign_witness(layouter, acc.map(|acc| *acc.zeta_e_comm.as_ref()))?;
        Ok(ProtostarInstance {
            instances,
            witness_comms,
            challenges,
            u,
            compressed_e_sum,
            zeta_e_comm,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn assign_incoming(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        incoming_instances: [Value<&C::Scalar>; 2],
        incoming_proof: Value<&[u8]>,
    ) -> Result<
        (
            AssignedCommittedPlonkish<C, EccChip, ScalarChip>,
            AssignedFoldingProof<C, EccChip, ScalarChip>,
            TranscriptChip::Challenge,
        ),
        Error,
    > {
        let Self { scalar_chip, .. } = self;
        let ProtostarFoldingVerifierParam {
            num_witness_polys,
            num_challenges,
            num_lookups,
            num_theta_primes,
            num_alpha_primes,
            num_compressed_cross_terms,
            ..
        } = &self.fvp;

        let mut transcript = self.transcript_chip.init(incoming_proof);

        let instances = incoming_instances
            .into_iter()
            .map(|instance| scalar_chip.assign_witness(layouter, instance.copied()))
            .try_collect::<_, Vec<_>, _>()?;
        for instance in instances.iter() {
            transcript.common_field_element(instance)?;
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in num_witness_polys.iter().zip_eq(num_challenges.iter()) {
            witness_comms.extend(transcript.read_commitments(layouter, *num_polys)?);
            challenges.extend(transcript.squeeze_challenges(layouter, *num_challenges)?);
        }

        // Round n

        let theta_prime = transcript.squeeze_challenge(layouter)?;
        let theta_primes = scalar_chip
            .powers(layouter, theta_prime.as_ref(), *num_theta_primes + 1)?
            .into_iter()
            .skip(1)
            .collect_vec();

        let lookup_m_comms = transcript.read_commitments(layouter, *num_lookups)?;

        // Round n+1

        let beta_prime = transcript.squeeze_challenge(layouter)?;

        let lookup_h_comms = transcript.read_commitments(layouter, 2 * num_lookups)?;

        // Round n+2

        let zeta = transcript.squeeze_challenge(layouter)?;

        let powers_of_zeta_comm = transcript.read_commitment(layouter)?;

        // Round n+3

        let alpha_prime = transcript.squeeze_challenge(layouter)?;
        let alpha_primes = scalar_chip
            .powers(layouter, alpha_prime.as_ref(), *num_alpha_primes + 1)?
            .into_iter()
            .skip(1)
            .collect_vec();

        let compressed_cross_term_sums =
            transcript.read_field_elements(layouter, *num_compressed_cross_terms)?;
        let zeta_cross_term_comm = transcript.read_commitment(layouter)?;

        // Round n+4

        let r = transcript.squeeze_challenge(layouter)?;

        let committed_plonkish = AssignedCommittedPlonkish::<_, EccChip, ScalarChip> {
            instances: vec![instances],
            witness_comms: iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms)
                .chain(Some(powers_of_zeta_comm))
                .collect(),
            challenges: iter::empty()
                .chain(challenges.iter().map(AsRef::as_ref).cloned())
                .chain(theta_primes)
                .chain(Some(beta_prime.as_ref().clone()))
                .chain(Some(zeta.as_ref().clone()))
                .chain(alpha_primes)
                .collect(),
        };
        let folding_proof = AssignedFoldingProof::<_, EccChip, ScalarChip> {
            compressed_cross_term_sums,
            zeta_cross_term_comm,
        };

        Ok((committed_plonkish, folding_proof, r))
    }

    #[allow(clippy::type_complexity)]
    pub fn fold(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        acc: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
        committed_plonkish: &AssignedCommittedPlonkish<C, EccChip, ScalarChip>,
        folding_proof: &AssignedFoldingProof<C, EccChip, ScalarChip>,
        r: &TranscriptChip::Challenge,
    ) -> Result<
        (
            AssignedProtostarInstance<C, EccChip, ScalarChip>,
            AssignedProtostarInstance<C, EccChip, ScalarChip>,
        ),
        Error,
    > {
        let Self {
            ecc_chip,
            scalar_chip,
            transcript_chip,
            ..
        } = self;
        let num_compressed_cross_terms = self.fvp.num_compressed_cross_terms;

        let r_le_bits = transcript_chip.challenge_to_le_bits(layouter, r)?;
        let r = r.as_ref();

        let incoming = {
            let powers_of_r = scalar_chip.powers(layouter, r, num_compressed_cross_terms + 1)?;

            let instances = committed_plonkish
                .instances
                .iter()
                .map(|instances| {
                    instances
                        .iter()
                        .map(|instance| scalar_chip.mul(layouter, r, instance))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = committed_plonkish
                .witness_comms
                .iter()
                .map(|comm| ecc_chip.mul(layouter, comm, &r_le_bits))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = committed_plonkish
                .challenges
                .iter()
                .map(|challenge| scalar_chip.mul(layouter, r, challenge))
                .try_collect::<_, Vec<_>, _>()?;
            let u = r.clone();
            let compressed_e_sum = scalar_chip.inner_product(
                layouter,
                &powers_of_r[1..],
                &folding_proof.compressed_cross_term_sums,
            )?;
            let zeta_e_comm =
                ecc_chip.mul(layouter, &folding_proof.zeta_cross_term_comm, &r_le_bits)?;

            ProtostarInstance {
                instances,
                witness_comms,
                challenges,
                u,
                compressed_e_sum,
                zeta_e_comm,
            }
        };

        let folded = {
            let instances = izip_eq!(&acc.instances, &incoming.instances)
                .map(|(lhs, rhs)| {
                    izip_eq!(lhs, rhs)
                        .map(|(lhs, rhs)| scalar_chip.add(layouter, lhs, rhs))
                        .try_collect::<_, Vec<_>, _>()
                })
                .try_collect::<_, Vec<_>, _>()?;
            let witness_comms = izip_eq!(&acc.witness_comms, &incoming.witness_comms)
                .map(|(lhs, rhs)| ecc_chip.add(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let challenges = izip_eq!(&acc.challenges, &incoming.challenges)
                .map(|(lhs, rhs)| scalar_chip.add(layouter, lhs, rhs))
                .try_collect::<_, Vec<_>, _>()?;
            let u = { scalar_chip.add(layouter, &acc.u, &incoming.u)? };
            let compressed_e_sum =
                scalar_chip.add(layouter, &acc.compressed_e_sum, &incoming.compressed_e_sum)?;
            let zeta_e_comm = ecc_chip.add(layouter, &acc.zeta_e_comm, &incoming.zeta_e_comm)?;

            ProtostarInstance {
                instances,
                witness_comms,
                challenges,
                u,
                compressed_e_sum,
                zeta_e_comm,
            }
        };

        Ok((incoming, folded))
    }

    #[allow(clippy::type_complexity)]
    pub fn assign_incoming_and_fold(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        acc: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
        incoming_instances: [Value<&C::Scalar>; 2],
        incoming_proof: Value<&[u8]>,
    ) -> Result<
        (
            AssignedCell<C::Base, C::Base>,
            AssignedCell<C::Base, C::Base>,
            AssignedProtostarInstance<C, EccChip, ScalarChip>,
            AssignedProtostarInstance<C, EccChip, ScalarChip>,
        ),
        Error,
    > {
        let (committed_plonkish, folding_proof, r) =
            self.assign_incoming(layouter, incoming_instances, incoming_proof)?;
        let h_from_incoming = self
            .scalar_chip
            .fit_in_native(layouter, &committed_plonkish.instances[0][0])?;
        let h_ohs_from_incoming = self
            .scalar_chip
            .fit_in_native(layouter, &committed_plonkish.instances[0][1])?;
        let (incoming, folded) =
            self.fold(layouter, acc, &committed_plonkish, &folding_proof, &r)?;
        Ok((h_from_incoming, h_ohs_from_incoming, incoming, folded))
    }

    fn select_accumulator(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        condition: &AssignedCell<C::Base, C::Base>,
        when_true: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
        when_false: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
    ) -> Result<AssignedProtostarInstance<C, EccChip, ScalarChip>, Error> {
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
        let compressed_e_sum = scalar_chip.select(
            layouter,
            condition,
            &when_true.compressed_e_sum,
            &when_false.compressed_e_sum,
        )?;
        let zeta_e_comm = ecc_chip.select(
            layouter,
            condition,
            &when_true.zeta_e_comm,
            &when_false.zeta_e_comm,
        )?;

        Ok(ProtostarInstance {
            instances,
            witness_comms,
            challenges,
            u,
            compressed_e_sum,
            zeta_e_comm,
        })
    }
}

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

    fn dummy_proof(fvp: &ProtostarFoldingVerifierParam<C::Base>) -> Vec<u8> {
        let g = C::generator().coordinates().unwrap();
        let g_x = g.x().to_repr();
        let g_y = g.y().to_repr();
        let zero = C::Scalar::ZERO.to_repr();
        iter::empty()
            .chain(
                iter::repeat_with(|| iter::empty().chain(g_x.as_ref()).chain(g_y.as_ref()))
                    .take(fvp.num_folding_witness_polys)
                    .flatten(),
            )
            .chain(
                iter::repeat(zero.as_ref())
                    .take(fvp.num_compressed_cross_terms)
                    .flatten(),
            )
            .chain(iter::empty().chain(g_x.as_ref()).chain(g_y.as_ref()))
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
}

pub trait HashInstruction<C: CurveAffine>: Clone + Debug {
    fn hash_state<Comm: AsRef<C>>(
        &self,
        vp_digest: C::Base,
        step_idx: usize,
        initial_input: &[C::Base],
        output: &[C::Base],
        acc: &ProtostarInstance<C::Scalar, Comm>,
    ) -> C::Base;

    fn hash_assigned_state<EccChip, ScalarChip>(
        &self,
        layouter: &mut impl Layouter<C::Base>,
        vp_digest: &AssignedCell<C::Base, C::Base>,
        step_idx: &AssignedCell<C::Base, C::Base>,
        initial_input: &[AssignedCell<C::Base, C::Base>],
        output: &[AssignedCell<C::Base, C::Base>],
        acc: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
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

#[derive(Debug, Default)]
pub struct RecursiveCircuit<C, Comm, Sc>
where
    C: CurveAffine,
    Sc: StepCircuit<C>,
{
    is_primary: bool,
    step_circuit: Sc,
    config: Sc::Config,
    chips: Sc::Chips,
    fvp: ProtostarFoldingVerifierParam<C::Base>,
    h_prime: Value<C::Base>,
    acc: Value<ProtostarInstance<C::Scalar, Comm>>,
    acc_prime: Value<ProtostarInstance<C::Scalar, Comm>>,
    incoming_instances: [Value<C::Scalar>; 2],
    incoming_proof: Value<Vec<u8>>,
}

impl<C, Comm, Sc> RecursiveCircuit<C, Comm, Sc>
where
    C: CurveAffine,
    Comm: AsRef<C>,
    Sc: StepCircuit<C>,
{
    pub fn new(
        is_primary: bool,
        fvp: ProtostarFoldingVerifierParam<C::Base>,
        step_circuit: Sc,
    ) -> Self
    where
        Comm: Default,
    {
        let config = Self::configure(&mut Default::default());
        let chips = step_circuit.chips(config.clone());
        let mut circuit = Self {
            is_primary,
            step_circuit,
            config,
            chips,
            fvp,
            h_prime: Value::unknown(),
            acc: Value::unknown(),
            acc_prime: Value::unknown(),
            incoming_instances: [Value::unknown(); 2],
            incoming_proof: Value::unknown(),
        };
        if circuit.fvp.vp_digest != C::Base::ZERO {
            assert_eq!(&circuit.fvp.num_instances, &[2]);
            circuit.update(
                circuit.fvp.dummy_protostar_instance(),
                circuit.fvp.dummy_protostar_instance(),
                [circuit.fvp.dummy_h(); 2].map(fe_to_fe),
                <Sc::Chips as Chips<_>>::TranscriptChip::dummy_proof(&circuit.fvp),
            );
        }
        circuit
    }

    pub fn update(
        &mut self,
        acc: ProtostarInstance<C::Scalar, Comm>,
        acc_prime: ProtostarInstance<C::Scalar, Comm>,
        incoming_instances: [C::Scalar; 2],
        incoming_proof: Vec<u8>,
    ) {
        if (self.is_primary && acc_prime.u != C::Scalar::ZERO)
            || (!self.is_primary && acc.u != C::Scalar::ZERO)
        {
            self.step_circuit.next();
        }
        self.h_prime = Value::known(self.chips.hash_chip().hash_state(
            self.fvp.vp_digest,
            self.step_circuit.step_idx() + 1,
            self.step_circuit.initial_input(),
            self.step_circuit.output(),
            &acc_prime,
        ));
        self.acc = Value::known(acc);
        self.acc_prime = Value::known(acc_prime);
        self.incoming_instances = incoming_instances.map(Value::known);
        self.incoming_proof = Value::known(incoming_proof);
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
        accumulator: &AssignedProtostarInstance<
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
            accumulator,
        )?;
        let rhs = if let Some(is_base_case) = is_base_case {
            let dummy_h = util_chip.assign_constant(layouter, self.fvp.dummy_h())?;
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
        let ecc_chip = self.chips.ecc_chip();
        let scalar_chip = self.chips.scalar_chip();
        let transcript_chip = self.chips.transcript_chip();
        let util_chip = self.chips.util_chip();

        let verifier = ProtostarFoldingVerifier::new(
            ecc_chip.clone(),
            scalar_chip.clone(),
            transcript_chip.clone(),
            self.fvp.clone(),
        )?;

        let zero = util_chip.assign_constant(&mut layouter, C::Base::ZERO)?;
        let one = util_chip.assign_constant(&mut layouter, C::Base::ONE)?;
        let vp_digest =
            util_chip.assign_witness(&mut layouter, Value::known(self.fvp.vp_digest))?;
        let step_idx = util_chip.assign_witness(
            &mut layouter,
            Value::known(C::Base::from(self.step_circuit.step_idx() as u64)),
        )?;
        let step_idx_plus_one = util_chip.add(&mut layouter, &step_idx, &one)?;
        let initial_input = self
            .step_circuit
            .initial_input()
            .iter()
            .map(|value| util_chip.assign_witness(&mut layouter, Value::known(*value)))
            .try_collect::<_, Vec<_>, _>()?;

        let is_base_case = util_chip.is_equal(&mut layouter, &step_idx, &zero)?;
        let h_prime = util_chip.assign_witness(&mut layouter, self.h_prime)?;

        self.check_initial_condition(&mut layouter, &is_base_case, &initial_input, input)?;

        let acc = verifier.assign_accumulator(&mut layouter, self.acc.as_ref())?;

        let (h_from_incoming, h_ohs_from_incoming, incoming, folded) = verifier
            .assign_incoming_and_fold(
                &mut layouter,
                &acc,
                [&self.incoming_instances[0], &self.incoming_instances[1]].map(Value::as_ref),
                self.incoming_proof.as_ref().map(Vec::as_slice),
            )?;

        let acc_prime = if self.is_primary {
            let acc_default = verifier.assign_default_accumulator(&mut layouter)?;
            verifier.select_accumulator(&mut layouter, &is_base_case, &acc_default, &folded)?
        } else {
            verifier.select_accumulator(&mut layouter, &is_base_case, &incoming, &folded)?
        };

        self.check_state_hash(
            &mut layouter,
            Some(&is_base_case),
            &h_from_incoming,
            &vp_digest,
            &step_idx,
            &initial_input,
            input,
            &acc,
        )?;
        self.check_state_hash(
            &mut layouter,
            None,
            &h_prime,
            &vp_digest,
            &step_idx_plus_one,
            &initial_input,
            output,
            &acc_prime,
        )?;

        util_chip.constrain_instance(&mut layouter, h_ohs_from_incoming.cell(), 0)?;
        util_chip.constrain_instance(&mut layouter, h_prime.cell(), 1)?;

        Ok(())
    }
}

impl<C, Comm, Sc> Circuit<C::Base> for RecursiveCircuit<C, Comm, Sc>
where
    C: CurveAffine,
    Comm: AsRef<C>,
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
            fvp: self.fvp.clone(),
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

impl<C, Comm, Sc> CircuitExt<C::Base> for RecursiveCircuit<C, Comm, Sc>
where
    C: CurveAffine,
    Comm: AsRef<C>,
    Sc: StepCircuit<C>,
{
    fn instances(&self) -> Vec<Vec<C::Base>> {
        let mut instances = vec![vec![self.fvp.dummy_h(); 2]];
        self.incoming_instances[1].map(|h_ohs| instances[0][0] = fe_to_fe(h_ohs));
        self.h_prime.map(|h_prime| instances[0][1] = h_prime);
        instances
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::{
                folding::protostar::{
                    verifier::{
                        halo2::{
                            AssignedProtostarInstance, Chips, FieldInstruction, HashInstruction,
                            NativeEccInstruction, ProtostarFoldingVerifierParam, RecursiveCircuit,
                            StepCircuit, TranscriptInstruction, UtilInstruction,
                        },
                        ProtostarInstance,
                    },
                    Protostar, ProtostarVerifierState,
                },
                HyperPlonk,
            },
            PlonkishBackend, PlonkishCircuit,
        },
        frontend::halo2::{CircuitExt, Halo2Circuit},
        pcs::{
            multilinear::{MultilinearIpa, MultilinearSimulator},
            univariate::UnivariateKzg,
            AdditiveCommitment, PolynomialCommitmentScheme,
        },
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::{
                div_ceil, fe_from_bool, fe_from_le_bytes, fe_to_fe, fe_truncated, BitField,
                Bn254Grumpkin, Coordinates, CurveAffine, CurveCycle, Field, FromUniformBytes,
                PrimeField, PrimeFieldBits,
            },
            end_timer,
            hash::Poseidon,
            start_timer,
            test::seeded_std_rng,
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, Transcript,
                TranscriptRead, TranscriptWrite,
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

    impl<F: FromUniformBytes<64>> PoseidonTranscript<F, Cursor<Vec<u8>>> {
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
        fn hash_state<Comm: AsRef<C>>(
            &self,
            vp_digest: C::Base,
            step_idx: usize,
            initial_input: &[C::Base],
            output: &[C::Base],
            acc: &ProtostarInstance<C::Scalar, Comm>,
        ) -> C::Base {
            let mut poseidon = self.poseidon.clone();
            let x_y_is_identity = |comm: &Comm| {
                Option::<Coordinates<_>>::from(comm.as_ref().coordinates())
                    .map(|coords| [*coords.x(), *coords.y(), C::Base::ZERO])
                    .unwrap_or_else(|| [C::Base::ZERO, C::Base::ZERO, C::Base::ONE])
            };
            let fe_to_limbs = |fe| fe_to_limbs(fe, self.num_limb_bits);
            let inputs = iter::empty()
                .chain([vp_digest, C::Base::from(step_idx as u64)])
                .chain(initial_input.iter().copied())
                .chain(output.iter().copied())
                .chain(fe_to_limbs(acc.instances[0][0]))
                .chain(fe_to_limbs(acc.instances[0][1]))
                .chain(acc.witness_comms.iter().flat_map(x_y_is_identity))
                .chain(acc.challenges.iter().copied().flat_map(fe_to_limbs))
                .chain(fe_to_limbs(acc.u))
                .chain(fe_to_limbs(acc.compressed_e_sum))
                .chain(x_y_is_identity(&acc.zeta_e_comm))
                .collect_vec();
            poseidon.update(&inputs);
            fe_truncated(poseidon.squeeze(), self.num_hash_bits)
        }

        fn hash_assigned_state<EccChip, ScalarChip>(
            &self,
            layouter: &mut impl Layouter<C::Base>,
            vp_digest: &AssignedCell<C::Base, C::Base>,
            step_idx: &AssignedCell<C::Base, C::Base>,
            initial_input: &[AssignedCell<C::Base, C::Base>],
            output: &[AssignedCell<C::Base, C::Base>],
            acc: &AssignedProtostarInstance<C, EccChip, ScalarChip>,
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
                .chain(AsRef::as_ref(&acc.compressed_e_sum))
                .chain(acc.zeta_e_comm.as_ref())
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

    fn run_protostar_folding_verifier<C, P1, P2>(
        num_hash_bits: usize,
        num_limb_bits: usize,
        num_vars: usize,
        num_steps: usize,
    ) where
        C: CurveCycle,
        C::Base: Ord + Hash + Serialize + DeserializeOwned,
        C::Scalar: Ord + Hash + Serialize + DeserializeOwned,
        P1: PolynomialCommitmentScheme<
            C::Scalar,
            Polynomial = MultilinearPolynomial<C::Scalar>,
            CommitmentChunk = C::Primary,
        >,
        P2: PolynomialCommitmentScheme<
            C::Base,
            Polynomial = MultilinearPolynomial<C::Base>,
            CommitmentChunk = C::Secondary,
        >,
        P1::Commitment: AdditiveCommitment<C::Scalar> + AsRef<C::Primary> + From<C::Primary>,
        P2::Commitment: AdditiveCommitment<C::Base> + AsRef<C::Secondary> + From<C::Secondary>,
    {
        let (mut primary, primary_pp, primary_vp, mut secondary, secondary_pp, secondary_vp) = {
            let timer = start_timer(|| format!("setup-primary-{num_vars}"));
            let primary_param = P1::setup(1 << num_vars, 0, seeded_std_rng()).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("setup-secondary-{num_vars}"));
            let secondary_param = P2::setup(1 << num_vars, 0, seeded_std_rng()).unwrap();
            end_timer(timer);

            let primary = RecursiveCircuit::<C::Secondary, P2::Commitment, _>::new(
                true,
                Default::default(),
                TrivialCircuit::new(num_hash_bits, num_limb_bits),
            );
            let primary = Halo2Circuit::new::<Protostar<HyperPlonk<P1>>>(num_vars, primary);
            let primary_info = primary.circuit_info_without_preprocess().unwrap();

            let timer = start_timer(|| format!("preprocess-primary-{num_vars}"));
            let (_, primary_vp) =
                Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_info).unwrap();
            end_timer(timer);

            let secondary = RecursiveCircuit::<C::Primary, P1::Commitment, _>::new(
                false,
                ProtostarFoldingVerifierParam::from(&primary_vp),
                TrivialCircuit::new(num_hash_bits, num_limb_bits),
            );
            let secondary = Halo2Circuit::new::<Protostar<HyperPlonk<P2>>>(num_vars, secondary);
            let secondary_info = secondary.circuit_info().unwrap();

            let timer = start_timer(|| format!("preprocess-secondary-{num_vars}"));
            let (secondary_pp, secondary_vp) =
                Protostar::<HyperPlonk<P2>>::preprocess(&secondary_param, &secondary_info).unwrap();
            end_timer(timer);

            let primary = RecursiveCircuit::<C::Secondary, P2::Commitment, _>::new(
                true,
                ProtostarFoldingVerifierParam::from(&secondary_vp),
                TrivialCircuit::new(num_hash_bits, num_limb_bits),
            );
            let primary = Halo2Circuit::new::<Protostar<HyperPlonk<P1>>>(num_vars, primary);
            let primary_info = primary.circuit_info().unwrap();

            let timer = start_timer(|| format!("preprocess-primary-{num_vars}"));
            let (primary_pp, primary_vp) =
                Protostar::<HyperPlonk<P1>>::preprocess(&primary_param, &primary_info).unwrap();
            end_timer(timer);

            (
                primary,
                primary_pp,
                primary_vp,
                secondary,
                secondary_pp,
                secondary_vp,
            )
        };

        let mut primary_prover_state = primary_pp.init();
        let mut secondary_prover_state = secondary_pp.init();

        let timer = start_timer(|| format!("fold-{}", num_steps - 1));
        for _ in 0..num_steps - 1 {
            {
                let primary_state = primary_prover_state.witness.instance.clone();
                let instnaces = primary.instance_slices();

                let timer = start_timer(|| format!("prove-primary-{num_vars}"));
                let proof = {
                    let mut transcript = PoseidonTranscript::new(num_limb_bits);
                    Protostar::<HyperPlonk<P1>>::prove(
                        &primary_pp,
                        &mut primary_prover_state,
                        &instnaces,
                        &primary,
                        &mut transcript,
                        seeded_std_rng(),
                    )
                    .unwrap();
                    transcript.into_proof()
                };
                end_timer(timer);

                secondary.update_witness(|circuit| {
                    circuit.update(
                        primary_state,
                        primary_prover_state.witness.instance.clone(),
                        [instnaces[0][0], instnaces[0][1]],
                        proof,
                    );
                });
            }
            {
                let secondary_state = secondary_prover_state.witness.instance.clone();
                let instnaces = secondary.instance_slices();

                let timer = start_timer(|| format!("prove-secondary-{num_vars}"));
                let proof = {
                    let mut transcript = PoseidonTranscript::new(num_limb_bits);
                    Protostar::<HyperPlonk<P2>>::prove(
                        &secondary_pp,
                        &mut secondary_prover_state,
                        &instnaces,
                        &secondary,
                        &mut transcript,
                        seeded_std_rng(),
                    )
                    .unwrap();
                    transcript.into_proof()
                };
                end_timer(timer);

                primary.update_witness(|circuit| {
                    circuit.update(
                        secondary_state,
                        secondary_prover_state.witness.instance.clone(),
                        [instnaces[0][0], instnaces[0][1]],
                        proof,
                    );
                });
            }
        }
        end_timer(timer);

        let timer = start_timer(|| "decide");
        let primary_state = {
            primary_prover_state.set_folding(false);
            let primary_verifier_state = ProtostarVerifierState::from(primary_prover_state.clone());
            let primary_state = primary_prover_state.witness.instance.clone();

            let instnaces = primary.instance_slices();

            let timer = start_timer(|| format!("prove-primary-{num_vars}"));
            let proof = {
                let mut transcript = PoseidonTranscript::new(num_limb_bits);
                Protostar::<HyperPlonk<P1>>::prove(
                    &primary_pp,
                    &mut primary_prover_state,
                    &instnaces,
                    &primary,
                    &mut transcript,
                    seeded_std_rng(),
                )
                .unwrap();
                transcript.into_proof()
            };
            end_timer(timer);

            let timer = start_timer(|| format!("verify-primary-{num_vars}"));
            let result = {
                let mut transcript =
                    PoseidonTranscript::from_proof(num_limb_bits, proof.as_slice());
                Protostar::<HyperPlonk<P1>>::verify(
                    &primary_vp,
                    primary_verifier_state,
                    &instnaces,
                    &mut transcript,
                    seeded_std_rng(),
                )
            };
            assert_eq!(result, Ok(()));
            end_timer(timer);

            secondary.update_witness(|circuit| {
                circuit.update(
                    primary_state,
                    primary_prover_state.witness.instance.clone(),
                    [instnaces[0][0], instnaces[0][1]],
                    proof,
                );
            });

            primary_prover_state.witness.instance
        };
        {
            secondary_prover_state.set_folding(false);
            let secondary_verifier_state =
                ProtostarVerifierState::from(secondary_prover_state.clone());

            let instnaces = secondary.instance_slices();

            let timer = start_timer(|| format!("prove-secondary-{num_vars}"));
            let proof = {
                let mut transcript = PoseidonTranscript::new(num_limb_bits);
                Protostar::<HyperPlonk<P2>>::prove(
                    &secondary_pp,
                    &mut secondary_prover_state,
                    &instnaces,
                    &secondary,
                    &mut transcript,
                    seeded_std_rng(),
                )
                .unwrap();
                transcript.into_proof()
            };
            end_timer(timer);

            assert_eq!(
                primary.circuit().chips.hash_chip().hash_state(
                    primary.circuit().fvp.vp_digest,
                    num_steps,
                    StepCircuit::<C::Secondary>::initial_input(&primary.circuit().step_circuit),
                    StepCircuit::<C::Secondary>::output(&primary.circuit().step_circuit),
                    &secondary_verifier_state.instance
                ),
                fe_to_fe(instnaces[0][0]),
            );
            assert_eq!(
                secondary.circuit().chips.hash_chip().hash_state(
                    secondary.circuit().fvp.vp_digest,
                    num_steps,
                    StepCircuit::<C::Primary>::initial_input(&secondary.circuit().step_circuit),
                    StepCircuit::<C::Primary>::output(&secondary.circuit().step_circuit),
                    &primary_state
                ),
                instnaces[0][1],
            );

            let timer = start_timer(|| format!("verify-secondary-{num_vars}"));
            let result = {
                let mut transcript =
                    PoseidonTranscript::from_proof(num_limb_bits, proof.as_slice());
                Protostar::<HyperPlonk<P2>>::verify(
                    &secondary_vp,
                    secondary_verifier_state,
                    &instnaces,
                    &mut transcript,
                    seeded_std_rng(),
                )
            };
            assert_eq!(result, Ok(()));
            end_timer(timer);
        }
        end_timer(timer);
    }

    #[test]
    fn kzg_protostar_folding_verifier() {
        const NUM_HASH_BITS: usize = 250;
        const NUM_LIMB_BITS: usize = 64;
        const NUM_VARS: usize = 9;
        const NUM_STEPS: usize = 3;
        run_protostar_folding_verifier::<
            Bn254Grumpkin,
            MultilinearSimulator<UnivariateKzg<Bn256>>,
            MultilinearIpa<grumpkin::G1Affine>,
        >(NUM_HASH_BITS, NUM_LIMB_BITS, NUM_VARS, NUM_STEPS);
    }
}
