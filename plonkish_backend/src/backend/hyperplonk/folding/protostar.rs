use crate::{
    backend::{
        hyperplonk::{
            folding::protostar::{
                preprocessor::{batch_size, preprocess},
                prover::{
                    evaluate_compressed_cross_term_sum, evaluate_zeta_cross_term, lookup_h_polys,
                    powers_of_zeta_poly, ProtostarWitness,
                },
                verifier::ProtostarInstance,
            },
            prover::{
                instance_polys, lookup_compressed_polys, lookup_m_polys, permutation_z_polys,
                prove_sum_check,
            },
            verifier::verify_sum_check,
            HyperPlonk, HyperPlonkProverParam, HyperPlonkVerifierParam,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        end_timer,
        expression::Expression,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::BorrowMut, hash::Hash, iter, marker::PhantomData};

mod preprocessor;
mod prover;
mod verifier;

#[derive(Clone, Debug)]
pub struct Protostar<Pb>(PhantomData<Pb>);

#[derive(Debug)]
pub struct ProtostarProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pp: HyperPlonkProverParam<F, Pcs>,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_wintess_polys: usize,
    num_folding_challenges: usize,
    compressed_cross_term_expressions: Vec<Expression<F>>,
    zeta_cross_term_expression: Expression<F>,
    sum_check_expression: Expression<F>,
}

impl<F, Pcs> ProtostarProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn init(&self) -> ProtostarProverState<F, Pcs> {
        ProtostarProverState {
            is_folding: true,
            witness: ProtostarWitness::init(
                self.pp.num_vars,
                &self.pp.num_instances,
                self.num_folding_wintess_polys,
                self.num_folding_challenges,
            ),
        }
    }
}

#[derive(Debug)]
pub struct ProtostarVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    vp: HyperPlonkVerifierParam<F, Pcs>,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_wintess_polys: usize,
    num_folding_challenges: usize,
    num_compressed_cross_terms: usize,
    sum_check_expression: Expression<F>,
}

impl<F, Pcs> ProtostarVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn init(&self) -> ProtostarVerifierState<F, Pcs> {
        ProtostarVerifierState {
            is_folding: true,
            instance: ProtostarInstance::init(
                &self.vp.num_instances,
                self.num_folding_wintess_polys,
                self.num_folding_challenges,
            ),
        }
    }
}
#[derive(Debug)]
pub struct ProtostarProverState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    is_folding: bool,
    witness: ProtostarWitness<F, Pcs::Commitment, Pcs::Polynomial>,
}

impl<F, Pcs> ProtostarProverState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn set_folding(&mut self, is_folding: bool) {
        self.is_folding = is_folding;
    }
}

#[derive(Debug)]
pub struct ProtostarVerifierState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    is_folding: bool,
    instance: ProtostarInstance<F, Pcs::Commitment>,
}

impl<F, Pcs> ProtostarVerifierState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn set_folding(&mut self, is_folding: bool) {
        self.is_folding = is_folding;
    }
}

impl<F, Pcs> PlonkishBackend<F, Pcs> for Protostar<HyperPlonk<Pcs>>
where
    F: PrimeField + Ord + Hash,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    Pcs::Commitment: AdditiveCommitment<F>,
    Pcs::CommitmentChunk: AdditiveCommitment<F>,
{
    type ProverParam = ProtostarProverParam<F, Pcs>;
    type VerifierParam = ProtostarVerifierParam<F, Pcs>;
    type ProverState = ProtostarProverState<F, Pcs>;
    type VerifierState = ProtostarVerifierState<F, Pcs>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(circuit_info.is_well_formed());

        preprocess(param, circuit_info)
    }

    fn prove(
        pp: &Self::ProverParam,
        mut state: impl BorrowMut<Self::ProverState>,
        instances: &[&[F]],
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarProverParam {
            pp,
            num_theta_primes,
            num_alpha_primes,
            compressed_cross_term_expressions,
            zeta_cross_term_expression,
            sum_check_expression,
            ..
        } = pp;
        let state = state.borrow_mut();

        for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>());
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n

        let theta_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_theta_primes)
            .collect_vec();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let instance_polys = instance_polys(pp.num_vars, instances.iter().cloned());
            let polys = iter::empty()
                .chain(instance_polys.iter())
                .chain(pp.preprocess_polys.iter())
                .chain(witness_polys.iter())
                .collect_vec();
            let thetas = iter::empty()
                .chain(Some(F::ONE))
                .chain(theta_primes.iter().cloned())
                .collect_vec();
            lookup_compressed_polys(&pp.lookups, &polys, &challenges, &thetas)
        };
        end_timer(timer);

        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1

        let beta_prime = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &beta_prime);
        end_timer(timer);

        let lookup_h_comms = {
            let polys = lookup_h_polys.iter().flatten();
            Pcs::batch_commit_and_write(&pp.pcs, polys, transcript)?
        };

        // Round n+2

        let zeta = transcript.squeeze_challenge();

        let powers_of_zeta_poly = powers_of_zeta_poly(pp.num_vars, zeta);
        let powers_of_zeta_comm = Pcs::commit_and_write(&pp.pcs, &powers_of_zeta_poly, transcript)?;

        // Round n+3

        let alpha_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_alpha_primes)
            .collect_vec();

        let incoming = ProtostarWitness::from_committed(
            pp.num_vars,
            instances,
            iter::empty()
                .chain(witness_polys)
                .chain(lookup_m_polys)
                .chain(lookup_h_polys.into_iter().flatten())
                .chain(Some(powers_of_zeta_poly)),
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms)
                .chain(Some(powers_of_zeta_comm)),
            iter::empty()
                .chain(challenges)
                .chain(theta_primes)
                .chain(Some(beta_prime))
                .chain(Some(zeta))
                .chain(alpha_primes)
                .collect(),
        );

        let timer = start_timer(|| {
            let len = compressed_cross_term_expressions.len();
            format!("compressed_cross_term_polys-{len}",)
        });
        let compressed_cross_term_sums = evaluate_compressed_cross_term_sum(
            compressed_cross_term_expressions,
            pp.num_vars,
            &pp.preprocess_polys,
            &state.witness,
            &incoming,
        );
        end_timer(timer);

        let timer = start_timer(|| "zeta_cross_term_polys");
        let zeta_cross_term_poly = evaluate_zeta_cross_term(
            zeta_cross_term_expression,
            pp.num_vars,
            &pp.preprocess_polys,
            &state.witness,
            &incoming,
        );
        end_timer(timer);

        transcript.write_field_elements(&compressed_cross_term_sums)?;
        let zeta_cross_term_comm =
            Pcs::commit_and_write(&pp.pcs, &zeta_cross_term_poly, transcript)?;

        // Round n+4

        let r = transcript.squeeze_challenge();

        let timer = start_timer(|| "fold");
        state.witness.fold(
            &incoming,
            &compressed_cross_term_sums,
            &zeta_cross_term_poly,
            &zeta_cross_term_comm,
            &r,
        );
        end_timer(timer);

        if !state.is_folding {
            let beta = transcript.squeeze_challenge();
            let gamma = transcript.squeeze_challenge();

            let timer =
                start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
            let builtin_witness_poly_offset = pp.num_witness_polys.iter().sum::<usize>();
            let instance_polys = instance_polys(pp.num_vars, &state.witness.instance.instances);
            let polys = iter::empty()
                .chain(&instance_polys)
                .chain(&pp.preprocess_polys)
                .chain(&state.witness.witness_polys[..builtin_witness_poly_offset])
                .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
                .collect_vec();
            let permutation_z_polys = permutation_z_polys(
                pp.num_permutation_z_polys,
                &pp.permutation_polys,
                &polys,
                &beta,
                &gamma,
            );
            end_timer(timer);

            let permutation_z_comms =
                Pcs::batch_commit_and_write(&pp.pcs, &permutation_z_polys, transcript)?;

            // Round n+5

            let alpha = transcript.squeeze_challenge();
            let y = transcript.squeeze_challenges(pp.num_vars);

            let polys = iter::empty()
                .chain(polys)
                .chain(&state.witness.witness_polys[builtin_witness_poly_offset..])
                .chain(permutation_z_polys.iter())
                .chain(Some(&state.witness.zeta_e_poly))
                .collect_vec();
            let challenges = iter::empty()
                .chain(state.witness.instance.challenges.iter().copied())
                .chain([beta, gamma, alpha, state.witness.instance.u])
                .collect();
            let (points, evals) = {
                prove_sum_check(
                    pp.num_instances.len(),
                    sum_check_expression,
                    state.witness.instance.compressed_e_sum,
                    &polys,
                    challenges,
                    y,
                    transcript,
                )?
            };

            // PCS open

            let dummy_comm = Pcs::Commitment::default();
            let comms = iter::empty()
                .chain(iter::repeat(&dummy_comm).take(pp.num_instances.len()))
                .chain(&pp.preprocess_comms)
                .chain(&state.witness.instance.witness_comms[..builtin_witness_poly_offset])
                .chain(&pp.permutation_comms)
                .chain(&state.witness.instance.witness_comms[builtin_witness_poly_offset..])
                .chain(&permutation_z_comms)
                .chain(Some(&zeta_cross_term_comm))
                .collect_vec();
            let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
            Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
            end_timer(timer);
        }

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        mut state: impl BorrowMut<Self::VerifierState>,
        instances: &[&[F]],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let ProtostarVerifierParam {
            vp,
            num_theta_primes,
            num_alpha_primes,
            num_compressed_cross_terms,
            sum_check_expression,
            ..
        } = vp;
        let state = state.borrow_mut();

        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_folding_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_folding_challenges));
        }

        // Round n

        let theta_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_theta_primes)
            .collect_vec();

        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let beta_prime = transcript.squeeze_challenge();

        let lookup_h_comms = Pcs::read_commitments(&vp.pcs, 2 * vp.num_lookups, transcript)?;

        // Round n+2

        let zeta = transcript.squeeze_challenge();

        let powers_of_zeta_comm = Pcs::read_commitment(&vp.pcs, transcript)?;

        // Round n+3

        let alpha_primes = powers(transcript.squeeze_challenge())
            .skip(1)
            .take(*num_alpha_primes)
            .collect_vec();

        let incoming = ProtostarInstance::from_committed(
            instances,
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms)
                .chain(Some(powers_of_zeta_comm)),
            iter::empty()
                .chain(challenges)
                .chain(theta_primes)
                .chain(Some(beta_prime))
                .chain(Some(zeta))
                .chain(alpha_primes)
                .collect(),
        );

        let compressed_cross_term_sums =
            transcript.read_field_elements(*num_compressed_cross_terms)?;
        let zeta_cross_term_comm = Pcs::read_commitment(&vp.pcs, transcript)?;

        // Round n+4

        let r = transcript.squeeze_challenge();

        state.instance.fold(
            &incoming,
            &compressed_cross_term_sums,
            &zeta_cross_term_comm,
            &r,
        );

        if !state.is_folding {
            let beta = transcript.squeeze_challenge();
            let gamma = transcript.squeeze_challenge();

            let permutation_z_comms =
                Pcs::read_commitments(&vp.pcs, vp.num_permutation_z_polys, transcript)?;

            // Round n+5

            let alpha = transcript.squeeze_challenge();
            let y = transcript.squeeze_challenges(vp.num_vars);

            let instances = state.instance.instance_slices();
            let challenges = iter::empty()
                .chain(state.instance.challenges.iter().copied())
                .chain([beta, gamma, alpha, state.instance.u])
                .collect_vec();
            let (points, evals) = {
                verify_sum_check(
                    vp.num_vars,
                    sum_check_expression,
                    state.instance.compressed_e_sum,
                    &instances,
                    &challenges,
                    &y,
                    transcript,
                )?
            };

            // PCS verify

            let builtin_witness_poly_offset = vp.num_witness_polys.iter().sum::<usize>();
            let dummy_comm = Pcs::Commitment::default();
            let comms = iter::empty()
                .chain(iter::repeat(&dummy_comm).take(vp.num_instances.len()))
                .chain(&vp.preprocess_comms)
                .chain(&state.instance.witness_comms[..builtin_witness_poly_offset])
                .chain(vp.permutation_comms.iter().map(|(_, comm)| comm))
                .chain(&state.instance.witness_comms[builtin_witness_poly_offset..])
                .chain(&permutation_z_comms)
                .chain(Some(&state.instance.zeta_e_comm))
                .collect_vec();
            Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;
        }

        Ok(())
    }
}

impl<Pb: WitnessEncoding> WitnessEncoding for Protostar<Pb> {
    fn row_mapping(k: usize) -> Vec<usize> {
        Pb::row_mapping(k)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::{
            hyperplonk::{
                folding::protostar::Protostar,
                util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_with_lookup_circuit},
                HyperPlonk,
            },
            PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo,
        },
        pcs::{
            multilinear::{MultilinearIpa, MultilinearKzg, MultilinearSimulator},
            univariate::UnivariateKzg,
            AdditiveCommitment, PolynomialCommitmentScheme,
        },
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::PrimeField,
            end_timer, start_timer,
            test::{seeded_std_rng, std_rng},
            transcript::{
                InMemoryTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite,
            },
            Itertools,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use std::{hash::Hash, iter, ops::Range};

    pub(crate) fn run_protostar_hyperplonk<F, Pcs, T, C>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<F>, Vec<Vec<Vec<F>>>, Vec<C>),
    ) where
        F: PrimeField + Ord + Hash,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
        Pcs::Commitment: AdditiveCommitment<F>,
        Pcs::CommitmentChunk: AdditiveCommitment<F>,
        T: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript,
        C: PlonkishCircuit<F>,
    {
        for num_vars in num_vars_range {
            let (circuit_info, instances, circuits) = circuit_fn(num_vars);

            let timer = start_timer(|| format!("setup-{num_vars}"));
            let param =
                Protostar::<HyperPlonk<Pcs>>::setup(&circuit_info, seeded_std_rng()).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("preprocess-{num_vars}"));
            let (pp, vp) = Protostar::<HyperPlonk<Pcs>>::preprocess(&param, &circuit_info).unwrap();
            end_timer(timer);

            let (mut prover_state, mut verifier_state) = (pp.init(), vp.init());
            for (idx, (instances, circuit)) in instances.iter().zip_eq(circuits.iter()).enumerate()
            {
                let is_folding = idx != circuits.len() - 1;
                let instances = instances.iter().map(Vec::as_slice).collect_vec();

                let timer = start_timer(|| format!("prove-{num_vars}"));
                let proof = {
                    prover_state.set_folding(is_folding);
                    let mut transcript = T::default();
                    Protostar::<HyperPlonk<Pcs>>::prove(
                        &pp,
                        &mut prover_state,
                        &instances,
                        circuit,
                        &mut transcript,
                        seeded_std_rng(),
                    )
                    .unwrap();
                    transcript.into_proof()
                };
                end_timer(timer);

                let timer = start_timer(|| format!("verify-{num_vars}"));
                let result = {
                    verifier_state.set_folding(is_folding);
                    let mut transcript = T::from_proof(proof.as_slice());
                    Protostar::<HyperPlonk<Pcs>>::verify(
                        &vp,
                        &mut verifier_state,
                        &instances,
                        &mut transcript,
                        seeded_std_rng(),
                    )
                };
                assert_eq!(result, Ok(()));
                end_timer(timer);
            }
        }
    }

    macro_rules! tests {
        ($name:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<$name _hyperplonk_protostar_vanilla_plonk>]() {
                    run_protostar_hyperplonk::<_, $pcs, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _, _) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                        let (instances, circuits) = iter::repeat_with(|| {
                            let (_, instances, circuit) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                            (instances, circuit)
                        }).take(3).unzip();
                        (circuit_info, instances, circuits)
                    });
                }

                #[test]
                fn [<$name _hyperplonk_protostar_vanilla_plonk_with_lookup>]() {
                    run_protostar_hyperplonk::<_, $pcs, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _, _) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                        let (instances, circuits) = iter::repeat_with(|| {
                            let (_, instances, circuit) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                            (instances, circuit)
                        }).take(3).unzip();
                        (circuit_info, instances, circuits)
                    });
                }
            }
        };
        ($name:ident, $pcs:ty) => {
            tests!($name, $pcs, 2..16);
        };
    }

    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(sim_kzg, MultilinearSimulator<UnivariateKzg<Bn256>>);
}
