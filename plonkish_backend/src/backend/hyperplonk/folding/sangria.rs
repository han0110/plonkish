use crate::{
    backend::{
        hyperplonk::{
            folding::sangria::{
                preprocessor::batch_size,
                preprocessor::preprocess,
                prover::{evaluate_cross_term, lookup_h_polys, SangriaWitness},
                verifier::SangriaInstance,
            },
            prover::{
                instance_polys, lookup_compressed_polys, lookup_m_polys, permutation_z_polys,
                prove_zero_check,
            },
            verifier::verify_zero_check,
            HyperPlonk, HyperPlonkProverParam, HyperPlonkVerifierParam,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::{AdditiveCommitment, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        end_timer,
        expression::Expression,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::BorrowMut, hash::Hash, iter, marker::PhantomData};

pub(crate) mod preprocessor;
pub(crate) mod prover;
mod verifier;

#[derive(Clone, Debug)]
pub struct Sangria<Pb>(PhantomData<Pb>);

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, HyperPlonkProverParam<F, Pcs>: Serialize",
    deserialize = "F: DeserializeOwned, HyperPlonkProverParam<F, Pcs>: DeserializeOwned"
))]
pub struct SangriaProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pp: HyperPlonkProverParam<F, Pcs>,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    cross_term_expressions: Vec<Expression<F>>,
    zero_check_expression: Expression<F>,
}

impl<F, Pcs> SangriaProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn init(&self) -> SangriaProverState<F, Pcs> {
        SangriaProverState {
            is_folding: true,
            witness: SangriaWitness::init(
                self.pp.num_vars,
                &self.pp.num_instances,
                self.num_folding_witness_polys,
                self.num_folding_challenges,
            ),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, HyperPlonkProverParam<F, Pcs>: Serialize",
    deserialize = "F: DeserializeOwned, HyperPlonkProverParam<F, Pcs>: DeserializeOwned"
))]
pub struct SangriaVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    vp: HyperPlonkVerifierParam<F, Pcs>,
    num_theta_primes: usize,
    num_alpha_primes: usize,
    num_folding_witness_polys: usize,
    num_folding_challenges: usize,
    num_cross_terms: usize,
    zero_check_expression: Expression<F>,
}

impl<F, Pcs> SangriaVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn init(&self) -> SangriaVerifierState<F, Pcs> {
        SangriaVerifierState {
            is_folding: true,
            instance: SangriaInstance::init(
                &self.vp.num_instances,
                self.num_folding_witness_polys,
                self.num_folding_challenges,
            ),
        }
    }
}
#[derive(Debug)]
pub struct SangriaProverState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    is_folding: bool,
    witness: SangriaWitness<F, Pcs::Commitment, Pcs::Polynomial>,
}

impl<F, Pcs> SangriaProverState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn set_folding(&mut self, is_folding: bool) {
        self.is_folding = is_folding;
    }
}

#[derive(Debug)]
pub struct SangriaVerifierState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    is_folding: bool,
    instance: SangriaInstance<F, Pcs::Commitment>,
}

impl<F, Pcs> SangriaVerifierState<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub fn set_folding(&mut self, is_folding: bool) {
        self.is_folding = is_folding;
    }
}

impl<F, Pcs> PlonkishBackend<F, Pcs> for Sangria<HyperPlonk<Pcs>>
where
    F: PrimeField + Ord + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    Pcs::Commitment: AdditiveCommitment<F>,
    Pcs::CommitmentChunk: AdditiveCommitment<F>,
{
    type ProverParam = SangriaProverParam<F, Pcs>;
    type VerifierParam = SangriaVerifierParam<F, Pcs>;
    type ProverState = SangriaProverState<F, Pcs>;
    type VerifierState = SangriaVerifierState<F, Pcs>;

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
        let SangriaProverParam {
            pp,
            num_theta_primes,
            num_alpha_primes,
            cross_term_expressions,
            zero_check_expression,
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
        for (round, (num_witness_polys, num_folding_challenges)) in pp
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
            challenges.extend(transcript.squeeze_challenges(*num_folding_challenges));
        }

        // Round n

        let theta_primes = transcript.squeeze_challenges(*num_theta_primes);

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

        let alpha_primes = transcript.squeeze_challenges(*num_alpha_primes);

        let incoming = SangriaWitness::from_committed(
            pp.num_vars,
            instances,
            iter::empty()
                .chain(witness_polys)
                .chain(lookup_m_polys)
                .chain(lookup_h_polys.into_iter().flatten()),
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms),
            iter::empty()
                .chain(challenges)
                .chain(theta_primes)
                .chain(Some(beta_prime))
                .chain(alpha_primes)
                .collect(),
        );

        let timer = start_timer(|| format!("cross_term_polys-{}", cross_term_expressions.len()));
        let cross_term_polys = evaluate_cross_term(
            cross_term_expressions,
            pp.num_vars,
            &pp.preprocess_polys,
            &state.witness,
            &incoming,
        );
        end_timer(timer);

        let cross_term_comms = Pcs::batch_commit_and_write(&pp.pcs, &cross_term_polys, transcript)?;

        // Round n+3

        let r = transcript.squeeze_challenge();

        let timer = start_timer(|| "fold");
        state
            .witness
            .fold(&incoming, &cross_term_polys, &cross_term_comms, &r);
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

            // Round n+4

            let alpha = transcript.squeeze_challenge();
            let y = transcript.squeeze_challenges(pp.num_vars);

            let polys = iter::empty()
                .chain(polys)
                .chain(&state.witness.witness_polys[builtin_witness_poly_offset..])
                .chain(permutation_z_polys.iter())
                .chain(Some(&state.witness.e_poly))
                .collect_vec();
            let challenges = iter::empty()
                .chain(state.witness.instance.challenges.iter().copied())
                .chain([beta, gamma, alpha, state.witness.instance.u])
                .collect();
            let (points, evals) = {
                prove_zero_check(
                    pp.num_instances.len(),
                    zero_check_expression,
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
                .chain(Some(&state.witness.instance.e_comm))
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
        let SangriaVerifierParam {
            vp,
            num_theta_primes,
            num_alpha_primes,
            num_cross_terms,
            zero_check_expression,
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

        let theta_primes = transcript.squeeze_challenges(*num_theta_primes);

        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let beta_prime = transcript.squeeze_challenge();

        let lookup_h_comms = Pcs::read_commitments(&vp.pcs, 2 * vp.num_lookups, transcript)?;

        // Round n+2

        let alpha_primes = transcript.squeeze_challenges(*num_alpha_primes);

        let incoming = SangriaInstance::from_committed(
            instances,
            iter::empty()
                .chain(witness_comms)
                .chain(lookup_m_comms)
                .chain(lookup_h_comms),
            iter::empty()
                .chain(challenges)
                .chain(theta_primes)
                .chain(Some(beta_prime))
                .chain(alpha_primes)
                .collect(),
        );

        let cross_term_comms = Pcs::read_commitments(&vp.pcs, *num_cross_terms, transcript)?;

        // Round n+3

        let r = transcript.squeeze_challenge();

        state.instance.fold(&incoming, &cross_term_comms, &r);

        if !state.is_folding {
            let beta = transcript.squeeze_challenge();
            let gamma = transcript.squeeze_challenge();

            let permutation_z_comms =
                Pcs::read_commitments(&vp.pcs, vp.num_permutation_z_polys, transcript)?;

            // Round n+4

            let alpha = transcript.squeeze_challenge();
            let y = transcript.squeeze_challenges(vp.num_vars);

            let instances = state.instance.instance_slices();
            let challenges = iter::empty()
                .chain(state.instance.challenges.iter().copied())
                .chain([beta, gamma, alpha, state.instance.u])
                .collect_vec();
            let (points, evals) = {
                verify_zero_check(
                    vp.num_vars,
                    zero_check_expression,
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
                .chain(Some(&state.instance.e_comm))
                .collect_vec();
            Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;
        }

        Ok(())
    }
}

impl<Pb: WitnessEncoding> WitnessEncoding for Sangria<Pb> {
    fn row_mapping(k: usize) -> Vec<usize> {
        Pb::row_mapping(k)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::{
            hyperplonk::{
                folding::sangria::Sangria,
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
            DeserializeOwned, Itertools, Serialize,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use std::{hash::Hash, iter, ops::Range};

    pub(crate) fn run_sangria_hyperplonk<F, Pcs, T, C>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<F>, Vec<Vec<Vec<F>>>, Vec<C>),
    ) where
        F: PrimeField + Ord + Hash + Serialize + DeserializeOwned,
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
            let param = Sangria::<HyperPlonk<Pcs>>::setup(&circuit_info, seeded_std_rng()).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("preprocess-{num_vars}"));
            let (pp, vp) = Sangria::<HyperPlonk<Pcs>>::preprocess(&param, &circuit_info).unwrap();
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
                    Sangria::<HyperPlonk<Pcs>>::prove(
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
                    Sangria::<HyperPlonk<Pcs>>::verify(
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
                fn [<$name _hyperplonk_sangria_vanilla_plonk>]() {
                    run_sangria_hyperplonk::<_, $pcs, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _, _) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                        let (instances, circuits) = iter::repeat_with(|| {
                            let (_, instances, circuit) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                            (instances, circuit)
                        }).take(3).unzip();
                        (circuit_info, instances, circuits)
                    });
                }

                #[test]
                fn [<$name _hyperplonk_sangria_vanilla_plonk_with_lookup>]() {
                    run_sangria_hyperplonk::<_, $pcs, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
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
