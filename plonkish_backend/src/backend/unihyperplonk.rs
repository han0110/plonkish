use crate::{
    backend::{
        hyperplonk::{HyperPlonkProverParam, HyperPlonkVerifierParam},
        unihyperplonk::{
            preprocessor::{batch_size, preprocess},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::{Additive, PolynomialCommitmentScheme},
    piop::multilinear_eval::ph23::{self, s_polys},
    poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
    util::{
        arithmetic::{powers, WithSmallOrderMulGroup},
        chain, end_timer,
        expression::rotate::{Lexical, Rotatable},
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::Cow, fmt::Debug, hash::Hash, iter, marker::PhantomData, ops::Deref};

pub(crate) mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct UniHyperPlonk<Pcs, const ADDITIVE_PCS: bool>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct UniHyperPlonkProverParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pp: HyperPlonkProverParam<F, Pcs>,
    pub(crate) s_polys: Vec<Vec<F>>,
}

impl<F, Pcs> Deref for UniHyperPlonkProverParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type Target = HyperPlonkProverParam<F, Pcs>;

    fn deref(&self) -> &Self::Target {
        &self.pp
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct UniHyperPlonkVerifierParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) vp: HyperPlonkVerifierParam<F, Pcs>,
}

impl<F, Pcs> Deref for UniHyperPlonkVerifierParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type Target = HyperPlonkVerifierParam<F, Pcs>;

    fn deref(&self) -> &Self::Target {
        &self.vp
    }
}

impl<F, Pcs> PlonkishBackend<F> for UniHyperPlonk<Pcs, true>
where
    F: WithSmallOrderMulGroup<3> + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
    Pcs::Commitment: Additive<F>,
{
    type Pcs = Pcs;
    type ProverParam = UniHyperPlonkProverParam<F, Pcs>;
    type VerifierParam = UniHyperPlonkVerifierParam<F, Pcs>;

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
        let (pp, vp) = preprocess(param, circuit_info, |pp, polys| {
            batch_commit::<_, Pcs>(pp, polys)
        })?;
        let s_polys = s_polys(circuit_info.k);
        Ok((
            UniHyperPlonkProverParam { pp, s_polys },
            UniHyperPlonkVerifierParam { vp },
        ))
    }

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys::<_, Lexical>(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
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

            let (polys, comms) = batch_commit_and_write::<_, Pcs>(&pp.pcs, polys, transcript)?;
            witness_comms.extend(comms);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = chain![
            instance_polys.into_iter().map(Cow::Owned),
            pp.preprocess_polys.iter().map(Cow::Borrowed),
            witness_polys.into_iter().map(Cow::Owned)
        ]
        .collect_vec();

        // Round n

        let beta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, Lexical>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);

        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?;
        end_timer(timer);

        let (lookup_m_polys, lookup_m_comms) =
            batch_commit_and_write::<_, Pcs>(&pp.pcs, lookup_m_polys, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, Lexical>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys, permutation_z_polys].collect_vec();
        let (lookup_h_permutation_z_polys, lookup_h_permutation_z_comms) =
            batch_commit_and_write::<_, Pcs>(&pp.pcs, lookup_h_permutation_z_polys, transcript)?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = chain![
            polys,
            chain![&pp.permutation_polys].map(|(_, poly)| Cow::Borrowed(poly)),
            lookup_m_polys.into_iter().map(Cow::Owned),
            lookup_h_permutation_z_polys.into_iter().map(Cow::Owned),
        ]
        .collect_vec();
        challenges.extend([beta, gamma, alpha]);
        let (point, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // Prove PH23 multilinear evaluation

        let polys = polys
            .into_iter()
            .map(|poly| poly.into_owned().into_evals())
            .map(UnivariatePolynomial::lagrange)
            .collect_vec();
        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(pp.num_instances.len()),
            &pp.preprocess_comms,
            &witness_comms,
            &pp.permutation_comms,
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        let timer = start_timer(|| format!("prove_multilinear_eval-{}", evals.len()));
        ph23::additive::prove_multilinear_eval::<_, Pcs>(
            &pp.pcs,
            pp.num_vars,
            &pp.s_polys,
            &polys,
            comms,
            &point,
            &evals,
            transcript,
        )?;
        end_timer(timer);

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n

        let beta = transcript.squeeze_challenge();

        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        let lookup_h_permutation_z_comms = Pcs::read_commitments(
            &vp.pcs,
            vp.num_lookups + vp.num_permutation_z_polys,
            transcript,
        )?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, alpha]);
        let (point, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // Verify PH23 multilinear evaluation

        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(vp.num_instances.len()),
            &vp.preprocess_comms,
            &witness_comms,
            vp.permutation_comms.iter().map(|(_, comm)| comm),
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
        ]
        .collect_vec();
        ph23::additive::verify_multilinear_eval::<_, Pcs>(
            &vp.pcs,
            vp.num_vars,
            comms,
            &point,
            &evals,
            transcript,
        )?;

        Ok(())
    }
}

impl<Pcs, const ADDITIVE_PCS: bool> WitnessEncoding for UniHyperPlonk<Pcs, ADDITIVE_PCS> {
    fn row_mapping(k: usize) -> Vec<usize> {
        Lexical::new(k).usable_indices()
    }
}

#[allow(clippy::type_complexity)]
fn batch_commit<F, Pcs>(
    pp: &Pcs::ProverParam,
    polys: impl IntoIterator<Item = MultilinearPolynomial<F>>,
) -> Result<(Vec<MultilinearPolynomial<F>>, Vec<Pcs::Commitment>), Error>
where
    F: WithSmallOrderMulGroup<3> + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
{
    let polys = polys
        .into_iter()
        .map(MultilinearPolynomial::into_evals)
        .map(UnivariatePolynomial::lagrange)
        .collect_vec();
    let comms = Pcs::batch_commit(pp, &polys)?;
    let polys = polys
        .into_iter()
        .map(UnivariatePolynomial::into_coeffs)
        .map(MultilinearPolynomial::new)
        .collect_vec();
    Ok((polys, comms))
}

#[allow(clippy::type_complexity)]
fn batch_commit_and_write<F, Pcs>(
    pp: &Pcs::ProverParam,
    polys: impl IntoIterator<Item = MultilinearPolynomial<F>>,
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> Result<(Vec<MultilinearPolynomial<F>>, Vec<Pcs::Commitment>), Error>
where
    F: WithSmallOrderMulGroup<3> + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
{
    let polys = polys
        .into_iter()
        .map(MultilinearPolynomial::into_evals)
        .map(UnivariatePolynomial::lagrange)
        .collect_vec();
    let comms = Pcs::batch_commit_and_write(pp, &polys, transcript)?;
    let polys = polys
        .into_iter()
        .map(UnivariatePolynomial::into_coeffs)
        .map(MultilinearPolynomial::new)
        .collect_vec();
    Ok((polys, comms))
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_w_lookup_circuit},
            test::run_plonkish_backend,
            unihyperplonk::UniHyperPlonk,
        },
        pcs::univariate::UnivariateKzg,
        util::{
            expression::rotate::Lexical, test::seeded_std_rng, transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::bn256::Bn256;

    macro_rules! tests {
        ($suffix:ident, $pcs:ty, $additive:literal, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    run_plonkish_backend::<_, UniHyperPlonk<$pcs, $additive>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    run_plonkish_backend::<_, UniHyperPlonk<$pcs, $additive>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_w_lookup_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($suffix:ident, $pcs:ty, $additive:literal) => {
            tests!($suffix, $pcs, $additive, 2..16);
        };
    }

    tests!(kzg, UnivariateKzg<Bn256>, true);
}
