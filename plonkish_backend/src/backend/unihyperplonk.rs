use crate::{
    backend::{
        unihyperplonk::{
            preprocessor::{batch_size, compose, permutation_polys},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::PolynomialCommitmentScheme,
    piop::multilinear_eval::ph23::{prove_multilinear_eval, s_polys, verify_multilinear_eval},
    poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
    util::{
        arithmetic::powers,
        chain, end_timer,
        expression::{
            rotate::{Lexical, Rotatable},
            Expression,
        },
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::ff::WithSmallOrderMulGroup;
use rand::RngCore;
use std::{borrow::Cow, fmt::Debug, hash::Hash, iter, marker::PhantomData};

pub(crate) mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Clone, Debug)]
pub struct UniHyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniHyperPlonkProverParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::ProverParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
    pub(crate) permutation_comms: Vec<Pcs::Commitment>,
    pub(crate) s_polys: Vec<Vec<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UniHyperPlonkVerifierParam<F, Pcs>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::VerifierParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) num_lookups: usize,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, Pcs> PlonkishBackend<F> for UniHyperPlonk<Pcs>
where
    F: WithSmallOrderMulGroup<3> + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
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
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();
        let (preprocess_polys, preprocess_comms) =
            batch_commit::<_, Pcs>(&pcs_pp, preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let (permutation_polys, permutation_comms) =
            batch_commit::<_, Pcs>(&pcs_pp, permutation_polys)?;

        // Compose expression
        let (num_permutation_z_polys, expression) = compose(circuit_info);
        let vp = UniHyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            num_lookups: circuit_info.lookups.len(),
            num_permutation_z_polys,
            num_vars,
            expression: expression.clone(),
            preprocess_comms: preprocess_comms.clone(),
            permutation_comms: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_comms.clone())
                .collect(),
        };
        let pp = UniHyperPlonkProverParam {
            pcs: pcs_pp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            num_permutation_z_polys,
            num_vars,
            expression,
            preprocess_polys,
            preprocess_comms,
            permutation_polys: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_polys)
                .collect(),
            permutation_comms,
            s_polys: s_polys(num_vars),
        };
        Ok((pp, vp))
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

        // Prove PH23

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
        prove_multilinear_eval::<_, Pcs>(
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

        // Verify PH23

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
        verify_multilinear_eval::<_, Pcs>(&vp.pcs, vp.num_vars, comms, &point, &evals, transcript)?;

        Ok(())
    }
}

impl<Pcs> WitnessEncoding for UniHyperPlonk<Pcs> {
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
        ($suffix:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    run_plonkish_backend::<_, UniHyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    run_plonkish_backend::<_, UniHyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_w_lookup_circuit::<_, Lexical>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($suffix:ident, $pcs:ty) => {
            tests!($suffix, $pcs, 2..16);
        };
    }

    tests!(kzg, UnivariateKzg<Bn256>);
}
