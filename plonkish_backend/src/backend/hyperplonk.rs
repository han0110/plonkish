use crate::{
    backend::{
        hyperplonk::{
            preprocess::{compose, permutation_polys},
            prover::{
                instances_polys, lookup_permuted_polys, lookup_z_polys, permutation_z_polys,
                prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, PrimeField},
        end_timer,
        expression::Expression,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData};

pub mod frontend;
mod preprocess;
mod prover;
mod verifier;

#[cfg(any(test, feature = "benchmark"))]
pub mod util;

#[derive(Clone, Debug)]
pub struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug)]
pub struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pcs: Pcs::ProverParam,
    num_instances: Vec<usize>,
    num_witness_polys: Vec<usize>,
    num_challenges: Vec<usize>,
    lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    max_degree: usize,
    num_vars: usize,
    expression: Expression<F>,
    preprocess_polys: Vec<MultilinearPolynomial<F>>,
    permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
}

#[derive(Clone, Debug)]
pub struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pcs: Pcs::VerifierParam,
    num_instances: Vec<usize>,
    num_witness_polys: Vec<usize>,
    num_challenges: Vec<usize>,
    num_lookup: usize,
    max_degree: usize,
    num_vars: usize,
    expression: Expression<F>,
    preprocess_comms: Vec<Pcs::Commitment>,
    permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, C, Pcs> PlonkishBackend<F, Pcs> for HyperPlonk<Pcs>
where
    F: PrimeField + Ord + Hash,
    C: Clone + Debug,
    Pcs: PolynomialCommitmentScheme<
        F,
        Polynomial = MultilinearPolynomial<F>,
        Point = Vec<F>,
        Commitment = C,
        BatchCommitment = Vec<C>,
    >,
{
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    fn setup(size: usize, rng: impl RngCore) -> Result<Pcs::Param, Error> {
        Pcs::setup(size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let size = 1 << num_vars;
        let (pcs_pp, pcs_vp) = Pcs::trim(param, size)?;

        // Compute preprocesses comms
        let preprocess_comms = circuit_info
            .preprocess_polys
            .iter()
            .map(|poly| Pcs::commit(&pcs_pp, poly))
            .try_collect()?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = permutation_polys
            .iter()
            .map(|(idx, poly)| Ok((*idx, Pcs::commit(&pcs_pp, poly)?)))
            .try_collect::<_, Vec<_>, _>()?;

        // Compose `VirtualPolynomialInfo`
        let (max_degree, expression) = compose(&circuit_info);
        let pp = HyperPlonkProverParam {
            pcs: pcs_pp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            max_degree,
            num_vars,
            expression: expression.clone(),
            preprocess_polys: circuit_info.preprocess_polys,
            permutation_polys,
        };
        let vp = HyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            num_lookup: circuit_info.lookups.len(),
            max_degree,
            num_vars,
            expression,
            preprocess_comms,
            permutation_comms,
        };
        Ok((pp, vp))
    }

    fn prove(
        pp: &Self::ProverParam,
        instances: &[&[F]],
        witness_collector: &impl Fn(&[F]) -> Result<Vec<Vec<F>>, Error>,
        transcript: &mut impl TranscriptWrite<F, Commitment = Pcs::Commitment>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_scalar(instance)?;
            }
        }
        let instances_polys = instances_polys(pp.num_vars, instances.iter().cloned());

        // Phase 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4);
        for (phase, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            witness_polys.extend({
                let timer = start_timer(|| format!("witness_collector-{phase}"));
                let witness_polys = witness_collector(&challenges)?
                    .into_iter()
                    .map(MultilinearPolynomial::new)
                    .collect_vec();
                assert_eq!(witness_polys.len(), *num_witness_polys);
                end_timer(timer);

                for comm in Pcs::batch_commit(&pp.pcs, &witness_polys)? {
                    transcript.write_commitment(comm)?;
                }
                witness_polys
            });
            challenges.extend(transcript.squeeze_n_challenges(*num_challenges));
        }
        let polys = iter::empty()
            .chain(instances_polys.iter())
            .chain(pp.preprocess_polys.iter())
            .chain(witness_polys.iter())
            .collect_vec();

        // Phase n

        let theta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_permuted_polys-{}", pp.lookups.len()));
        let (lookup_compressed_polys, lookup_permuted_polys) =
            lookup_permuted_polys(&pp.lookups, &polys, &challenges, &theta)?;
        end_timer(timer);

        if !lookup_permuted_polys.is_empty() {
            for comm in Pcs::batch_commit(&pp.pcs, lookup_permuted_polys.iter().flatten())? {
                transcript.write_commitment(comm)?;
            }
        }

        // Phase n+1

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_z_polys-{}", pp.lookups.len()));
        let lookup_z_polys = lookup_z_polys(
            &lookup_compressed_polys,
            &lookup_permuted_polys,
            &beta,
            &gamma,
        );
        drop(lookup_compressed_polys);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys =
            permutation_z_polys(pp.max_degree, &pp.permutation_polys, &polys, &beta, &gamma);
        end_timer(timer);

        if !(lookup_z_polys.is_empty() && permutation_z_polys.is_empty()) {
            for z in Pcs::batch_commit(&pp.pcs, lookup_z_polys.iter().chain(&permutation_z_polys))?
            {
                transcript.write_commitment(z)?;
            }
        }

        // Phase n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_n_challenges(pp.num_vars);

        let polys = iter::empty()
            .chain(polys)
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .chain(lookup_permuted_polys.iter().flatten())
            .chain(lookup_z_polys.iter())
            .chain(permutation_z_polys.iter())
            .collect_vec();
        challenges.extend([theta, beta, gamma, alpha]);
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // PCS open

        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        Pcs::batch_open(&pp.pcs, polys, &points, &evals, transcript)?;
        end_timer(timer);

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[&[F]],
        transcript: &mut impl TranscriptRead<F, Commitment = Pcs::Commitment>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_scalar(instance)?;
            }
        }

        // Phase 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_witness_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(transcript.read_n_commitments(*num_witness_polys)?);
            challenges.extend(transcript.squeeze_n_challenges(*num_challenges));
        }

        // Phase n

        let theta = transcript.squeeze_challenge();

        let permuted_comms = iter::repeat_with(|| {
            Ok((transcript.read_commitment()?, transcript.read_commitment()?))
        })
        .take(vp.num_lookup)
        .try_collect::<_, Vec<_>, _>()?;

        // Phase n+1

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();

        let lookup_z_comms = transcript.read_n_commitments(vp.num_lookup)?;
        let permutation_z_comms = transcript
            .read_n_commitments(div_ceil(vp.permutation_comms.len(), vp.max_degree - 1))?;

        // Phase n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_n_challenges(vp.num_vars);

        let comms = iter::empty()
            .chain(iter::repeat(witness_comms[0].clone()).take(vp.num_instances.len()))
            .chain(vp.preprocess_comms.iter().cloned())
            .chain(witness_comms)
            .chain(vp.permutation_comms.iter().map(|(_, comm)| comm.clone()))
            .chain(permuted_comms.into_iter().flat_map(
                |(permuted_input_comm, permuted_table_comm)| {
                    [permuted_input_comm, permuted_table_comm]
                },
            ))
            .chain(lookup_z_comms)
            .chain(permutation_z_comms)
            .collect_vec();
        challenges.extend([theta, beta, gamma, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS verify

        Pcs::batch_verify(&vp.pcs, &comms, &points, &evals, transcript)?;

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::{
            hyperplonk::{
                self,
                util::{rand_plonk_circuit, rand_plonk_with_lookup_circuit},
            },
            PlonkishBackend, PlonkishCircuitInfo,
        },
        pcs::multilinear_kzg,
        util::{end_timer, start_timer, transcript::Keccak256Transcript, Itertools},
        Error,
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::rngs::OsRng;
    use std::ops::Range;

    pub(crate) fn run_hyperplonk<W>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<Fr>, Vec<Vec<Fr>>, W),
    ) where
        W: Fn(&[Fr]) -> Result<Vec<Vec<Fr>>, Error>,
    {
        type Pcs = multilinear_kzg::MultilinearKzg<Bn256>;
        type HyperPlonk = hyperplonk::HyperPlonk<Pcs>;

        for num_vars in num_vars_range {
            let (circuit_info, instances, witness) = circuit_fn(num_vars);
            let instances = instances.iter().map(Vec::as_slice).collect_vec();

            let timer = start_timer(|| format!("setup-{num_vars}"));
            let param = HyperPlonk::setup(1 << num_vars, OsRng).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("preprocess-{num_vars}"));
            let (pp, vp) = HyperPlonk::preprocess(&param, circuit_info).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("prove-{num_vars}"));
            let proof = {
                let mut transcript = Keccak256Transcript::new(Vec::new());
                HyperPlonk::prove(&pp, &instances, &witness, &mut transcript, OsRng).unwrap();
                transcript.finalize()
            };
            end_timer(timer);

            let timer = start_timer(|| format!("verify-{num_vars}"));
            let accept = {
                let mut transcript = Keccak256Transcript::new(proof.as_slice());
                HyperPlonk::verify(&vp, &instances, &mut transcript, OsRng).is_ok()
            };
            assert!(accept);
            end_timer(timer);
        }
    }

    #[test]
    fn hyperplonk_plonk() {
        run_hyperplonk(2..16, |num_vars| rand_plonk_circuit(num_vars, OsRng));
    }

    #[test]
    fn hyperplonk_plonk_with_lookup() {
        run_hyperplonk(2..16, |num_vars| {
            rand_plonk_with_lookup_circuit(num_vars, OsRng)
        });
    }
}
