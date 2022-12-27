use crate::{
    pcs::PolynomialCommitmentScheme,
    piop::sum_check::VirtualPolynomialInfo,
    poly::multilinear::MultilinearPolynomial,
    snark::{
        hyperplonk::{
            preprocess::{compose, permutation_polys},
            prover::{
                instances_polys, lookup_permuted_polys, lookup_z_polys, permutation_z_polys,
                prove_sum_check,
            },
            verifier::verify_sum_check,
        },
        UniversalSnark,
    },
    util::{
        arithmetic::{div_ceil, PrimeField},
        expression::Expression,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{fmt::Debug, iter, marker::PhantomData};

pub(crate) mod preprocess;
mod prover;
mod verifier;

pub use preprocess::PlonkishCircuitInfo;

#[derive(Clone, Debug)]
struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug)]
struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pcs: Pcs::ProverParam,
    num_instance: Vec<usize>,
    num_witness_poly: Vec<usize>,
    num_challenge: Vec<usize>,
    lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    max_degree: usize,
    num_vars: usize,
    virtual_poly_info: VirtualPolynomialInfo<F>,
    preprocess_polys: Vec<MultilinearPolynomial<F>>,
    permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
}

#[derive(Clone, Debug)]
struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pcs: Pcs::VerifierParam,
    num_instance: Vec<usize>,
    num_witness_poly: Vec<usize>,
    num_challenge: Vec<usize>,
    num_lookup: usize,
    max_degree: usize,
    num_vars: usize,
    virtual_poly_info: VirtualPolynomialInfo<F>,
    preprocess_comms: Vec<Pcs::Commitment>,
    permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, C, Pcs> UniversalSnark<F, Pcs> for HyperPlonk<Pcs>
where
    F: PrimeField + Ord,
    C: Clone + Debug,
    Pcs: PolynomialCommitmentScheme<
        F,
        Polynomial = MultilinearPolynomial<F>,
        Point = Vec<F>,
        Commitment = C,
        BatchCommitment = Vec<C>,
    >,
{
    type CircuitInfo = PlonkishCircuitInfo<F>;
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    fn setup(size: usize, rng: impl RngCore) -> Result<Pcs::Param, Error> {
        Pcs::setup(size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: Self::CircuitInfo,
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
        let (max_degree, virtual_poly_info) = compose(&circuit_info);
        let pp = HyperPlonkProverParam {
            pcs: pcs_pp,
            num_instance: circuit_info.num_instance.clone(),
            num_witness_poly: circuit_info.num_witness_poly.clone(),
            num_challenge: circuit_info.num_challenge.clone(),
            lookups: circuit_info.lookups.clone(),
            max_degree,
            num_vars,
            virtual_poly_info: virtual_poly_info.clone(),
            preprocess_polys: circuit_info.preprocess_polys,
            permutation_polys,
        };
        let vp = HyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instance: circuit_info.num_instance.clone(),
            num_witness_poly: circuit_info.num_witness_poly.clone(),
            num_challenge: circuit_info.num_challenge.clone(),
            num_lookup: circuit_info.lookups.len(),
            max_degree,
            num_vars,
            virtual_poly_info,
            preprocess_comms,
            permutation_comms,
        };
        Ok((pp, vp))
    }

    fn prove(
        pp: &Self::ProverParam,
        instances: &[&[F]],
        mut witness_collector: impl FnMut(&[F]) -> Vec<Vec<F>>,
        transcript: &mut impl TranscriptWrite<F, Commitment = Pcs::Commitment>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instance, instances) in pp.num_instance.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instance);
            for instance in instances.iter() {
                transcript.common_scalar(instance)?;
            }
        }
        let instances_polys = instances_polys(pp.num_vars, instances);

        let mut witness_polys = Vec::with_capacity(pp.num_witness_poly.iter().sum());
        let mut challenges = Vec::with_capacity(pp.num_challenge.iter().sum::<usize>() + 4);
        for (num_witness_poly, num_challenge) in
            pp.num_witness_poly.iter().zip_eq(pp.num_challenge.iter())
        {
            witness_polys.extend({
                let witness_polys = witness_collector(&challenges)
                    .into_iter()
                    .map(MultilinearPolynomial::new)
                    .collect_vec();
                assert_eq!(witness_polys.len(), *num_witness_poly);
                for witness_poly in witness_polys.iter() {
                    transcript.write_commitment(Pcs::commit(&pp.pcs, witness_poly)?)?;
                }
                witness_polys
            });
            challenges.extend(transcript.squeeze_n_challenges(*num_challenge));
        }
        let polys = iter::empty()
            .chain(instances_polys.iter())
            .chain(pp.preprocess_polys.iter())
            .chain(witness_polys.iter())
            .collect_vec();

        let theta = transcript.squeeze_challenge();
        let (lookup_compressed_polys, lookup_permuted_polys) =
            lookup_permuted_polys(&pp.lookups, &polys, &challenges, &theta)?;
        for (permuted_input_poly, permuted_table_poly) in lookup_permuted_polys.iter() {
            transcript.write_commitment(Pcs::commit(&pp.pcs, permuted_input_poly)?)?;
            transcript.write_commitment(Pcs::commit(&pp.pcs, permuted_table_poly)?)?;
        }

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();
        let lookup_z_polys = lookup_z_polys(
            &lookup_compressed_polys,
            &lookup_permuted_polys,
            &beta,
            &gamma,
        );
        drop(lookup_compressed_polys);
        let permutation_z_polys =
            permutation_z_polys(pp.max_degree, &pp.permutation_polys, &polys, &beta, &gamma);
        for z in lookup_z_polys.iter().chain(permutation_z_polys.iter()) {
            transcript.write_commitment(Pcs::commit(&pp.pcs, z)?)?;
        }

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_n_challenges(pp.num_vars);
        let polys = iter::empty()
            .chain(polys)
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .chain(lookup_permuted_polys.iter().flat_map(
                |(permuted_input_poly, permuted_table_poly)| {
                    [permuted_input_poly, permuted_table_poly]
                },
            ))
            .chain(lookup_z_polys.iter())
            .chain(permutation_z_polys.iter())
            .collect_vec();
        challenges.extend([theta, beta, gamma, alpha]);
        let (points, evals) = prove_sum_check(
            pp.num_instance.len(),
            &pp.virtual_poly_info,
            &polys,
            challenges,
            y,
            transcript,
        )?;
        Pcs::batch_open(&pp.pcs, polys, &points, &evals, transcript)?;

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[&[F]],
        transcript: &mut impl TranscriptRead<F, Commitment = Pcs::Commitment>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instance, instances) in vp.num_instance.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instance);
            for instance in instances.iter() {
                transcript.common_scalar(instance)?;
            }
        }

        let mut witness_comms = Vec::with_capacity(vp.num_witness_poly.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenge.iter().sum::<usize>() + 4);
        for (num_witness_poly, num_challenge) in
            vp.num_witness_poly.iter().zip_eq(vp.num_challenge.iter())
        {
            witness_comms.extend(transcript.read_n_commitments(*num_witness_poly)?);
            challenges.extend(transcript.squeeze_n_challenges(*num_challenge));
        }

        let theta = transcript.squeeze_challenge();
        let permuted_comms = iter::repeat_with(|| {
            Ok((transcript.read_commitment()?, transcript.read_commitment()?))
        })
        .take(vp.num_lookup)
        .try_collect::<_, Vec<_>, _>()?;

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();
        let lookup_z_comms = transcript.read_n_commitments(vp.num_lookup)?;
        let permutation_z_comms = transcript
            .read_n_commitments(div_ceil(vp.permutation_comms.len(), vp.max_degree - 1))?;

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_n_challenges(vp.num_vars);
        let comms = iter::empty()
            .chain(iter::repeat(witness_comms[0].clone()).take(vp.num_instance.len()))
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
        let (points, evals) = verify_sum_check(
            vp.num_vars,
            &vp.virtual_poly_info,
            instances,
            &challenges,
            &y,
            transcript,
        )?;
        Pcs::batch_verify(&vp.pcs, &comms, &points, &evals, transcript)?;

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        pcs::multilinear_kzg,
        poly::multilinear::MultilinearPolynomial,
        snark::{
            hyperplonk::{
                self,
                preprocess::{
                    permutation_polys, test::plonk_circuit_info,
                    test::plonk_with_lookup_circuit_info, PlonkishCircuitInfo,
                },
                prover::{
                    instances_polys, lookup_permuted_polys, lookup_z_polys, permutation_z_polys,
                },
            },
            UniversalSnark,
        },
        util::{
            arithmetic::PrimeField,
            test::{rand_array, rand_idx, rand_vec},
            transcript::Keccak256Transcript,
            Itertools,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use num_integer::Integer;
    use rand::{rngs::OsRng, RngCore};
    use std::{iter, ops::Range};

    fn run_hyperplonk(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<Fr>, Vec<Fr>, Vec<Vec<Fr>>),
    ) {
        type Pcs = multilinear_kzg::MultilinearKzg<Bn256>;
        type HyperPlonk = hyperplonk::HyperPlonk<Pcs>;

        for num_vars in num_vars_range {
            let param = HyperPlonk::setup(1 << num_vars, OsRng).unwrap();
            let (circuit_info, instances, witnesses) = circuit_fn(num_vars);
            let (pp, vp) = HyperPlonk::preprocess(&param, circuit_info).unwrap();

            let proof = {
                let witnesses = |_: &[Fr]| witnesses.clone();
                let mut transcript = Keccak256Transcript::new(Vec::new());
                HyperPlonk::prove(&pp, &[&instances], witnesses, &mut transcript, OsRng).unwrap();
                transcript.finalize()
            };

            let accept = {
                let mut transcript = Keccak256Transcript::new(proof.as_slice());
                HyperPlonk::verify(&vp, &[&instances], &mut transcript, OsRng).is_ok()
            };
            assert!(accept);
        }
    }

    #[test]
    fn test_hyperplonk_plonk() {
        run_hyperplonk(2..16, |num_vars| rand_plonk_circuit(num_vars, OsRng));
    }

    #[test]
    fn test_hyperplonk_plonk_with_lookup() {
        run_hyperplonk(2..16, |num_vars| {
            rand_plonk_with_lookup_circuit(num_vars, OsRng)
        });
    }

    pub(crate) fn rand_plonk_circuit<F: PrimeField>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (PlonkishCircuitInfo<F>, Vec<F>, Vec<Vec<F>>) {
        let size = 1 << num_vars;
        let mut polys = [(); 9].map(|_| vec![F::zero(); size]);

        let instances = rand_vec(num_vars, &mut rng);
        polys[0] = instances_polys(num_vars, &[&instances])[0].evals().to_vec();

        let mut cycles = Vec::new();
        for idx in 0..size {
            let [w_l, w_r, q_c] = if rng.next_u32().is_even() && idx > 1 {
                let [l_copy_idx, r_copy_idx] =
                    [(); 2].map(|_| (rand_idx(6..9, &mut rng), rand_idx(1..idx, &mut rng)));
                copy(&mut cycles, l_copy_idx, (6, idx));
                copy(&mut cycles, r_copy_idx, (7, idx));
                [
                    polys[l_copy_idx.0][l_copy_idx.1],
                    polys[r_copy_idx.0][r_copy_idx.1],
                    F::zero(),
                ]
            } else {
                rand_array(&mut rng)
            };
            let values = if rng.next_u32().is_even() {
                vec![
                    (1, F::one()),
                    (2, F::one()),
                    (4, -F::one()),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l + w_r + q_c + polys[0][idx]),
                ]
            } else {
                vec![
                    (3, F::one()),
                    (4, -F::one()),
                    (5, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l * w_r + q_c + polys[0][idx]),
                ]
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }

        let [_, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] = polys;
        let circuit_info = plonk_circuit_info(
            num_vars,
            instances.len(),
            [q_l, q_r, q_m, q_o, q_c].map(MultilinearPolynomial::new),
            cycles,
        );
        (circuit_info, instances, vec![w_l, w_r, w_o])
    }

    pub(crate) fn rand_plonk_assignment<F: PrimeField>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
        let (circuit_info, instances, witnesses) = rand_plonk_circuit(num_vars, &mut rng);
        let challenges: [_; 4] = rand_array(&mut rng);
        let [_, beta, gamma, _] = challenges;

        let polys = iter::empty()
            .chain(instances_polys(num_vars, &[&instances]))
            .chain(circuit_info.preprocess_polys)
            .chain(witnesses.into_iter().map(MultilinearPolynomial::new))
            .collect_vec();

        let permutation_polys = permutation_polys(num_vars, &[6, 7, 8], &circuit_info.permutations);
        let permutation_z_polys = permutation_z_polys(
            4,
            &permutation_polys,
            &polys.iter().collect_vec(),
            &beta,
            &gamma,
        );

        (
            iter::empty()
                .chain(polys)
                .chain(permutation_polys.into_iter().map(|(_, poly)| poly))
                .chain(permutation_z_polys)
                .collect_vec(),
            challenges.to_vec(),
        )
    }

    pub(crate) fn rand_plonk_with_lookup_circuit<F: PrimeField + Ord>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (PlonkishCircuitInfo<F>, Vec<F>, Vec<Vec<F>>) {
        let size = 1 << num_vars;
        let mut polys = [(); 13].map(|_| vec![F::zero(); size]);

        let (t_l, t_r, t_o) = {
            let max = 1u64 << ((num_vars >> 1) - num_vars.is_even() as usize);
            iter::once((F::zero(), F::zero(), F::zero()))
                .chain(
                    (0..max)
                        .cartesian_product(0..max)
                        .map(|(lhs, rhs)| (F::from(lhs), F::from(rhs), F::from(lhs ^ rhs))),
                )
                .chain(iter::repeat_with(|| (F::zero(), F::zero(), F::zero())))
                .take(size)
                .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>()
        };
        polys[7] = t_l;
        polys[8] = t_r;
        polys[9] = t_o;

        let instances = rand_vec(num_vars, &mut rng);
        polys[0] = instances_polys(num_vars, &[&instances])[0].evals().to_vec();

        let mut cycles = Vec::new();
        for idx in 0..size {
            let use_copy = rng.next_u32().is_even() && idx > 1;
            let [w_l, w_r, q_c] = if use_copy {
                let [l_copy_idx, r_copy_idx] =
                    [(); 2].map(|_| (rand_idx(10..13, &mut rng), rand_idx(1..idx, &mut rng)));
                copy(&mut cycles, l_copy_idx, (10, idx));
                copy(&mut cycles, r_copy_idx, (11, idx));
                [
                    polys[l_copy_idx.0][l_copy_idx.1],
                    polys[r_copy_idx.0][r_copy_idx.1],
                    F::zero(),
                ]
            } else {
                rand_array(&mut rng)
            };
            let values = match (
                use_copy || !polys[0][idx].is_zero_vartime(),
                rng.next_u32().is_even(),
            ) {
                (true, true) => {
                    vec![
                        (1, F::one()),
                        (2, F::one()),
                        (4, -F::one()),
                        (5, q_c),
                        (10, w_l),
                        (11, w_r),
                        (12, w_l + w_r + q_c + polys[0][idx]),
                    ]
                }
                (true, false) => {
                    vec![
                        (3, F::one()),
                        (4, -F::one()),
                        (5, q_c),
                        (10, w_l),
                        (11, w_r),
                        (12, w_l * w_r + q_c + polys[0][idx]),
                    ]
                }
                (false, _) => {
                    let idx = rand_idx(1..size, &mut rng);
                    vec![
                        (6, F::one()),
                        (10, polys[7][idx]),
                        (11, polys[8][idx]),
                        (12, polys[9][idx]),
                    ]
                }
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }

        let [_, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o] = polys;
        let circuit_info = plonk_with_lookup_circuit_info(
            num_vars,
            instances.len(),
            [q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o].map(MultilinearPolynomial::new),
            cycles,
        );
        (circuit_info, instances, vec![w_l, w_r, w_o])
    }

    pub fn rand_plonk_with_lookup_assignment<F: PrimeField + Ord>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
        let (circuit_info, instances, witnesses) =
            rand_plonk_with_lookup_circuit(num_vars, &mut rng);
        let challenges: [_; 4] = rand_array(&mut rng);
        let [theta, beta, gamma, _] = challenges;

        let polys = iter::empty()
            .chain(instances_polys(num_vars, &[&instances]))
            .chain(circuit_info.preprocess_polys)
            .chain(witnesses.into_iter().map(MultilinearPolynomial::new))
            .collect_vec();

        let (lookup_compressed_polys, lookup_permuted_polys) = {
            let PlonkishCircuitInfo { lookups, .. } =
                plonk_with_lookup_circuit_info(0, 0, Default::default(), Vec::new());
            lookup_permuted_polys(&lookups, &polys.iter().collect_vec(), &[], &theta).unwrap()
        };
        let lookup_z_polys = lookup_z_polys(
            &lookup_compressed_polys,
            &lookup_permuted_polys,
            &beta,
            &gamma,
        );

        let permutation_polys =
            permutation_polys(num_vars, &[10, 11, 12], &circuit_info.permutations);
        let permutation_z_polys = permutation_z_polys(
            4,
            &permutation_polys,
            &polys.iter().collect_vec(),
            &beta,
            &gamma,
        );

        (
            iter::empty()
                .chain(polys)
                .chain(permutation_polys.into_iter().map(|(_, poly)| poly))
                .chain(
                    lookup_permuted_polys
                        .into_iter()
                        .flat_map(|(input, table)| [input, table]),
                )
                .chain(lookup_z_polys)
                .chain(permutation_z_polys)
                .collect_vec(),
            challenges.to_vec(),
        )
    }

    fn copy(cycles: &mut Vec<Vec<(usize, usize)>>, lhs: (usize, usize), rhs: (usize, usize)) {
        assert_ne!(lhs, rhs);
        if let Some(pos) = cycles.iter().position(|cycle| cycle.contains(&lhs)) {
            cycles[pos].push(rhs);
        } else {
            cycles.push(vec![lhs, rhs]);
        }
    }
}
