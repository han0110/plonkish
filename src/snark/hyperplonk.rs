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
pub(crate) mod preprocess;
mod prover;
mod verifier;

pub use preprocess::PlonkishCircuitInfo;

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
    virtual_poly_info: VirtualPolynomialInfo<F>,
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
    virtual_poly_info: VirtualPolynomialInfo<F>,
    preprocess_comms: Vec<Pcs::Commitment>,
    permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, C, Pcs> UniversalSnark<F, Pcs> for HyperPlonk<Pcs>
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
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            max_degree,
            num_vars,
            virtual_poly_info: virtual_poly_info.clone(),
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
            virtual_poly_info,
            preprocess_comms,
            permutation_comms,
        };
        Ok((pp, vp))
    }

    fn prove(
        pp: &Self::ProverParam,
        instances: &[&[F]],
        mut witness_collector: impl FnMut(&[F]) -> Result<Vec<Vec<F>>, Error>,
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

                for witness_poly in witness_polys.iter() {
                    transcript.write_commitment(Pcs::commit(&pp.pcs, witness_poly)?)?;
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

        for (permuted_input_poly, permuted_table_poly) in lookup_permuted_polys.iter() {
            transcript.write_commitment(Pcs::commit(&pp.pcs, permuted_input_poly)?)?;
            transcript.write_commitment(Pcs::commit(&pp.pcs, permuted_table_poly)?)?;
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

        for z in lookup_z_polys.iter().chain(permutation_z_polys.iter()) {
            transcript.write_commitment(Pcs::commit(&pp.pcs, z)?)?;
        }

        // Phase n+2

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
            pp.num_instances.len(),
            &pp.virtual_poly_info,
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
        let (points, evals) = verify_sum_check(
            vp.num_vars,
            &vp.virtual_poly_info,
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
            end_timer, start_timer,
            test::{rand_array, rand_idx, rand_vec},
            transcript::Keccak256Transcript,
            Itertools,
        },
        Error,
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use num_integer::Integer;
    use rand::{rngs::OsRng, RngCore};
    use std::{
        collections::{HashMap, HashSet},
        hash::Hash,
        iter,
        ops::Range,
    };

    pub(crate) fn run_hyperplonk<W>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<Fr>, Vec<Vec<Fr>>, W),
    ) where
        W: FnMut(&[Fr]) -> Result<Vec<Vec<Fr>>, Error>,
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
                HyperPlonk::prove(&pp, &instances, witness, &mut transcript, OsRng).unwrap();
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
    ) -> (
        PlonkishCircuitInfo<F>,
        Vec<Vec<F>>,
        impl FnMut(&[F]) -> Result<Vec<Vec<F>>, Error>,
    ) {
        let size = 1 << num_vars;
        let mut polys = [(); 9].map(|_| vec![F::zero(); size]);

        let instances = rand_vec(num_vars, &mut rng);
        polys[0] = instances_polys(num_vars, [&instances])[0].evals().to_vec();

        let mut permutation = Permutation::default();
        for idx in 0..size {
            let [w_l, w_r, q_c] = if rng.next_u32().is_even() && idx > 1 {
                let [l_copy_idx, r_copy_idx] =
                    [(); 2].map(|_| (rand_idx(6..9, &mut rng), rand_idx(1..idx, &mut rng)));
                permutation.copy(l_copy_idx, (6, idx));
                permutation.copy(r_copy_idx, (7, idx));
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
            permutation.into_cycles(),
        );
        (circuit_info, vec![instances], move |_| {
            Ok(vec![w_l.clone(), w_r.clone(), w_o.clone()])
        })
    }

    pub(crate) fn rand_plonk_assignment<F: PrimeField>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
        let (polys, permutations) = {
            let (circuit_info, instances, mut witness) = rand_plonk_circuit(num_vars, &mut rng);
            let witness = witness(&[]).unwrap();
            let polys = iter::empty()
                .chain(instances_polys(num_vars, &instances))
                .chain(circuit_info.preprocess_polys)
                .chain(witness.into_iter().map(MultilinearPolynomial::new))
                .collect_vec();
            (polys, circuit_info.permutations)
        };
        let challenges: [_; 4] = rand_array(&mut rng);
        let [_, beta, gamma, _] = challenges;

        let permutation_polys = permutation_polys(num_vars, &[6, 7, 8], &permutations);
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
    ) -> (
        PlonkishCircuitInfo<F>,
        Vec<Vec<F>>,
        impl FnMut(&[F]) -> Result<Vec<Vec<F>>, Error>,
    ) {
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
        polys[0] = instances_polys(num_vars, [&instances])[0].evals().to_vec();

        let mut permutation = Permutation::default();
        for idx in 0..size {
            let use_copy = rng.next_u32().is_even() && idx > 1;
            let [w_l, w_r, q_c] = if use_copy {
                let [l_copy_idx, r_copy_idx] =
                    [(); 2].map(|_| (rand_idx(10..13, &mut rng), rand_idx(1..idx, &mut rng)));
                permutation.copy(l_copy_idx, (10, idx));
                permutation.copy(r_copy_idx, (11, idx));
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
            permutation.into_cycles(),
        );
        (circuit_info, vec![instances], move |_| {
            Ok(vec![w_l.clone(), w_r.clone(), w_o.clone()])
        })
    }

    pub fn rand_plonk_with_lookup_assignment<F: PrimeField + Ord + Hash>(
        num_vars: usize,
        mut rng: impl RngCore,
    ) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
        let (polys, permutations) = {
            let (circuit_info, instances, mut witness) =
                rand_plonk_with_lookup_circuit(num_vars, &mut rng);
            let witness = witness(&[]).unwrap();
            let polys = iter::empty()
                .chain(instances_polys(num_vars, &instances))
                .chain(circuit_info.preprocess_polys)
                .chain(witness.into_iter().map(MultilinearPolynomial::new))
                .collect_vec();
            (polys, circuit_info.permutations)
        };
        let challenges: [_; 4] = rand_array(&mut rng);
        let [theta, beta, gamma, _] = challenges;

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

        let permutation_polys = permutation_polys(num_vars, &[10, 11, 12], &permutations);
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

    #[derive(Default)]
    struct Permutation {
        cycles: Vec<HashSet<(usize, usize)>>,
        cycle_idx: HashMap<(usize, usize), usize>,
    }

    impl Permutation {
        fn copy(&mut self, lhs: (usize, usize), rhs: (usize, usize)) {
            match self.cycle_idx.get(&lhs).copied() {
                Some(cycle_idx) => {
                    self.cycles[cycle_idx].insert(rhs);
                    self.cycle_idx.insert(rhs, cycle_idx);
                }
                None => {
                    let cycle_idx = self.cycles.len();
                    self.cycles.push(HashSet::from_iter([lhs, rhs]));
                    for cell in [lhs, rhs] {
                        self.cycle_idx.insert(cell, cycle_idx);
                    }
                }
            };
        }

        fn into_cycles(self) -> Vec<Vec<(usize, usize)>> {
            self.cycles
                .into_iter()
                .map(|cycle| cycle.into_iter().sorted().collect_vec())
                .collect()
        }
    }
}
