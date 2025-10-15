use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{compose, permutation_polys},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys,
            },
        },
        mock::MockCircuit,
        PlonkishCircuit, PlonkishCircuitInfo,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        chain,
        expression::{rotate::Rotatable, Expression, Query, Rotation},
        test::{rand_array, rand_idx, rand_vec},
        Itertools,
    },
};
use num_integer::Integer;
use rand::RngCore;
use std::{
    array,
    collections::{HashMap, HashSet},
    hash::Hash,
    iter, mem,
};

pub fn vanilla_plonk_circuit_info<F: PrimeField>(
    num_vars: usize,
    num_instances: usize,
    preprocess_polys: [Vec<F>; 5],
    permutations: Vec<Vec<(usize, usize)>>,
) -> PlonkishCircuitInfo<F> {
    let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o]: &[Expression<F>; 9] =
        &array::from_fn(|poly| Query::new(poly, Rotation::cur())).map(Expression::Polynomial);
    PlonkishCircuitInfo {
        k: num_vars,
        num_instances: vec![num_instances],
        preprocess_polys: preprocess_polys.to_vec(),
        num_witness_polys: vec![3],
        num_challenges: vec![0],
        constraints: vec![q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi],
        lookups: Vec::new(),
        permutations,
        max_degree: Some(4),
    }
}

pub fn vanilla_plonk_expression<F: PrimeField>(num_vars: usize) -> Expression<F> {
    let circuit_info = vanilla_plonk_circuit_info(
        num_vars,
        0,
        Default::default(),
        vec![vec![(6, 1)], vec![(7, 1)], vec![(8, 1)]],
    );
    let (num_permutation_z_polys, expression) = compose(&circuit_info);
    assert_eq!(num_permutation_z_polys, 1);
    expression
}

pub fn vanilla_plonk_w_lookup_circuit_info<F: PrimeField>(
    num_vars: usize,
    num_instances: usize,
    preprocess_polys: [Vec<F>; 9],
    permutations: Vec<Vec<(usize, usize)>>,
) -> PlonkishCircuitInfo<F> {
    let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o] =
        &array::from_fn(|poly| Query::new(poly, Rotation::cur())).map(Expression::Polynomial);
    PlonkishCircuitInfo {
        k: num_vars,
        num_instances: vec![num_instances],
        preprocess_polys: preprocess_polys.to_vec(),
        num_witness_polys: vec![3],
        num_challenges: vec![0],
        constraints: vec![q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi],
        lookups: vec![vec![
            (q_lookup * w_l, t_l.clone()),
            (q_lookup * w_r, t_r.clone()),
            (q_lookup * w_o, t_o.clone()),
        ]],
        permutations,
        max_degree: Some(4),
    }
}

pub fn vanilla_plonk_w_lookup_expression<F: PrimeField>(num_vars: usize) -> Expression<F> {
    let circuit_info = vanilla_plonk_w_lookup_circuit_info(
        num_vars,
        0,
        Default::default(),
        vec![vec![(10, 1)], vec![(11, 1)], vec![(12, 1)]],
    );
    let (num_permutation_z_polys, expression) = compose(&circuit_info);
    assert_eq!(num_permutation_z_polys, 1);
    expression
}

pub fn rand_vanilla_plonk_circuit<F: PrimeField, R: Rotatable + From<usize>>(
    num_vars: usize,
    mut preprocess_rng: impl RngCore,
    mut witness_rng: impl RngCore,
) -> (PlonkishCircuitInfo<F>, impl PlonkishCircuit<F>) {
    let size = 1 << num_vars;
    let mut polys = [(); 9].map(|_| vec![F::ZERO; size]);

    let instances = rand_vec(num_vars, &mut witness_rng);
    polys[0] = mem::take(&mut instance_polys::<_, R>(num_vars, [&instances])[0]).into_evals();

    let mut permutation = Permutation::default();
    for poly in [6, 7, 8] {
        permutation.copy((poly, 1), (poly, 1));
    }
    for idx in 0..size - 1 {
        let [w_l, w_r] = if preprocess_rng.next_u32().is_even() && idx > 1 {
            let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                (
                    rand_idx(6..9, &mut preprocess_rng),
                    rand_idx(1..idx, &mut preprocess_rng),
                )
            });
            permutation.copy(l_copy_idx, (6, idx));
            permutation.copy(r_copy_idx, (7, idx));
            [
                polys[l_copy_idx.0][l_copy_idx.1],
                polys[r_copy_idx.0][r_copy_idx.1],
            ]
        } else {
            rand_array(&mut witness_rng)
        };
        let q_c = F::random(&mut preprocess_rng);
        let values = if preprocess_rng.next_u32().is_even() {
            vec![
                (1, F::ONE),
                (2, F::ONE),
                (4, -F::ONE),
                (5, q_c),
                (6, w_l),
                (7, w_r),
                (8, w_l + w_r + q_c + polys[0][idx]),
            ]
        } else {
            vec![
                (3, F::ONE),
                (4, -F::ONE),
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
    let circuit_info = vanilla_plonk_circuit_info(
        num_vars,
        instances.len(),
        [q_l, q_r, q_m, q_o, q_c],
        permutation.into_cycles(),
    );
    (
        circuit_info,
        MockCircuit::new(vec![instances], vec![w_l, w_r, w_o]),
    )
}

pub fn rand_vanilla_plonk_assignment<F: PrimeField, R: Rotatable + From<usize>>(
    num_vars: usize,
    mut preprocess_rng: impl RngCore,
    mut witness_rng: impl RngCore,
) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
    let (polys, permutations) = {
        let (circuit_info, circuit) =
            rand_vanilla_plonk_circuit::<_, R>(num_vars, &mut preprocess_rng, &mut witness_rng);
        let witness = circuit.synthesize(0, &[]).unwrap();
        let polys = chain![
            instance_polys::<_, R>(num_vars, circuit.instances()),
            chain![circuit_info.preprocess_polys, witness].map(MultilinearPolynomial::new),
        ]
        .collect_vec();
        (polys, circuit_info.permutations)
    };
    let challenges: [_; 3] = rand_array(&mut witness_rng);
    let [beta, gamma, _] = challenges;

    let permutation_polys = permutation_polys(num_vars, &[6, 7, 8], &permutations);
    let permutation_z_polys = permutation_z_polys::<_, R>(
        1,
        &[6, 7, 8]
            .into_iter()
            .zip(permutation_polys.iter().cloned())
            .collect_vec(),
        &polys.iter().collect_vec(),
        &beta,
        &gamma,
    );

    (
        chain![polys, permutation_polys, permutation_z_polys].collect_vec(),
        challenges.to_vec(),
    )
}

pub fn rand_vanilla_plonk_w_lookup_circuit<F: PrimeField + Hash, R: Rotatable + From<usize>>(
    num_vars: usize,
    mut preprocess_rng: impl RngCore,
    mut witness_rng: impl RngCore,
) -> (PlonkishCircuitInfo<F>, impl PlonkishCircuit<F>) {
    let size = 1 << num_vars;
    let mut polys = [(); 13].map(|_| vec![F::ZERO; size]);

    let [t_l, t_r, t_o] = [(); 3].map(|_| {
        chain![
            [F::ZERO; 2],
            iter::repeat_with(|| F::random(&mut preprocess_rng)),
        ]
        .take(size)
        .collect_vec()
    });
    polys[7] = t_l;
    polys[8] = t_r;
    polys[9] = t_o;

    let instances = rand_vec(num_vars, &mut witness_rng);
    polys[0] = instance_polys::<_, R>(num_vars, [&instances])[0]
        .evals()
        .to_vec();
    let instance_rows = R::from(num_vars)
        .usable_indices()
        .into_iter()
        .take(num_vars + 1)
        .collect::<HashSet<_>>();

    let mut permutation = Permutation::default();
    for poly in [10, 11, 12] {
        permutation.copy((poly, 1), (poly, 1));
    }
    for idx in 0..size - 1 {
        let use_copy = preprocess_rng.next_u32().is_even() && idx > 1;
        let [w_l, w_r] = if use_copy {
            let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                (
                    rand_idx(10..13, &mut preprocess_rng),
                    rand_idx(1..idx, &mut preprocess_rng),
                )
            });
            permutation.copy(l_copy_idx, (10, idx));
            permutation.copy(r_copy_idx, (11, idx));
            [
                polys[l_copy_idx.0][l_copy_idx.1],
                polys[r_copy_idx.0][r_copy_idx.1],
            ]
        } else {
            rand_array(&mut witness_rng)
        };
        let q_c = F::random(&mut preprocess_rng);
        let values = match (
            use_copy || instance_rows.contains(&idx),
            preprocess_rng.next_u32().is_even(),
        ) {
            (true, true) => {
                vec![
                    (1, F::ONE),
                    (2, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (10, w_l),
                    (11, w_r),
                    (12, w_l + w_r + q_c + polys[0][idx]),
                ]
            }
            (true, false) => {
                vec![
                    (3, F::ONE),
                    (4, -F::ONE),
                    (5, q_c),
                    (10, w_l),
                    (11, w_r),
                    (12, w_l * w_r + q_c + polys[0][idx]),
                ]
            }
            (false, _) => {
                let idx = rand_idx(1..size, &mut witness_rng);
                vec![
                    (6, F::ONE),
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
    let circuit_info = vanilla_plonk_w_lookup_circuit_info(
        num_vars,
        instances.len(),
        [q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o],
        permutation.into_cycles(),
    );
    (
        circuit_info,
        MockCircuit::new(vec![instances], vec![w_l, w_r, w_o]),
    )
}

pub fn rand_vanilla_plonk_w_lookup_assignment<F: PrimeField + Hash, R: Rotatable + From<usize>>(
    num_vars: usize,
    mut preprocess_rng: impl RngCore,
    mut witness_rng: impl RngCore,
) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
    let (polys, permutations) = {
        let (circuit_info, circuit) = rand_vanilla_plonk_w_lookup_circuit::<_, R>(
            num_vars,
            &mut preprocess_rng,
            &mut witness_rng,
        );
        let witness = circuit.synthesize(0, &[]).unwrap();
        let polys = chain![
            instance_polys::<_, R>(num_vars, circuit.instances()),
            chain![circuit_info.preprocess_polys, witness].map(MultilinearPolynomial::new),
        ]
        .collect_vec();
        (polys, circuit_info.permutations)
    };
    let challenges: [_; 3] = rand_array(&mut witness_rng);
    let [beta, gamma, _] = challenges;

    let (lookup_compressed_polys, lookup_m_polys) = {
        let PlonkishCircuitInfo { lookups, .. } =
            vanilla_plonk_w_lookup_circuit_info(0, 0, Default::default(), Vec::new());
        let polys = polys.iter().collect_vec();
        let betas = powers(beta).take(3).collect_vec();
        let lookup_compressed_polys =
            lookup_compressed_polys::<_, R>(&lookups, &polys, &[], &betas);
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys).unwrap();
        (lookup_compressed_polys, lookup_m_polys)
    };
    let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);

    let permutation_polys = permutation_polys(num_vars, &[10, 11, 12], &permutations);
    let permutation_z_polys = permutation_z_polys::<_, R>(
        1,
        &[10, 11, 12]
            .into_iter()
            .zip(permutation_polys.iter().cloned())
            .collect_vec(),
        &polys.iter().collect_vec(),
        &beta,
        &gamma,
    );

    (
        chain![
            polys,
            permutation_polys,
            lookup_m_polys,
            lookup_h_polys,
            permutation_z_polys,
        ]
        .collect_vec(),
        challenges.to_vec(),
    )
}

#[derive(Default)]
pub struct Permutation {
    cycles: Vec<HashSet<(usize, usize)>>,
    cycle_idx: HashMap<(usize, usize), usize>,
}

impl Permutation {
    pub fn copy(&mut self, lhs: (usize, usize), rhs: (usize, usize)) {
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

    pub fn into_cycles(self) -> Vec<Vec<(usize, usize)>> {
        self.cycles
            .into_iter()
            .map(|cycle| cycle.into_iter().sorted().collect_vec())
            .collect()
    }
}
