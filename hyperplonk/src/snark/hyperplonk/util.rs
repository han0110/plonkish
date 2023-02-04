use crate::{
    poly::multilinear::MultilinearPolynomial,
    snark::hyperplonk::{
        preprocess::{compose, permutation_polys, PlonkishCircuitInfo},
        prover::{instances_polys, lookup_permuted_polys, lookup_z_polys, permutation_z_polys},
    },
    util::{
        arithmetic::PrimeField,
        expression::{Expression, Query, Rotation},
        test::{rand_array, rand_idx, rand_vec},
        Itertools,
    },
    Error,
};
use num_integer::Integer;
use rand::RngCore;
use std::{
    array,
    collections::{HashMap, HashSet},
    hash::Hash,
    iter,
};

pub fn plonk_circuit_info<F: PrimeField>(
    num_vars: usize,
    num_instances: usize,
    preprocess_polys: [MultilinearPolynomial<F>; 5],
    permutations: Vec<Vec<(usize, usize)>>,
) -> PlonkishCircuitInfo<F> {
    let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] =
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

pub fn plonk_expression<F: PrimeField>() -> Expression<F> {
    let circuit_info = plonk_circuit_info(
        0,
        0,
        Default::default(),
        vec![vec![(6, 1)], vec![(7, 1)], vec![(8, 1)]],
    );
    let (max_degree, expression) = compose(&circuit_info);
    assert_eq!(max_degree, 4);
    expression
}

pub fn plonk_with_lookup_circuit_info<F: PrimeField>(
    num_vars: usize,
    num_instances: usize,
    preprocess_polys: [MultilinearPolynomial<F>; 9],
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

pub fn plonk_with_lookup_expression<F: PrimeField>() -> Expression<F> {
    let circuit_info = plonk_with_lookup_circuit_info(
        0,
        0,
        Default::default(),
        vec![vec![(10, 1)], vec![(11, 1)], vec![(12, 1)]],
    );
    let (max_degree, expression) = compose(&circuit_info);
    assert_eq!(max_degree, 4);
    expression
}

pub fn rand_plonk_circuit<F: PrimeField>(
    num_vars: usize,
    mut rng: impl RngCore,
) -> (
    PlonkishCircuitInfo<F>,
    Vec<Vec<F>>,
    impl Fn(&[F]) -> Result<Vec<Vec<F>>, Error>,
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

pub fn rand_plonk_assignment<F: PrimeField>(
    num_vars: usize,
    mut rng: impl RngCore,
) -> (Vec<MultilinearPolynomial<F>>, Vec<F>) {
    let (polys, permutations) = {
        let (circuit_info, instances, witness) = rand_plonk_circuit(num_vars, &mut rng);
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

pub fn rand_plonk_with_lookup_circuit<F: PrimeField + Ord>(
    num_vars: usize,
    mut rng: impl RngCore,
) -> (
    PlonkishCircuitInfo<F>,
    Vec<Vec<F>>,
    impl Fn(&[F]) -> Result<Vec<Vec<F>>, Error>,
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
        let (circuit_info, instances, witness) = rand_plonk_with_lookup_circuit(num_vars, &mut rng);
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
            .chain(lookup_permuted_polys.into_iter().flatten())
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
