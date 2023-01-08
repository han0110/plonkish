use crate::{
    snark::hyperplonk::{
        frontend::halo2::{
            circuit::{CircuitExt, StandardPlonk},
            circuit_info, witness_collector,
        },
        preprocess::test::plonk_circuit_info,
        test::run_hyperplonk,
    },
    util::Itertools,
};
use halo2_curves::bn256::Fr;
use rand::rngs::OsRng;
use std::ops::Range;

#[test]
fn circuit_info_plonk() {
    let circuit = StandardPlonk::<Fr>::rand(3, OsRng);
    let circuit_info = circuit_info(3, &circuit, vec![1]).unwrap();
    assert_eq!(
        circuit_info.constraints,
        plonk_circuit_info(
            0,
            0,
            Default::default(),
            vec![vec![(6, 1)], vec![(7, 1)], vec![(8, 1)]],
        )
        .constraints
    );
    assert_eq!(circuit_info.permutation_polys(), [6, 7, 8]);
}

#[test]
fn e2e_plonk() {
    const RANGE: Range<usize> = 3..16;
    let circuits = RANGE.map(|k| StandardPlonk::rand(k, OsRng)).collect_vec();
    let instances = circuits.iter().map(CircuitExt::instances).collect_vec();
    let instances = instances
        .iter()
        .map(|instances| instances.iter().map(Vec::as_slice).collect_vec())
        .collect_vec();
    run_hyperplonk(RANGE, |num_vars| {
        let idx = num_vars - RANGE.start;
        (
            circuit_info(num_vars, &circuits[idx], vec![instances[idx][0].len()]).unwrap(),
            circuits[idx].instances(),
            witness_collector(num_vars, &circuits[idx], &instances[idx]),
        )
    });
}
