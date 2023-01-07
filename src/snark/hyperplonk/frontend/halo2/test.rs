use crate::{
    snark::hyperplonk::{
        frontend::halo2::{circuit::StandardPlonk, circuit_info, witness_collector},
        preprocess::test::plonk_circuit_info,
        test::run_hyperplonk,
    },
    util::Itertools,
};
use halo2_curves::bn256::Fr;
use rand::rngs::OsRng;

#[test]
fn circuit_info_plonk() {
    let circuit = StandardPlonk::<Fr>::rand(OsRng);
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
    )
}

#[test]
fn e2e_plonk() {
    let circuit = StandardPlonk::rand(OsRng);
    let instances = circuit.instances();
    let instances = instances.iter().map(Vec::as_slice).collect_vec();
    run_hyperplonk(3..16, |num_vars| {
        (
            circuit_info(num_vars, &circuit, vec![1]).unwrap(),
            circuit.instances(),
            witness_collector(num_vars, &circuit, &instances),
        )
    });
}
