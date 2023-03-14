use crate::{
    backend::hyperplonk::{
        frontend::halo2::{
            circuit::{CircuitExt, StandardPlonk},
            circuit_info, Halo2Circuit,
        },
        test::run_hyperplonk,
        util::plonk_circuit_info,
    },
    pcs::multilinear::MultilinearKzg,
    util::transcript::Keccak256Transcript,
};
use halo2_curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;

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
    run_hyperplonk::<_, MultilinearKzg<Bn256>, Keccak256Transcript<_>, _>(3..16, |num_vars| {
        let circuit = StandardPlonk::rand(num_vars, OsRng);
        let instances = circuit.instances();
        (
            circuit_info(num_vars, &circuit, vec![instances[0].len()]).unwrap(),
            circuit.instances(),
            Halo2Circuit::new(num_vars, instances.clone(), circuit.clone()),
        )
    });
}
