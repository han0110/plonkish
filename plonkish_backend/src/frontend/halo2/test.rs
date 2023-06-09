use crate::backend::{
    hyperplonk::{test::run_hyperplonk, util, HyperPlonk},
    PlonkishCircuit,
};
use crate::{
    frontend::halo2::{circuit::VanillaPlonk, CircuitExt, Halo2Circuit},
    pcs::multilinear::MultilinearKzg,
    util::transcript::Keccak256Transcript,
};
use halo2_curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;

#[test]
fn vanilla_plonk_circuit_info() {
    let circuit = Halo2Circuit::new::<HyperPlonk<()>>(3, VanillaPlonk::<Fr>::rand(3, OsRng));
    let circuit_info = circuit.circuit_info().unwrap();
    assert_eq!(
        circuit_info.constraints,
        util::vanilla_plonk_circuit_info(
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
fn e2e_vanilla_plonk() {
    run_hyperplonk::<_, MultilinearKzg<Bn256>, Keccak256Transcript<_>, _>(3..16, |num_vars| {
        let circuit =
            Halo2Circuit::new::<HyperPlonk<()>>(num_vars, VanillaPlonk::rand(num_vars, OsRng));
        (
            circuit.circuit_info().unwrap(),
            circuit.instances(),
            circuit,
        )
    });
}
