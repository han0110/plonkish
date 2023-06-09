use ark_ff::PrimeField;
use espresso_hyperplonk::prelude::{CustomizedGates, MockCircuit};

pub fn vanilla_plonk<F: PrimeField>(num_vars: usize) -> MockCircuit<F> {
    let mut circuit = MockCircuit::new(1 << num_vars, &CustomizedGates::vanilla_plonk_gate());
    circuit.public_inputs.truncate(1);
    circuit.index.params.num_pub_input = 1;
    circuit
}
