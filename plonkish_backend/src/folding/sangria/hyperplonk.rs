#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::hyperplonk::{
            util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_with_lookup_circuit},
            HyperPlonk,
        },
        folding::{sangria::Sangria, test::run_folding_scheme},
        pcs::{
            multilinear::{MultilinearIpa, MultilinearKzg, MultilinearSimulator},
            univariate::UnivariateKzg,
        },
        util::{
            test::{seeded_std_rng, std_rng},
            transcript::Keccak256Transcript,
            Itertools,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use std::iter;

    macro_rules! tests {
        ($name:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<$name _sangria_hyperplonk_vanilla_plonk>]() {
                    run_folding_scheme::<_, Sangria<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_circuit(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }

                #[test]
                fn [<$name _sangria_hyperplonk_vanilla_plonk_with_lookup>]() {
                    run_folding_scheme::<_, Sangria<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_with_lookup_circuit(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }
            }
        };
        ($name:ident, $pcs:ty) => {
            tests!($name, $pcs, 2..16);
        };
    }

    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(sim_kzg, MultilinearSimulator<UnivariateKzg<Bn256>>);
}
