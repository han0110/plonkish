#[cfg(test)]
pub(crate) mod test {
    use crate::{
        accumulation::{sangria::Sangria, test::run_accumulation_scheme},
        backend::hyperplonk::{
            util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_w_lookup_circuit},
            HyperPlonk,
        },
        pcs::{
            multilinear::{Gemini, MultilinearIpa, MultilinearKzg, Zeromorph},
            univariate::UnivariateKzg,
        },
        util::{
            expression::rotate::BinaryField,
            test::{seeded_std_rng, std_rng},
            transcript::Keccak256Transcript,
            Itertools,
        },
    };
    use halo2_curves::{bn256::Bn256, grumpkin};
    use std::iter;

    macro_rules! tests {
        ($suffix:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    run_accumulation_scheme::<_, Sangria<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_circuit::<_, BinaryField>(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_circuit::<_, BinaryField>(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    run_accumulation_scheme::<_, Sangria<HyperPlonk<$pcs>>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        let (circuit_info, _) = rand_vanilla_plonk_w_lookup_circuit::<_, BinaryField>(num_vars, std_rng(), seeded_std_rng());
                        let circuits = iter::repeat_with(|| {
                            let (_, circuit) = rand_vanilla_plonk_w_lookup_circuit::<_, BinaryField>(num_vars, std_rng(), seeded_std_rng());
                            circuit
                        }).take(3).collect_vec();
                        (circuit_info, circuits)
                    });
                }
            }
        };
        ($suffix:ident, $pcs:ty) => {
            tests!($suffix, $pcs, 2..16);
        };
    }

    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
}
