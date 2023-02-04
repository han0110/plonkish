use crate::util::arithmetic::Field;
use halo2_proofs::plonk::Circuit;
use rand::RngCore;

pub use stand_plonk::StandardPlonk;

pub trait CircuitExt<F: Field>: Circuit<F> {
    fn rand(k: usize, rng: impl RngCore) -> Self;

    fn num_instances() -> Vec<usize>;

    fn instances(&self) -> Vec<Vec<F>>;
}

mod stand_plonk {
    use crate::{
        backend::hyperplonk::frontend::halo2::circuit::CircuitExt,
        util::{arithmetic::Field, Itertools},
    };
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Fixed},
        poly::Rotation,
    };
    use rand::RngCore;
    use std::iter;

    #[derive(Clone)]
    pub struct StandardPlonkConfig {
        selectors: [Column<Fixed>; 5],
        wires: [Column<Advice>; 3],
    }

    impl StandardPlonkConfig {
        fn configure<F: Field>(meta: &mut ConstraintSystem<F>) -> Self {
            let pi = meta.instance_column();
            let [q_l, q_r, q_m, q_o, q_c] = [(); 5].map(|_| meta.fixed_column());
            let [w_l, w_r, w_o] = [(); 3].map(|_| meta.advice_column());
            [w_l, w_r, w_o].map(|column| meta.enable_equality(column));
            meta.create_gate(
                "q_l·w_l + q_r·w_r + q_m·w_l·w_r + q_o·w_o + q_c + pi = 0",
                |meta| {
                    let [w_l, w_r, w_o] =
                        [w_l, w_r, w_o].map(|column| meta.query_advice(column, Rotation::cur()));
                    let [q_l, q_r, q_o, q_m, q_c] = [q_l, q_r, q_o, q_m, q_c]
                        .map(|column| meta.query_fixed(column, Rotation::cur()));
                    let pi = meta.query_instance(pi, Rotation::cur());
                    Some(
                        q_l * w_l.clone()
                            + q_r * w_r.clone()
                            + q_m * w_l * w_r
                            + q_o * w_o
                            + q_c
                            + pi,
                    )
                },
            );
            StandardPlonkConfig {
                selectors: [q_l, q_r, q_m, q_o, q_c],
                wires: [w_l, w_r, w_o],
            }
        }
    }

    #[derive(Clone, Default)]
    pub struct StandardPlonk<F>(usize, Vec<[Assigned<F>; 8]>);

    impl<F: Field> Circuit<F> for StandardPlonk<F> {
        type Config = StandardPlonkConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            unimplemented!()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            meta.set_minimum_degree(4);
            StandardPlonkConfig::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "",
                |mut region| {
                    for (offset, values) in self.1.iter().enumerate() {
                        let (selectors, wires) = values.split_at(config.selectors.len());
                        for (column, value) in
                            config.selectors.into_iter().zip(selectors.iter().copied())
                        {
                            region.assign_fixed(|| "", column, offset, || Value::known(value))?;
                        }
                        for (column, value) in config.wires.into_iter().zip(wires.iter().copied()) {
                            let cell = region
                                .assign_advice(|| "", column, offset, || Value::known(value))?
                                .cell();
                            if offset == 0 {
                                region.constrain_equal(cell, cell)?;
                            }
                        }
                    }
                    Ok(())
                },
            )
        }
    }

    impl<F: Field> CircuitExt<F> for StandardPlonk<F> {
        fn rand(k: usize, mut rng: impl RngCore) -> Self {
            let mut rand_row =
                || [(); 8].map(|_| Assigned::Rational(F::random(&mut rng), F::random(&mut rng)));
            let values = iter::empty()
                .chain(Some(rand_row()))
                .chain(
                    iter::repeat_with(|| {
                        let mut values = rand_row();
                        let [q_l, q_r, q_m, q_o, _, w_l, w_r, w_o] = values;
                        values[4] = -(q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o);
                        values
                    })
                    .take((1 << k) - 7)
                    .collect_vec(),
                )
                .collect();
            Self(k, values)
        }

        fn num_instances() -> Vec<usize> {
            vec![1]
        }

        fn instances(&self) -> Vec<Vec<F>> {
            let [q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] = self.1[0];
            let pi = (-(q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c)).evaluate();
            vec![vec![pi]]
        }
    }
}
