pub mod halo2_wrong {
    use crate::util::{
        arithmetic::{div_ceil, fe_from_bool, FromUniformBytes, PrimeField, PrimeFieldBits},
        hash::poseidon::{self, SparseMDSMatrix, Spec as PoseidonSpec},
        Itertools,
    };
    use halo2_wrong_v2::{
        integer::{chip::IntegerChip, rns::Rns, Integer},
        maingate::operations::Collector,
        Composable, Scaled, SecondDegreeScaled, Term, Witness,
    };
    use std::{iter, mem};

    #[derive(Clone, Debug)]
    struct PoseidonState<F: PrimeField, const T: usize, const RATE: usize> {
        inner: [Witness<F>; T],
    }

    impl<F: PrimeField + Ord, const T: usize, const RATE: usize> PoseidonState<F, T, RATE> {
        fn new(collector: &mut Collector<F>) -> Self {
            let inner = poseidon::State::default()
                .words()
                .map(|state| collector.register_constant(state));
            Self { inner }
        }

        fn power5_with_constant(
            collector: &mut Collector<F>,
            x: &Witness<F>,
            constant: &F,
        ) -> Witness<F> {
            let x2 = collector.mul(x, x);
            let x4 = collector.mul(&x2, &x2);
            collector.mul_add_constant_scaled(F::ONE, &x4, x, *constant)
        }

        fn sbox_full(&mut self, collector: &mut Collector<F>, constants: &[F; T]) {
            for (state, constant) in self.inner.iter_mut().zip(constants.iter()) {
                *state = Self::power5_with_constant(collector, state, constant);
            }
        }

        fn sbox_part(&mut self, collector: &mut Collector<F>, constant: &F) {
            self.inner[0] = Self::power5_with_constant(collector, &self.inner[0], constant);
        }

        fn absorb_with_pre_constants(
            &mut self,
            collector: &mut Collector<F>,
            inputs: &[Witness<F>],
            pre_constants: &[F; T],
        ) {
            assert!(inputs.len() < T);

            self.inner[0] = collector.add_constant(&self.inner[0], pre_constants[0]);
            self.inner
                .iter_mut()
                .zip(pre_constants.iter())
                .skip(1)
                .zip(inputs)
                .for_each(|((state, constant), input)| {
                    *state = sum_with_coeff_and_constant(
                        collector,
                        [(state as &_, F::ONE), (input, F::ONE)],
                        *constant,
                    );
                });
            self.inner
                .iter_mut()
                .zip(pre_constants.iter())
                .skip(1 + inputs.len())
                .enumerate()
                .for_each(|(idx, (state, constant))| {
                    *state = collector.add_constant(
                        state,
                        if idx == 0 {
                            F::ONE + constant
                        } else {
                            *constant
                        },
                    )
                });
        }

        fn apply_mds(&mut self, collector: &mut Collector<F>, mds: &[[F; T]; T]) {
            self.inner = mds
                .iter()
                .map(|row| sum_with_coeff(collector, self.inner.iter().zip(row.iter().copied())))
                .collect_vec()
                .try_into()
                .unwrap();
        }

        fn apply_sparse_mds(
            &mut self,
            collector: &mut Collector<F>,
            mds: &SparseMDSMatrix<F, T, RATE>,
        ) {
            self.inner = iter::once(sum_with_coeff(
                collector,
                self.inner.iter().zip(mds.row().iter().copied()),
            ))
            .chain(
                mds.col_hat()
                    .iter()
                    .zip(self.inner.iter().skip(1))
                    .map(|(coeff, state)| {
                        collector.add_scaled(
                            &Scaled::new(&self.inner[0], *coeff),
                            &Scaled::new(state, F::ONE),
                        )
                    }),
            )
            .collect_vec()
            .try_into()
            .unwrap();
        }
    }

    #[derive(Clone, Debug)]
    pub struct PoseidonChip<F: PrimeField, const T: usize, const RATE: usize> {
        spec: PoseidonSpec<F, T, RATE>,
        state: PoseidonState<F, T, RATE>,
        buf: Vec<Witness<F>>,
    }

    impl<F: PrimeField + Ord, const T: usize, const RATE: usize> PoseidonChip<F, T, RATE> {
        pub fn new(collector: &mut Collector<F>, r_f: usize, r_p: usize) -> Self
        where
            F: FromUniformBytes<64>,
        {
            Self {
                spec: PoseidonSpec::new(r_f, r_p),
                state: PoseidonState::new(collector),
                buf: Vec::new(),
            }
        }

        pub fn from_spec(collector: &mut Collector<F>, spec: PoseidonSpec<F, T, RATE>) -> Self {
            Self {
                spec,
                state: PoseidonState::new(collector),
                buf: Vec::new(),
            }
        }

        pub fn update(&mut self, elements: &[Witness<F>]) {
            self.buf.extend_from_slice(elements);
        }

        pub fn squeeze(&mut self, collector: &mut Collector<F>) -> Witness<F> {
            let buf = mem::take(&mut self.buf);
            let exact = buf.len() % RATE == 0;

            for chunk in buf.chunks(RATE) {
                self.permutation(collector, chunk);
            }
            if exact {
                self.permutation(collector, &[]);
            }

            self.state.inner[1]
        }

        fn permutation(&mut self, collector: &mut Collector<F>, inputs: &[Witness<F>]) {
            let r_f = self.spec.r_f() / 2;
            let mds = self.spec.mds_matrices().mds().rows();
            let pre_sparse_mds = self.spec.mds_matrices().pre_sparse_mds().rows();
            let sparse_matrices = self.spec.mds_matrices().sparse_matrices();

            // First half of the full rounds
            let constants = self.spec.constants().start();
            self.state
                .absorb_with_pre_constants(collector, inputs, &constants[0]);
            for constants in constants.iter().skip(1).take(r_f - 1) {
                self.state.sbox_full(collector, constants);
                self.state.apply_mds(collector, &mds);
            }
            self.state.sbox_full(collector, constants.last().unwrap());
            self.state.apply_mds(collector, &pre_sparse_mds);

            // Partial rounds
            let constants = self.spec.constants().partial();
            for (constant, sparse_mds) in constants.iter().zip(sparse_matrices.iter()) {
                self.state.sbox_part(collector, constant);
                self.state.apply_sparse_mds(collector, sparse_mds);
            }

            // Second half of the full rounds
            let constants = self.spec.constants().end();
            for constants in constants.iter() {
                self.state.sbox_full(collector, constants);
                self.state.apply_mds(collector, &mds);
            }
            self.state.sbox_full(collector, &[F::ZERO; T]);
            self.state.apply_mds(collector, &mds);
        }
    }

    pub fn to_le_bits<F: PrimeFieldBits + Ord>(
        collector: &mut Collector<F>,
        witness: &Witness<F>,
        num_bits: usize,
    ) -> Vec<Witness<F>> {
        let le_bits = witness
            .value()
            .map(|witness| {
                witness
                    .to_le_bits()
                    .into_iter()
                    .take(num_bits)
                    .collect_vec()
            })
            .transpose_vec(num_bits)
            .into_iter()
            .map(|bit| {
                let bit = collector.new_witness(bit.map(fe_from_bool));
                collector.assert_bit(&bit);
                bit
            })
            .collect_vec();

        let composed = from_le_bits(collector, &le_bits);
        collector.equal(witness, &composed);
        le_bits
    }

    pub fn to_le_bits_strict<F: PrimeFieldBits + Ord>(
        collector: &mut Collector<F>,
        witness: &Witness<F>,
    ) -> Vec<Witness<F>> {
        let num_bits_halved = div_ceil(F::NUM_BITS as usize, 2);
        let num_bits = num_bits_halved * 2;
        let max_le_bits = (-F::ONE)
            .to_le_bits()
            .into_iter()
            .take(num_bits)
            .collect_vec();
        let witness_le_bits = to_le_bits(collector, witness, num_bits);

        let carry = F::ONE.double().pow([(num_bits_halved + 1) as u64]);
        let (terms, constant) = max_le_bits
            .chunks_exact(2)
            .zip(witness_le_bits.chunks_exact(2))
            .enumerate()
            .fold(
                (Vec::new(), F::ZERO),
                |(mut terms, mut constant), (exp, (max_le_bits, witness_le_bits))| {
                    let lt = F::ONE.double().pow([exp as u64]);
                    let gt = carry - lt;
                    let (lsb, msb) = (&witness_le_bits[0], &witness_le_bits[1]);
                    match max_le_bits {
                        [false, false] => terms.extend([
                            Term::First(Scaled::new(lsb, gt)),
                            Term::First(Scaled::new(msb, gt)),
                            Term::Second(SecondDegreeScaled::new(lsb, msb, -gt)),
                        ]),
                        [true, false] => {
                            terms.extend([
                                Term::First(Scaled::new(msb, gt)),
                                Term::First(Scaled::new(lsb, -lt)),
                                Term::First(Scaled::new(msb, -lt)),
                                Term::Second(SecondDegreeScaled::new(lsb, msb, lt)),
                            ]);
                            constant += lt;
                        }
                        [false, true] => {
                            terms.extend([
                                Term::First(Scaled::new(msb, -lt)),
                                Term::Second(SecondDegreeScaled::new(lsb, msb, gt)),
                            ]);
                            constant += lt;
                        }
                        [true, true] => {
                            terms.extend([Term::Second(SecondDegreeScaled::new(lsb, msb, -lt))]);
                            constant += lt;
                        }
                        _ => unreachable!(),
                    };
                    (terms, constant)
                },
            );
        let is_gt_indicator = collector.compose_second_degree(&terms, constant, F::ONE);
        let is_gt_indicator_le_bits = to_le_bits(
            collector,
            &is_gt_indicator,
            num_bits_halved + (num_bits as f64).log2().ceil() as usize,
        );
        collector.assert_zero(&is_gt_indicator_le_bits[num_bits_halved]);
        is_gt_indicator_le_bits[num_bits_halved]
            .value()
            .assert_if_known(|bit| *bit == F::ZERO);

        witness_le_bits
    }

    pub fn from_le_bits<F: PrimeField + Ord>(
        collector: &mut Collector<F>,
        le_bits: &[Witness<F>],
    ) -> Witness<F> {
        sum_with_coeff(
            collector,
            le_bits
                .iter()
                .enumerate()
                .map(|(exp, witness)| (witness, F::ONE.double().pow([exp as u64]))),
        )
    }

    pub fn integer_to_native<
        F,
        N,
        const NUM_LIMBS: usize,
        const NUM_LIMB_BITS: usize,
        const NUM_SUBLIMBS: usize,
    >(
        rns: &Rns<F, N, NUM_LIMBS, NUM_LIMB_BITS, NUM_SUBLIMBS>,
        collector: &mut Collector<N>,
        integer: &Integer<F, N, NUM_LIMBS, NUM_LIMB_BITS>,
        num_bits: usize,
    ) -> Witness<N>
    where
        F: PrimeField + Ord,
        N: PrimeField + Ord,
    {
        let mut integer_chip = IntegerChip::new(collector, rns);
        let scalar_le_bits = integer_chip.to_bits(integer);
        scalar_le_bits
            .iter()
            .skip(num_bits)
            .for_each(|bit| collector.assert_zero(bit));
        from_le_bits(collector, &scalar_le_bits[..num_bits])
    }

    pub fn sum_with_coeff_and_constant<'a, F: PrimeField + Ord>(
        collector: &mut Collector<F>,
        terms: impl IntoIterator<Item = (&'a Witness<F>, F)>,
        constant: F,
    ) -> Witness<F> {
        collector.compose(
            &terms
                .into_iter()
                .map(|(witness, coeff)| Scaled::new(witness, coeff))
                .collect_vec(),
            constant,
            F::ONE,
        )
    }

    pub fn sum_with_coeff<'a, F: PrimeField + Ord>(
        collector: &mut Collector<F>,
        terms: impl IntoIterator<Item = (&'a Witness<F>, F)>,
    ) -> Witness<F> {
        sum_with_coeff_and_constant(collector, terms, F::ZERO)
    }

    #[cfg(test)]
    mod test {
        use crate::{
            frontend::halo2::chip::halo2_wrong::PoseidonChip,
            util::{hash::poseidon::Poseidon, Itertools},
        };
        use halo2_curves::bn256::Fr;
        use halo2_proofs::circuit::Value;
        use halo2_wrong_v2::{maingate::operations::Collector, Composable};

        #[test]
        fn poseidon_chip() {
            const T: usize = 5;
            const RATE: usize = 4;
            const R_F: usize = 8;
            const R_P: usize = 60;

            let mut collector = Collector::<Fr>::default();
            let mut poseidon_chip = PoseidonChip::<_, T, RATE>::new(&mut collector, R_F, R_P);
            let mut poseidon = Poseidon::<_, T, RATE>::new(R_F, R_P);

            let inputs = vec![];
            let witnesses = inputs
                .iter()
                .map(|input| collector.new_witness(Value::known(*input)))
                .collect_vec();

            let lhs = {
                poseidon.update(&inputs);
                poseidon.squeeze()
            };
            let rhs = {
                poseidon_chip.update(&witnesses);
                poseidon_chip.squeeze(&mut collector)
            };

            rhs.value().assert_if_known(|rhs| lhs == *rhs);
        }
    }
}
