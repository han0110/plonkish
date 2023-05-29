pub use aggregation::AggregationCircuit;

mod aggregation {
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
        halo2curves::{
            ff::{FromUniformBytes, WithSmallOrderMulGroup},
            serde::SerdeObject,
        },
        plonk::{create_proof, keygen_pk, keygen_vk, Circuit, ConstraintSystem, Error},
        poly::{
            commitment::{Params, ParamsProver},
            kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::ProverGWC,
            },
        },
    };
    use itertools::Itertools;
    use plonkish_backend::{
        backend::hyperplonk::frontend::halo2::circuit::{CircuitExt, StandardPlonk},
        halo2_curves::{ff::PrimeField, pairing::Engine, CurveAffine},
    };
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use snark_verifier::{
        loader::{
            self,
            halo2::{
                halo2_wrong_ecc::{self, integer::rns::Rns, maingate::*, EccConfig},
                Scalar,
            },
            native::NativeLoader,
        },
        pcs::{
            kzg::{self, *},
            AccumulationDecider, AccumulationScheme, AccumulationSchemeProver,
        },
        system::halo2::{compile, transcript, Config},
        util::arithmetic::{fe_to_limbs, MultiMillerLoop},
        verifier::{self, plonk::PlonkProtocol, SnarkVerifier},
    };
    use std::rc::Rc;

    const LIMBS: usize = 4;
    const BITS: usize = 68;

    type KzgAs<M> = kzg::KzgAs<M, Gwc19>;
    type KzgSvk<M> = KzgSuccinctVerifyingKey<<M as Engine>::G1Affine>;
    type KzgDk<M> = KzgDecidingKey<M>;
    type PlonkSuccinctVerifier<M> = verifier::plonk::PlonkSuccinctVerifier<KzgAs<M>>;

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    type EccChip<C> = halo2_wrong_ecc::BaseFieldEccChip<C, LIMBS, BITS>;
    type Halo2Loader<'a, C> = loader::halo2::Halo2Loader<'a, C, EccChip<C>>;
    type PoseidonTranscript<C, S> =
        transcript::halo2::PoseidonTranscript<C, NativeLoader, S, T, RATE, R_F, R_P>;

    #[derive(Clone)]
    struct Snark<C: CurveAffine> {
        protocol: PlonkProtocol<C>,
        instances: Vec<Vec<C::Scalar>>,
        proof: Vec<u8>,
    }

    impl<C: CurveAffine> Snark<C> {
        fn new(protocol: PlonkProtocol<C>, instances: Vec<Vec<C::Scalar>>, proof: Vec<u8>) -> Self {
            Self {
                protocol,
                instances,
                proof,
            }
        }
    }

    impl<C: CurveAffine> From<Snark<C>> for SnarkWitness<C> {
        fn from(snark: Snark<C>) -> Self {
            Self {
                protocol: snark.protocol,
                instances: Value::known(snark.instances),
                proof: Value::known(snark.proof),
            }
        }
    }

    #[derive(Clone)]
    struct SnarkWitness<C: CurveAffine> {
        protocol: PlonkProtocol<C>,
        instances: Value<Vec<Vec<C::Scalar>>>,
        proof: Value<Vec<u8>>,
    }

    impl<C: CurveAffine> SnarkWitness<C> {
        fn proof(&self) -> Value<&[u8]> {
            self.proof.as_ref().map(Vec::as_slice)
        }

        fn loaded_instances<'b>(
            &self,
            loader: &Rc<Halo2Loader<'b, C>>,
        ) -> Vec<Vec<Scalar<'b, C, EccChip<C>>>> {
            self.instances
                .as_ref()
                .transpose_vec(self.protocol.num_instance.len())
                .into_iter()
                .zip(self.protocol.num_instance.iter())
                .map(|(instances, num_instance)| {
                    instances
                        .transpose_vec(*num_instance)
                        .into_iter()
                        .map(|instance| loader.assign_scalar(instance.copied()))
                        .collect_vec()
                })
                .collect_vec()
        }
    }

    #[derive(Clone)]
    pub struct AggregationConfig {
        main_gate_config: MainGateConfig,
        range_config: RangeConfig,
    }

    impl AggregationConfig {
        fn configure<C: CurveAffine>(meta: &mut ConstraintSystem<C::Scalar>) -> Self {
            let main_gate_config = MainGate::<C::Scalar>::configure(meta);
            let range_config = RangeChip::<C::Scalar>::configure(
                meta,
                &main_gate_config,
                vec![BITS / LIMBS],
                Rns::<C::Base, C::Scalar, LIMBS, BITS>::construct().overflow_lengths(),
            );
            Self {
                main_gate_config,
                range_config,
            }
        }

        fn main_gate<F: PrimeField>(&self) -> MainGate<F> {
            MainGate::new(self.main_gate_config.clone())
        }

        fn range_chip<F: PrimeField>(&self) -> RangeChip<F> {
            RangeChip::new(self.range_config.clone())
        }

        fn ecc_chip<C: CurveAffine>(&self) -> EccChip<C> {
            EccChip::new(EccConfig::new(
                self.range_config.clone(),
                self.main_gate_config.clone(),
            ))
        }

        fn load_table<F: PrimeField>(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
            self.range_chip().load_table(layouter)
        }

        #[allow(clippy::type_complexity)]
        fn aggregate<'a, M: MultiMillerLoop>(
            &self,
            layouter: &mut impl Layouter<M::Scalar>,
            svk: &KzgSvk<M>,
            snarks: impl IntoIterator<Item = &'a SnarkWitness<M::G1Affine>>,
        ) -> Result<Vec<AssignedCell<M::Scalar, M::Scalar>>, Error>
        where
            M::Scalar: FromUniformBytes<64>,
            M::G1Affine: SerdeObject,
            M::G2Affine: SerdeObject,
        {
            type PoseidonTranscript<'a, C, S> = transcript::halo2::PoseidonTranscript<
                C,
                Rc<Halo2Loader<'a, C>>,
                S,
                T,
                RATE,
                R_F,
                R_P,
            >;
            let snarks = snarks.into_iter().collect_vec();
            layouter.assign_region(
                || "Aggregate snarks",
                |region| {
                    let ctx = RegionCtx::new(region, 0);

                    let ecc_chip = self.ecc_chip::<M::G1Affine>();
                    let loader = Halo2Loader::new(ecc_chip, ctx);

                    let accumulators = snarks
                        .iter()
                        .map(|snark| {
                            let protocol = snark.protocol.loaded(&loader);
                            let instances = snark.loaded_instances(&loader);
                            let mut transcript = PoseidonTranscript::new(&loader, snark.proof());
                            let proof = PlonkSuccinctVerifier::<M>::read_proof(
                                svk,
                                &protocol,
                                &instances,
                                &mut transcript,
                            )
                            .unwrap();
                            PlonkSuccinctVerifier::verify(svk, &protocol, &instances, &proof)
                                .unwrap()
                        })
                        .collect_vec();

                    let accumulator = {
                        let as_vk = Default::default();
                        let as_proof = Vec::new();
                        let accumulators = accumulators.into_iter().flatten().collect_vec();
                        let mut transcript =
                            PoseidonTranscript::new(&loader, Value::known(as_proof.as_slice()));
                        let proof =
                            KzgAs::<M>::read_proof(&as_vk, &accumulators, &mut transcript).unwrap();
                        KzgAs::<M>::verify(&as_vk, &accumulators, &proof).unwrap()
                    };

                    let accumulator_limbs = [accumulator.lhs, accumulator.rhs]
                        .iter()
                        .map(|ec_point| {
                            loader.ecc_chip().assign_ec_point_to_limbs(
                                &mut loader.ctx_mut(),
                                ec_point.assigned(),
                            )
                        })
                        .collect::<Result<Vec<_>, Error>>()?
                        .into_iter()
                        .flatten()
                        .collect();

                    Ok(accumulator_limbs)
                },
            )
        }
    }

    fn aggregate<'a, M: MultiMillerLoop>(
        param: &ParamsKZG<M>,
        snarks: impl IntoIterator<Item = &'a Snark<M::G1Affine>>,
    ) -> Option<[M::Scalar; 4 * LIMBS]>
    where
        M::Scalar: FromUniformBytes<64>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        let svk = KzgSvk::<M>::new(param.get_g()[0]);
        let dk = KzgDk::new(svk, param.g2(), param.s_g2());

        let accumulators = snarks
            .into_iter()
            .map(|snark| {
                let mut transcript = PoseidonTranscript::new(snark.proof.as_slice());
                let proof = PlonkSuccinctVerifier::<M>::read_proof(
                    &svk,
                    &snark.protocol,
                    &snark.instances,
                    &mut transcript,
                )?;
                PlonkSuccinctVerifier::verify(&svk, &snark.protocol, &snark.instances, &proof)
            })
            .try_collect::<_, Vec<_>, _>()
            .ok()?
            .into_iter()
            .flatten()
            .collect_vec();

        let accumulator = {
            let as_pk = Default::default();
            let rng = StdRng::from_seed(Default::default());
            let mut transcript = PoseidonTranscript::new(Vec::new());
            let accumulator =
                KzgAs::<M>::create_proof(&as_pk, &accumulators, &mut transcript, rng).ok()?;
            assert!(transcript.finalize().is_empty());
            accumulator
        };
        KzgAs::<M>::decide(&dk, accumulator.clone()).ok()?;

        let KzgAccumulator { lhs, rhs } = accumulator;
        let accumulator_limbs = [lhs, rhs]
            .into_iter()
            .flat_map(|point| {
                let coordinates = point.coordinates().unwrap();
                [*coordinates.x(), *coordinates.y()]
            })
            .flat_map(fe_to_limbs::<_, _, LIMBS, BITS>)
            .collect_vec()
            .try_into()
            .unwrap();
        Some(accumulator_limbs)
    }

    pub struct AggregationCircuit<M: MultiMillerLoop> {
        svk: KzgSvk<M>,
        snarks: Vec<SnarkWitness<M::G1Affine>>,
        instances: Vec<M::Scalar>,
    }

    impl<M: MultiMillerLoop> Circuit<M::Scalar> for AggregationCircuit<M>
    where
        M::Scalar: FromUniformBytes<64>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        type Config = AggregationConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            unimplemented!()
        }

        fn configure(meta: &mut ConstraintSystem<M::Scalar>) -> Self::Config {
            AggregationConfig::configure::<M::G1Affine>(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<M::Scalar>,
        ) -> Result<(), Error> {
            config.load_table(&mut layouter)?;

            let main_gate = config.main_gate();
            for (row, limb) in config
                .aggregate::<M>(&mut layouter, &self.svk, &self.snarks)?
                .into_iter()
                .enumerate()
            {
                main_gate.expose_public(layouter.namespace(|| ""), limb, row)?;
            }

            Ok(())
        }
    }

    impl<M: MultiMillerLoop> CircuitExt<M::Scalar> for AggregationCircuit<M>
    where
        M::Scalar: FromUniformBytes<64> + WithSmallOrderMulGroup<3>,
        M::G1Affine: SerdeObject,
        M::G2Affine: SerdeObject,
    {
        fn rand(k: usize, mut rng: impl RngCore) -> Self {
            let param = ParamsKZG::<M>::setup(4, &mut rng);
            let snark = {
                let circuit = StandardPlonk::rand(param.k() as usize, &mut rng);
                let vk = keygen_vk::<_, _, _, true>(&param, &circuit).unwrap();
                let pk = keygen_pk::<_, _, _, true>(&param, vk, &circuit).unwrap();
                let protocol = compile(
                    &param,
                    pk.get_vk(),
                    Config::kzg().with_num_instance(vec![1]),
                );
                let instances = circuit.instances();
                let proof = {
                    let mut transcript = PoseidonTranscript::new(Vec::new());
                    create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _, true>(
                        &param,
                        &pk,
                        &[circuit],
                        &[&instances.iter().map(Vec::as_slice).collect_vec()],
                        &mut rng,
                        &mut transcript,
                    )
                    .unwrap();
                    transcript.finalize()
                };
                Snark::new(protocol, instances, proof)
            };
            let snarks = vec![snark; (1 << k) / 1000000];
            let accumulator_limbs = aggregate(&param, &snarks).unwrap();
            Self {
                svk: KzgSvk::<M>::new(param.get_g()[0]),
                snarks: snarks.into_iter().map_into().collect(),
                instances: accumulator_limbs.to_vec(),
            }
        }

        fn num_instances() -> Vec<usize> {
            vec![4 * LIMBS]
        }

        fn instances(&self) -> Vec<Vec<M::Scalar>> {
            vec![self.instances.clone()]
        }
    }
}
