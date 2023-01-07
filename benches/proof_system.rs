use halo2_curves::bn256::{Bn256, Fr};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::ParamsKZG,
        multiopen::{ProverGWC, VerifierGWC},
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use hyperplonk::{
    pcs::multilinear_kzg,
    snark::{
        self,
        hyperplonk::frontend::halo2::{
            circuit::{AggregationCircuit, CircuitExt, StandardPlonk},
            circuit_info, witness_collector,
        },
        UniversalSnark,
    },
    util::{end_timer, start_timer, transcript::Keccak256Transcript, Itertools},
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::{env::args, ops::Range};

fn main() {
    let (system, circuit, k_range) = parse_args();
    match circuit {
        Circuit::Aggregation => {
            bench::<AggregationCircuit<Bn256>>(system, k_range);
        }
        Circuit::StandardPlonk => {
            bench::<StandardPlonk<Fr>>(system, k_range);
        }
    }
}

fn bench<C: CircuitExt<Fr>>(system: System, k_range: Range<usize>) {
    match system {
        System::All => {
            k_range.for_each(|k| {
                bench_hyperplonk::<C>(k);
                bench_halo2::<C>(k);
            });
        }
        System::HyperPlonk => {
            k_range.for_each(bench_hyperplonk::<C>);
        }
        System::Halo2 => {
            k_range.for_each(bench_halo2::<C>);
        }
    }
}

fn bench_hyperplonk<C: CircuitExt<Fr>>(k: usize) {
    type MultilinearKzg = multilinear_kzg::MultilinearKzg<Bn256>;
    type HyperPlonk = snark::hyperplonk::HyperPlonk<MultilinearKzg>;

    let circuit = C::rand(k, std_rng());
    let instances = circuit.instances();
    let instances = instances.iter().map(Vec::as_slice).collect_vec();
    let witness = witness_collector(k, &circuit, &instances);

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(1 << k, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    let circuit_info = circuit_info(k, &circuit, C::num_instances()).unwrap();
    let (pp, vp) = HyperPlonk::preprocess(&param, circuit_info).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let proof = {
        let mut transcript = Keccak256Transcript::new(Vec::new());
        HyperPlonk::prove(&pp, &instances, witness, &mut transcript, std_rng()).unwrap();
        transcript.finalize()
    };
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_verify-{k}"));
    let accept = {
        let mut transcript = Keccak256Transcript::new(proof.as_slice());
        HyperPlonk::verify(&vp, &instances, &mut transcript, std_rng()).is_ok()
    };
    assert!(accept);
    end_timer(timer);
}

fn bench_halo2<C: CircuitExt<Fr>>(k: usize) {
    let circuit = C::rand(k, std_rng());
    let instances = circuit.instances();
    let instances = instances.iter().map(Vec::as_slice).collect_vec();

    let timer = start_timer(|| format!("halo2_setup-{k}"));
    let param = ParamsKZG::<Bn256>::setup(k as u32, std_rng());
    end_timer(timer);

    let timer = start_timer(|| format!("halo2_preprocess-{k}"));
    let vk = keygen_vk::<_, _, _, false>(&param, &circuit).unwrap();
    let pk = keygen_pk::<_, _, _, false>(&param, vk, &circuit).unwrap();
    let create_proof =
        |c, d, e, f| create_proof::<_, ProverGWC<_>, _, _, _, _, false>(&param, &pk, c, d, e, f);
    let verify_proof =
        |c, d, e| verify_proof::<_, VerifierGWC<_>, _, _, _, false>(&param, pk.get_vk(), c, d, e);
    end_timer(timer);

    let timer = start_timer(|| format!("halo2_prove-{k}"));
    let proof = {
        let mut transcript = Blake2bWrite::init(Vec::new());
        create_proof(&[circuit], &[&instances], std_rng(), &mut transcript).unwrap();
        transcript.finalize()
    };
    end_timer(timer);

    let timer = start_timer(|| format!("halo2_verify-{k}"));
    let accept = {
        let mut transcript = Blake2bRead::init(proof.as_slice());
        let strategy = SingleStrategy::new(&param);
        verify_proof(strategy, &[&instances], &mut transcript).is_ok()
    };
    assert!(accept);
    end_timer(timer);
}

#[derive(Debug, Clone, Copy)]
enum System {
    All,
    HyperPlonk,
    Halo2,
}

#[derive(Debug, Clone, Copy)]
enum Circuit {
    Aggregation,
    StandardPlonk,
}

impl Circuit {
    fn min_k(&self) -> usize {
        match self {
            Circuit::Aggregation => 20,
            Circuit::StandardPlonk => 4,
        }
    }
}

fn parse_args() -> (System, Circuit, Range<usize>) {
    let (system, circuit, k_range) = args().chain(Some("".to_string())).tuple_windows().fold(
        (System::All, Circuit::Aggregation, 20..26),
        |(mut system, mut circuit, mut k_range), (key, value)| {
            match key.as_str() {
                "--system" => match value.as_str() {
                    "all" => system = System::All,
                    "hyperplonk" => system = System::HyperPlonk,
                    "halo2" => system = System::Halo2,
                    _ => panic!("system should be one of {{all,hyperplonk,halo2}}"),
                },
                "--circuit" => match value.as_str() {
                    "aggregation" => circuit = Circuit::Aggregation,
                    "standard_plonk" => circuit = Circuit::StandardPlonk,
                    _ => panic!("circuit should be one of {{aggregation,standard_plonk}}"),
                },
                "--k" => {
                    if let Some((start, end)) = value.split_once("..") {
                        k_range = start.parse().expect("k range start to be usize")
                            ..end.parse().expect("k range end to be usize");
                    } else {
                        k_range.start = value.parse().expect("k to be usize");
                        k_range.end = k_range.start + 1;
                    }
                }
                _ => {}
            }
            (system, circuit, k_range)
        },
    );
    if k_range.start < circuit.min_k() {
        panic!("k should be at least {} for {circuit:?}", circuit.min_k());
    }
    (system, circuit, k_range)
}

fn std_rng() -> impl RngCore {
    StdRng::from_seed(Default::default())
}
