use benchmark::{
    espresso,
    halo2::{AggregationCircuit, Keccak256Circuit, Sha256Circuit},
};
use espresso_hyperplonk::{prelude::MockCircuit, HyperPlonkSNARK};
use espresso_subroutines::{MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::ParamsKZG,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use itertools::Itertools;
use plonkish_backend::{
    backend::{self, PlonkishBackend, PlonkishCircuit, WitnessEncoding},
    frontend::halo2::{circuit::VanillaPlonk, CircuitExt, Halo2Circuit},
    halo2_curves::bn256::{Bn256, Fr},
    pcs::{multilinear, univariate, CommitmentChunk},
    util::{
        end_timer, start_timer,
        test::std_rng,
        transcript::{InMemoryTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite},
    },
};
use std::{
    env::args,
    fmt::Display,
    fs::{create_dir, File, OpenOptions},
    io::{Cursor, Write},
    iter,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};

const OUTPUT_DIR: &str = "../target/bench";

fn main() {
    let (systems, circuit, k_range) = parse_args();
    create_output(&systems);
    k_range.for_each(|k| systems.iter().for_each(|system| system.bench(k, circuit)));
}

fn bench_plonkish_backend<B, C>(system: System, k: usize)
where
    B: PlonkishBackend<Fr> + WitnessEncoding,
    C: CircuitExt<Fr>,
    Keccak256Transcript<Cursor<Vec<u8>>>: TranscriptRead<CommitmentChunk<Fr, B::Pcs>, Fr>
        + TranscriptWrite<CommitmentChunk<Fr, B::Pcs>, Fr>
        + InMemoryTranscript,
{
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<B>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("{system}_setup-{k}"));
    let param = B::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("{system}_preprocess-{k}"));
    let (pp, vp) = B::preprocess(&param, &circuit_info).unwrap();
    end_timer(timer);

    let proof = sample(system, k, || {
        let _timer = start_timer(|| format!("{system}_prove-{k}"));
        let mut transcript = Keccak256Transcript::default();
        B::prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
        transcript.into_proof()
    });

    let _timer = start_timer(|| format!("{system}_verify-{k}"));
    let accept = {
        let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
        B::verify(&vp, instances, &mut transcript, std_rng()).is_ok()
    };
    assert!(accept);
}

fn bench_hyperplonk<C: CircuitExt<Fr>>(k: usize) {
    type GeminiKzg = multilinear::Gemini<univariate::UnivariateKzg<Bn256>>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<GeminiKzg>;
    bench_plonkish_backend::<HyperPlonk, C>(System::HyperPlonk, k)
}

fn bench_unihyperplonk<C: CircuitExt<Fr>>(k: usize) {
    type UnivariateKzg = univariate::UnivariateKzg<Bn256>;
    type UniHyperPlonk = backend::unihyperplonk::UniHyperPlonk<UnivariateKzg, true>;
    bench_plonkish_backend::<UniHyperPlonk, C>(System::UniHyperPlonk, k)
}

fn bench_halo2<C: CircuitExt<Fr>>(k: usize) {
    let circuit = C::rand(k, std_rng());
    let circuits = &[circuit];
    let instances = circuits[0].instances();
    let instances = instances.iter().map(Vec::as_slice).collect_vec();
    let instances = [instances.as_slice()];

    let timer = start_timer(|| format!("halo2_setup-{k}"));
    let param = ParamsKZG::<Bn256>::setup(k as u32, std_rng());
    end_timer(timer);

    let timer = start_timer(|| format!("halo2_preprocess-{k}"));
    let vk = keygen_vk::<_, _, _, false>(&param, &circuits[0]).unwrap();
    let pk = keygen_pk::<_, _, _, false>(&param, vk, &circuits[0]).unwrap();
    end_timer(timer);

    let create_proof = |c, d, e, mut f: Blake2bWrite<_, _, _>| {
        create_proof::<_, ProverSHPLONK<_>, _, _, _, _, false>(&param, &pk, c, d, e, &mut f)
            .unwrap();
        f.finalize()
    };
    let verify_proof = |c, d, e| {
        verify_proof::<_, VerifierSHPLONK<_>, _, _, _, false>(&param, pk.get_vk(), c, d, e)
    };

    let proof = sample(System::Halo2, k, || {
        let _timer = start_timer(|| format!("halo2_prove-{k}"));
        let transcript = Blake2bWrite::init(Vec::new());
        create_proof(circuits, &instances, std_rng(), transcript)
    });

    let _timer = start_timer(|| format!("halo2_verify-{k}"));
    let accept = {
        let mut transcript = Blake2bRead::init(proof.as_slice());
        let strategy = SingleStrategy::new(&param);
        verify_proof(strategy, &instances, &mut transcript).is_ok()
    };
    assert!(accept);
}

fn bench_espresso_hyperplonk(circuit: MockCircuit<ark_bn254::Fr>) {
    macro_rules! hyperplonk {
        ($func:ident($($args:expr),*)) => {
            <PolyIOP<ark_bn254::Fr> as HyperPlonkSNARK<ark_bn254::Bn254, MultilinearKzgPCS<ark_bn254::Bn254>>>::$func($($args),*)
        };
    }

    let k = circuit.num_variables();
    let MockCircuit {
        index,
        public_inputs,
        witnesses,
        ..
    } = &circuit;

    let timer = start_timer(|| format!("espresso_hyperplonk_setup-{k}"));
    let param = MultilinearKzgPCS::gen_srs_for_testing(&mut std_rng(), k).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("espresso_hyperplonk_preprocess-{k}"));
    let (pk, vk) = hyperplonk!(preprocess(index, &param)).unwrap();
    end_timer(timer);

    let proof = sample(System::EspressoHyperPlonk, k, || {
        let _timer = start_timer(|| format!("espresso_hyperplonk_prove-{k}"));
        hyperplonk!(prove(&pk, public_inputs, witnesses)).unwrap()
    });

    let _timer = start_timer(|| format!("espresso_hyperplonk_verify-{k}"));
    let accept = hyperplonk!(verify(&vk, public_inputs, &proof)).unwrap();
    assert!(accept);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    HyperPlonk,
    UniHyperPlonk,
    Halo2,
    EspressoHyperPlonk,
}

impl System {
    fn all() -> Vec<System> {
        vec![
            System::HyperPlonk,
            System::UniHyperPlonk,
            System::Halo2,
            System::EspressoHyperPlonk,
        ]
    }

    fn output_path(&self) -> String {
        format!("{OUTPUT_DIR}/{self}")
    }

    fn output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.output_path())
            .unwrap()
    }

    fn support(&self, circuit: Circuit) -> bool {
        match self {
            System::HyperPlonk | System::UniHyperPlonk | System::Halo2 => match circuit {
                Circuit::VanillaPlonk
                | Circuit::Aggregation
                | Circuit::Sha256
                | Circuit::Keccak256 => true,
            },
            System::EspressoHyperPlonk => match circuit {
                Circuit::VanillaPlonk => true,
                Circuit::Aggregation | Circuit::Sha256 | Circuit::Keccak256 => false,
            },
        }
    }

    fn bench(&self, k: usize, circuit: Circuit) {
        if !self.support(circuit) {
            println!("skip benchmark on {circuit} with {self} because it's not compatible");
            return;
        }

        println!("start benchmark on 2^{k} {circuit} with {self}");

        match self {
            System::HyperPlonk => match circuit {
                Circuit::VanillaPlonk => bench_hyperplonk::<VanillaPlonk<Fr>>(k),
                Circuit::Aggregation => bench_hyperplonk::<AggregationCircuit<Bn256>>(k),
                Circuit::Sha256 => bench_hyperplonk::<Sha256Circuit>(k),
                Circuit::Keccak256 => bench_hyperplonk::<Keccak256Circuit>(k),
            },
            System::UniHyperPlonk => match circuit {
                Circuit::VanillaPlonk => bench_unihyperplonk::<VanillaPlonk<Fr>>(k),
                Circuit::Aggregation => bench_unihyperplonk::<AggregationCircuit<Bn256>>(k),
                Circuit::Sha256 => bench_unihyperplonk::<Sha256Circuit>(k),
                Circuit::Keccak256 => bench_unihyperplonk::<Keccak256Circuit>(k),
            },
            System::Halo2 => match circuit {
                Circuit::VanillaPlonk => bench_halo2::<VanillaPlonk<Fr>>(k),
                Circuit::Aggregation => bench_halo2::<AggregationCircuit<Bn256>>(k),
                Circuit::Sha256 => bench_halo2::<Sha256Circuit>(k),
                Circuit::Keccak256 => bench_halo2::<Keccak256Circuit>(k),
            },
            System::EspressoHyperPlonk => match circuit {
                Circuit::VanillaPlonk => bench_espresso_hyperplonk(espresso::vanilla_plonk(k)),
                Circuit::Aggregation | Circuit::Sha256 | Circuit::Keccak256 => unreachable!(),
            },
        }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::HyperPlonk => write!(f, "hyperplonk"),
            System::UniHyperPlonk => write!(f, "unihyperplonk"),
            System::Halo2 => write!(f, "halo2"),
            System::EspressoHyperPlonk => write!(f, "espresso_hyperplonk"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Circuit {
    VanillaPlonk,
    Aggregation,
    Sha256,
    Keccak256,
}

impl Circuit {
    fn min_k(&self) -> usize {
        match self {
            Circuit::VanillaPlonk => 4,
            Circuit::Aggregation => 20,
            Circuit::Sha256 => 17,
            Circuit::Keccak256 => 10,
        }
    }
}

impl Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Circuit::VanillaPlonk => write!(f, "vanilla_plonk"),
            Circuit::Aggregation => write!(f, "aggregation"),
            Circuit::Sha256 => write!(f, "sha256"),
            Circuit::Keccak256 => write!(f, "keccak256"),
        }
    }
}

fn parse_args() -> (Vec<System>, Circuit, Range<usize>) {
    let (systems, circuit, k_range) = args().chain(Some("".to_string())).tuple_windows().fold(
        (Vec::new(), Circuit::Aggregation, 20..26),
        |(mut systems, mut circuit, mut k_range), (key, value)| {
            match key.as_str() {
                "--system" => match value.as_str() {
                    "all" => systems = System::all(),
                    "hyperplonk" => systems.push(System::HyperPlonk),
                    "unihyperplonk" => systems.push(System::UniHyperPlonk),
                    "halo2" => systems.push(System::Halo2),
                    "espresso_hyperplonk" => systems.push(System::EspressoHyperPlonk),
                    _ => panic!(
                        "system should be one of {{all,hyperplonk,unihyperplonk,halo2,espresso_hyperplonk}}"
                    ),
                },
                "--circuit" => match value.as_str() {
                    "vanilla_plonk" => circuit = Circuit::VanillaPlonk,
                    "aggregation" => circuit = Circuit::Aggregation,
                    "sha256" => circuit = Circuit::Sha256,
                    "keccak256" => circuit = Circuit::Keccak256,
                    _ => panic!("circuit should be one of {{aggregation,vanilla_plonk}}"),
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
            (systems, circuit, k_range)
        },
    );
    if k_range.start < circuit.min_k() {
        panic!("k should be at least {} for {circuit:?}", circuit.min_k());
    }
    let mut systems = systems.into_iter().sorted().dedup().collect_vec();
    if systems.is_empty() {
        systems = System::all();
    };
    (systems, circuit, k_range)
}

fn create_output(systems: &[System]) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    for system in systems {
        File::create(system.output_path()).unwrap();
    }
}

fn sample<T>(system: System, k: usize, prove: impl Fn() -> T) -> T {
    let mut proof = None;
    let sample_size = sample_size(k);
    let sum = iter::repeat_with(|| {
        let start = Instant::now();
        proof = Some(prove());
        start.elapsed()
    })
    .take(sample_size)
    .sum::<Duration>();
    let avg = sum / sample_size as u32;
    writeln!(&mut system.output(), "{k}, {}", avg.as_millis()).unwrap();
    proof.unwrap()
}

fn sample_size(k: usize) -> usize {
    if k < 16 {
        20
    } else if k < 20 {
        5
    } else {
        1
    }
}
