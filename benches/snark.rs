use halo2_curves::bn256::Bn256;
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::{ProverGWC, VerifierGWC},
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use hyperplonk::{
    pcs::multilinear_kzg,
    snark::{
        hyperplonk::{
            frontend::halo2::{circuit, circuit_info, witness_collector},
            HyperPlonk,
        },
        UniversalSnark,
    },
    util::{end_timer, start_timer, transcript::Keccak256Transcript, Itertools},
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::ops::Range;

type AggregationCircuit = circuit::AggregationCircuit<Bn256>;
type MultilinearKzg = multilinear_kzg::MultilinearKzg<Bn256>;

const K_RANGE: Range<usize> = 20..26;

fn rng() -> impl RngCore {
    StdRng::from_seed(Default::default())
}

fn bench_halo2() {
    for k in K_RANGE {
        let circuit = AggregationCircuit::rand(&ParamsKZG::setup(4, rng()), k, rng()).unwrap();
        let instances = circuit.instances();
        let instances = instances.iter().map(Vec::as_slice).collect_vec();

        let timer = start_timer(|| format!("halo2_setup-{k}"));
        let param = ParamsKZG::<Bn256>::setup(k as u32, rng());
        end_timer(timer);

        let timer = start_timer(|| format!("halo2_preprocess-{k}"));
        let vk = keygen_vk::<_, _, _, false>(&param, &circuit).unwrap();
        let pk = keygen_pk::<_, _, _, false>(&param, vk, &circuit).unwrap();
        end_timer(timer);

        let timer = start_timer(|| format!("halo2_prove-{k}"));
        let proof = {
            let mut transcript = Blake2bWrite::init(Vec::new());
            create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _, false>(
                &param,
                &pk,
                &[circuit],
                &[&instances],
                rng(),
                &mut transcript,
            )
            .unwrap();
            transcript.finalize()
        };
        end_timer(timer);

        let timer = start_timer(|| format!("halo2_verify-{k}"));
        let accept = {
            let mut transcript = Blake2bRead::init(proof.as_slice());
            verify_proof::<_, VerifierGWC<_>, _, _, _, false>(
                &param,
                pk.get_vk(),
                SingleStrategy::new(&param),
                &[&instances],
                &mut transcript,
            )
            .is_ok()
        };
        assert!(accept);
        end_timer(timer);
    }
}

fn bench_hyperplonk() {
    for k in K_RANGE {
        let circuit = AggregationCircuit::rand(&ParamsKZG::setup(4, rng()), k, rng()).unwrap();
        let instances = circuit.instances();
        let instances = instances.iter().map(Vec::as_slice).collect_vec();

        let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
        let param = HyperPlonk::<MultilinearKzg>::setup(1 << k, rng()).unwrap();
        end_timer(timer);

        let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
        let circuit_info = circuit_info(k, &circuit, AggregationCircuit::num_instance()).unwrap();
        let (pp, vp) = HyperPlonk::<MultilinearKzg>::preprocess(&param, circuit_info).unwrap();
        end_timer(timer);

        let timer = start_timer(|| format!("hyperplonk_prove-{k}"));
        let proof = {
            let mut transcript = Keccak256Transcript::new(Vec::new());
            HyperPlonk::prove(
                &pp,
                &instances,
                witness_collector(k, &circuit, &instances),
                &mut transcript,
                rng(),
            )
            .unwrap();
            transcript.finalize()
        };
        end_timer(timer);

        let timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let accept = {
            let mut transcript = Keccak256Transcript::new(proof.as_slice());
            HyperPlonk::verify(&vp, &instances, &mut transcript, rng()).is_ok()
        };
        assert!(accept);
        end_timer(timer);
    }
}

fn main() {
    bench_hyperplonk();
    bench_halo2();
}
