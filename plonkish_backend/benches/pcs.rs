use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use halo2_curves::bn256::{Bn256, Fr, G1Affine};
use plonkish_backend::{
    pcs::{
        multilinear::{
            Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa, MultilinearKzg,
            Zeromorph,
        },
        univariate::UnivariateKzg,
        Point, PolynomialCommitmentScheme,
    },
    poly::Polynomial,
    util::{
        arithmetic::PrimeField,
        code::BrakedownSpec6,
        hash::Keccak256,
        test::std_rng,
        transcript::{InMemoryTranscript, Keccak256Transcript, TranscriptWrite},
    },
};
use std::{any::type_name, io::Cursor, ops::Range};

const NUM_VARS_RANGE: Range<usize> = 16..21;

fn pcs_name<Pcs>() -> &'static str {
    type_name::<Pcs>()
        .split("::")
        .find(|s| s.chars().next().unwrap().is_uppercase())
        .unwrap()
        .split('<')
        .next()
        .unwrap()
}

#[allow(clippy::type_complexity)]
fn prepare<F, Pcs>(
    k: usize,
) -> (
    Pcs::ProverParam,
    Pcs::Polynomial,
    Pcs::Commitment,
    Point<F, Pcs::Polynomial>,
    F,
)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    let n = 1 << k;
    let mut rng = std_rng();
    let param = Pcs::setup(n, 1, &mut rng).unwrap();
    let (pp, _) = Pcs::trim(&param, n, 1).unwrap();
    let poly = Pcs::Polynomial::rand(n, &mut rng);
    let comm = Pcs::commit(&pp, &poly).unwrap();
    let point = Pcs::Polynomial::rand_point(k, &mut rng);
    let eval = poly.evaluate(&point);
    (pp, poly, comm, point, eval)
}

fn commit<F, Pcs>(group: &mut BenchmarkGroup<impl Measurement>)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    let name = pcs_name::<Pcs>();
    for k in NUM_VARS_RANGE {
        let (pp, poly, _, _, _) = prepare::<F, Pcs>(k);
        group.bench_with_input(BenchmarkId::new(name, k), &k, |b, _| {
            b.iter(|| Pcs::commit(black_box(&pp), black_box(&poly)).unwrap())
        });
    }
}

fn open<F, Pcs>(group: &mut BenchmarkGroup<impl Measurement>)
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
    Keccak256Transcript<Cursor<Vec<u8>>>: TranscriptWrite<Pcs::CommitmentChunk, F>,
{
    let name = pcs_name::<Pcs>();
    for k in NUM_VARS_RANGE {
        let (pp, poly, comm, point, eval) = prepare::<F, Pcs>(k);
        group.bench_with_input(BenchmarkId::new(name, k), &k, |b, _| {
            b.iter(|| {
                Pcs::open(
                    black_box(&pp),
                    black_box(&poly),
                    black_box(&comm),
                    black_box(&point),
                    black_box(&eval),
                    black_box(&mut Keccak256Transcript::new(())),
                )
                .unwrap()
            })
        });
    }
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit");
    group.sample_size(10);

    commit::<_, MultilinearBrakedown<Fr, Keccak256, BrakedownSpec6>>(&mut group);
    commit::<_, MultilinearKzg<Bn256>>(&mut group);
    commit::<_, MultilinearIpa<G1Affine>>(&mut group);
    commit::<_, MultilinearHyrax<G1Affine>>(&mut group);
    commit::<_, Gemini<UnivariateKzg<Bn256>>>(&mut group);
    commit::<_, Zeromorph<UnivariateKzg<Bn256>>>(&mut group);
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    group.sample_size(10);

    open::<_, MultilinearBrakedown<Fr, Keccak256, BrakedownSpec6>>(&mut group);
    open::<_, MultilinearKzg<Bn256>>(&mut group);
    open::<_, MultilinearIpa<G1Affine>>(&mut group);
    open::<_, MultilinearHyrax<G1Affine>>(&mut group);
    open::<_, Gemini<UnivariateKzg<Bn256>>>(&mut group);
    open::<_, Zeromorph<UnivariateKzg<Bn256>>>(&mut group);
}

criterion_group!(benches, bench_commit, bench_open);
criterion_main!(benches);
