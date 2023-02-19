use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use halo2_curves::bn256::Fr;
use plonkish_backend::{
    backend::hyperplonk::util::{plonk_expression, rand_plonk_assignment},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    util::{
        test::{rand_vec, std_rng},
        transcript::Keccak256Transcript,
    },
};
use pprof::criterion::{Output, PProfProfiler};

type ZeroCheck = ClassicSumCheck<EvaluationsProver<Fr, true>>;

fn run(num_vars: usize, virtual_poly: VirtualPolynomial<Fr>) {
    let mut transcript = Keccak256Transcript::<Vec<u8>>::default();
    ZeroCheck::prove(&(), num_vars, virtual_poly, Fr::zero(), &mut transcript).unwrap();
}

fn zero_check(c: &mut Criterion) {
    let setup = |num_vars: usize| {
        let mut rng = std_rng();
        let expression = plonk_expression();
        let (polys, challenges) = rand_plonk_assignment::<Fr>(num_vars, &mut rng);
        let ys = [rand_vec(num_vars, &mut rng)];
        (expression, polys, challenges, ys)
    };

    let mut group = c.benchmark_group("zero_check");
    group.sample_size(10);
    for num_vars in 20..24 {
        let (expression, polys, challenges, ys) = setup(num_vars);
        let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
        let id = BenchmarkId::from_parameter(num_vars);
        group.bench_with_input(id, &num_vars, |b, &num_vars| {
            b.iter(|| run(num_vars, virtual_poly.clone()));
        });
    }
}

criterion_group! {
    name = bench;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = zero_check
}
criterion_main!(bench);
