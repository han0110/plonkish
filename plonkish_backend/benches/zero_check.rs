use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use halo2_curves::bn256::Fr;
use plonkish_backend::{
    backend::hyperplonk::util::{rand_vanilla_plonk_assignment, vanilla_plonk_expression},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    util::{
        arithmetic::Field,
        expression::rotate::BinaryField,
        test::{rand_vec, seeded_std_rng},
        transcript::Keccak256Transcript,
    },
};
use pprof::criterion::{Output, PProfProfiler};

type ZeroCheck = ClassicSumCheck<EvaluationsProver<Fr>, BinaryField>;

fn run(num_vars: usize, virtual_poly: VirtualPolynomial<Fr>) {
    let mut transcript = Keccak256Transcript::<Vec<u8>>::default();
    ZeroCheck::prove(&(), num_vars, virtual_poly, Fr::ZERO, &mut transcript).unwrap();
}

fn zero_check(c: &mut Criterion) {
    let setup = |num_vars: usize| {
        let expression = vanilla_plonk_expression(num_vars);
        let (polys, challenges) = rand_vanilla_plonk_assignment::<Fr, BinaryField>(
            num_vars,
            seeded_std_rng(),
            seeded_std_rng(),
        );
        let ys = [rand_vec(num_vars, seeded_std_rng())];
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
