use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use halo2_curves::bn256::{Fr, G1Affine};
use hyperplonk::{
    piop::sum_check::{
        vanilla::{EvaluationsProver, VanillaSumCheck},
        SumCheck, VirtualPolynomial,
    },
    snark::hyperplonk::util::{plonk_expression, rand_plonk_assignment},
    util::{
        test::{rand_vec, std_rng},
        transcript::Keccak256Transcript,
    },
};

type ZeroCheck = VanillaSumCheck<EvaluationsProver<Fr, true>>;

fn run(num_vars: usize, virtual_poly: VirtualPolynomial<Fr>) {
    let mut transcript = Keccak256Transcript::<_, G1Affine>::new(Vec::new());
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
    for num_vars in 20..26 {
        let (expression, polys, challenges, ys) = setup(num_vars);
        let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
        let id = BenchmarkId::from_parameter(num_vars);
        group.bench_with_input(id, &num_vars, |b, &num_vars| {
            b.iter(|| run(num_vars, virtual_poly.clone()));
        });
    }
}

criterion_group!(benches, zero_check);
criterion_main!(benches);
