use crate::util::{
    arithmetic::{
        batch_projective_to_affine, fft, root_of_unity_inv, CurveAffine, Field, PrimeField,
    },
    parallel::parallelize,
    Itertools,
};

mod kzg;

pub use kzg::{
    UnivariateKzg, UnivariateKzgCommitment, UnivariateKzgParam, UnivariateKzgProverParam,
    UnivariateKzgVerifierParam,
};

fn monomial_g1_to_lagrange_g1<C: CurveAffine>(monomial_g1: &[C]) -> Vec<C> {
    assert!(monomial_g1.len().is_power_of_two());

    let k = monomial_g1.len().ilog2() as usize;
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64]);
    let omega_inv = root_of_unity_inv(k);

    let mut lagrange = monomial_g1.iter().map(C::to_curve).collect_vec();
    fft(&mut lagrange, omega_inv, k);
    parallelize(&mut lagrange, |(g, _)| {
        g.iter_mut().for_each(|g| *g *= n_inv)
    });

    batch_projective_to_affine(&lagrange)
}
