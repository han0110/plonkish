use crate::util::arithmetic::Field;
use std::{fmt::Debug, ops::AddAssign};

pub mod multilinear;
pub mod univariate;

pub trait Polynomial<F: Field>:
    Clone + Debug + Default + for<'a> AddAssign<(&'a F, &'a Self)>
{
    type Basis: Copy + Debug;
    type Point: Clone + Debug;

    fn new(basis: Self::Basis, coeffs: Vec<F>) -> Self;

    fn basis(&self) -> Self::Basis;

    fn coeffs(&self) -> &[F];

    fn evaluate(&self, point: &Self::Point) -> F;

    #[cfg(any(test, feature = "benchmark"))]
    fn rand(n: usize, rng: impl rand::RngCore) -> Self;

    #[cfg(any(test, feature = "benchmark"))]
    fn rand_point(k: usize, rng: impl rand::RngCore) -> Self::Point;
}
