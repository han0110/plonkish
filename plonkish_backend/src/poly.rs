use crate::util::arithmetic::Field;
use std::{fmt::Debug, ops::AddAssign};

pub mod multilinear;
pub mod univariate;

pub trait Polynomial<F: Field>: Clone + Debug + for<'a> AddAssign<(&'a F, &'a Self)> {
    type Point: Clone + Debug;

    fn from_evals(evals: Vec<F>) -> Self;

    fn into_evals(self) -> Vec<F>;

    fn evals(&self) -> &[F];

    fn evaluate(&self, point: &Self::Point) -> F;

    #[cfg(any(test, feature = "benchmark"))]
    fn rand(n: usize, rng: &mut impl rand::RngCore) -> Self {
        Self::from_evals(crate::util::test::rand_vec(n, rng))
    }

    #[cfg(any(test, feature = "benchmark"))]
    fn rand_point(k: usize, rng: &mut impl rand::RngCore) -> Self::Point;
}
