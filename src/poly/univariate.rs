use crate::{
    poly::impl_index,
    util::{
        arithmetic::{horner, powers, Field},
        num_threads, parallelize, parallelize_iter,
    },
};
use itertools::Itertools;
use num_integer::Integer;
use rand::RngCore;
use std::{
    cmp::Ordering::{Equal, Greater, Less},
    iter::{self, Sum},
    ops::{Add, AddAssign, Sub, SubAssign},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnivariatePolynomial<F>(Vec<F>);

impl<F> Default for UnivariatePolynomial<F> {
    fn default() -> Self {
        UnivariatePolynomial::zero()
    }
}

impl<F> UnivariatePolynomial<F> {
    pub const fn zero() -> Self {
        Self(Vec::new())
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs().is_empty()
    }

    pub fn coeffs(&self) -> &[F] {
        self.0.as_slice()
    }

    pub fn degree(&self) -> usize {
        self.coeffs().len().checked_sub(1).unwrap_or_default()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.0.iter()
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut poly = Self(coeffs);
        poly.truncate_leading_zeros();
        poly
    }

    pub fn rand(degree: usize, mut rng: impl RngCore) -> Self {
        Self::new(
            iter::repeat_with(|| F::random(&mut rng))
                .take(degree + 1)
                .collect(),
        )
    }

    pub fn evaluate(&self, x: &F) -> F {
        let num_threads = num_threads();
        if self.coeffs().len() * 2 < num_threads {
            return horner(&self.0, x);
        }

        let chunk_size = Integer::div_ceil(&self.coeffs().len(), &num_threads);
        let mut results = vec![F::zero(); num_threads];
        parallelize_iter(
            results
                .iter_mut()
                .zip(self.0.chunks(chunk_size))
                .zip(powers(x.pow_vartime([chunk_size as u64]))),
            |((result, coeffs), scalar)| *result = horner(coeffs, x) * scalar,
        );
        results.iter().fold(F::zero(), |acc, result| acc + result)
    }

    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        match (self.is_zero(), divisor.is_zero()) {
            (_, true) => unreachable!(),
            (true, _) => (Self::zero(), Self::zero()),
            (_, _) => {
                let mut quotient = vec![F::zero(); self.degree() - divisor.degree() + 1];
                let mut remainder = self.clone();
                let divisor_leading_inv = divisor.coeffs().last().unwrap().invert().unwrap();
                while remainder.degree() >= divisor.degree() {
                    let quotient_coeff = divisor_leading_inv * remainder.coeffs().last().unwrap();
                    let degree = remainder.degree() - divisor.degree();
                    quotient[degree] = quotient_coeff;
                    for (idx, coeff) in divisor.0.iter().enumerate() {
                        remainder[degree + idx] -= &(quotient_coeff * coeff);
                    }
                    remainder.truncate_leading_zeros();
                }
                (Self::new(quotient), remainder)
            }
        }
    }

    fn truncate_leading_zeros(&mut self) {
        self.0.truncate(
            self.0
                .iter()
                .rev()
                .position(|coeff| !coeff.is_zero_vartime())
                .map(|num_zeros| self.0.len() - num_zeros)
                .unwrap_or_default(),
        );
    }
}

impl<F: Field> Neg for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn neg(self) -> Self::Output {
        Self(self.0.into_iter().map(|coeff| -coeff).collect())
    }
}

impl<'lhs, 'rhs, F: Field> Add<&'rhs UnivariatePolynomial<F>> for &'lhs UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn add(self, rhs: &'rhs UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<'rhs, F: Field> AddAssign<&'rhs UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    fn add_assign(&mut self, rhs: &'rhs UnivariatePolynomial<F>) {
        match self.degree().cmp(&rhs.degree()) {
            Less => {
                parallelize(&mut self.0, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs += rhs;
                    }
                });
                self.0.extend(rhs[self.coeffs().len()..].iter().cloned());
            }
            ord @ (Greater | Equal) => {
                parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip_eq(rhs[start..].iter()) {
                        *lhs += rhs;
                    }
                });
                if matches!(ord, Equal) {
                    self.truncate_leading_zeros();
                }
            }
        }
    }
}

impl<'lhs, 'rhs, F: Field> Sub<&'rhs UnivariatePolynomial<F>> for &'lhs UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn sub(self, rhs: &'rhs UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output -= rhs;
        output
    }
}

impl<'rhs, F: Field> SubAssign<&'rhs UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    fn sub_assign(&mut self, rhs: &'rhs UnivariatePolynomial<F>) {
        match self.degree().cmp(&rhs.degree()) {
            Less => {
                parallelize(&mut self.0, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs -= rhs;
                    }
                });
                self.0
                    .extend(rhs[self.coeffs().len()..].iter().cloned().map(Neg::neg));
            }
            ord @ (Greater | Equal) => {
                parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip_eq(rhs[start..].iter()) {
                        *lhs -= rhs;
                    }
                });
                if matches!(ord, Equal) {
                    self.truncate_leading_zeros();
                }
            }
        }
    }
}

impl<F: Field> AddAssign<F> for UnivariatePolynomial<F> {
    fn add_assign(&mut self, rhs: F) {
        self.0[0] += &rhs;
    }
}

impl<F: Field> SubAssign<F> for UnivariatePolynomial<F> {
    fn sub_assign(&mut self, rhs: F) {
        self.0[0] -= &rhs;
    }
}

impl<'lhs, F: Field> Mul<F> for &'lhs UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn mul(self, rhs: F) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<F: Field> MulAssign<F> for UnivariatePolynomial<F> {
    fn mul_assign(&mut self, rhs: F) {
        parallelize(&mut self.0, |(lhs, _)| {
            for lhs in lhs.iter_mut() {
                *lhs *= &rhs;
            }
        });
    }
}

impl<F: Field> Sum<UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    fn sum<I: Iterator<Item = UnivariatePolynomial<F>>>(mut iter: I) -> UnivariatePolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => &lhs + &rhs,
            (Some(lhs), None) => return lhs,
            _ => unreachable!(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += &poly;
            acc
        })
    }
}

impl<'a, F: Field> Sum<&'a UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    fn sum<I: Iterator<Item = &'a UnivariatePolynomial<F>>>(
        mut iter: I,
    ) -> UnivariatePolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => lhs + rhs,
            (Some(lhs), None) => return lhs.clone(),
            _ => unreachable!(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += poly;
            acc
        })
    }
}

impl_index!(
    UnivariatePolynomial, 0,
    [
        usize => F,
        Range<usize> => [F],
        RangeFrom<usize> => [F],
        RangeFull => [F],
        RangeInclusive<usize> => [F],
        RangeTo<usize> => [F],
        RangeToInclusive<usize> => [F],
    ]
);
