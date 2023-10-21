use crate::{
    pcs::Additive,
    poly::{univariate::UnivariateBasis::*, Polynomial},
    util::{
        arithmetic::{div_ceil, horner, powers, Field},
        impl_index, izip_eq,
        parallel::{num_threads, parallelize, parallelize_iter},
        Deserialize, Itertools, Serialize,
    },
};
use std::{
    borrow::Borrow,
    cmp::Ordering::{Equal, Greater, Less},
    fmt::Debug,
    iter::Sum,
    mem,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnivariateBasis {
    Monomial,
    Lagrange,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnivariatePolynomial<F> {
    basis: UnivariateBasis,
    coeffs: Vec<F>,
}

impl<F> Default for UnivariatePolynomial<F> {
    fn default() -> Self {
        UnivariatePolynomial::zero()
    }
}

impl<F: Field> Additive<F> for UnivariatePolynomial<F> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a F>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self
    where
        Self: 'b,
    {
        izip_eq!(scalars, bases).sum()
    }
}

impl<F> UnivariatePolynomial<F> {
    pub const fn zero() -> Self {
        Self {
            basis: Monomial,
            coeffs: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.coeffs.iter()
    }

    pub fn coeffs(&self) -> &[F] {
        self.coeffs.as_slice()
    }

    pub fn coeffs_mut(&mut self) -> &mut [F] {
        self.coeffs.as_mut_slice()
    }

    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
    }

    pub fn degree(&self) -> usize {
        assert_eq!(self.basis, Monomial);

        self.coeffs().len().checked_sub(1).unwrap_or_default()
    }
}

impl<F: Field> Polynomial<F> for UnivariatePolynomial<F> {
    type Point = F;

    fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    fn evaluate(&self, point: &Self::Point) -> F {
        UnivariatePolynomial::evaluate(self, point)
    }

    #[cfg(any(test, feature = "benchmark"))]
    fn rand(n: usize, rng: impl rand::RngCore) -> Self {
        Self::monomial(crate::util::test::rand_vec(n, rng))
    }

    #[cfg(any(test, feature = "benchmark"))]
    fn rand_point(_: usize, rng: impl rand::RngCore) -> F {
        F::random(rng)
    }

    #[cfg(any(test, feature = "benchmark"))]
    fn squeeze_point(
        _: usize,
        transcript: &mut impl crate::util::transcript::FieldTranscript<F>,
    ) -> Self::Point {
        transcript.squeeze_challenge()
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    pub fn monomial(coeffs: Vec<F>) -> Self {
        let mut poly = Self {
            basis: Monomial,
            coeffs,
        };
        poly.truncate_leading_zeros();
        poly
    }

    pub fn lagrange(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());

        Self {
            basis: Lagrange,
            coeffs,
        }
    }

    pub fn vanishing<'a>(points: impl IntoIterator<Item = &'a F>, scalar: F) -> Self {
        let points = points.into_iter().collect_vec();
        assert!(!points.is_empty());

        let mut buf;
        let mut coeffs = vec![F::ZERO; points.len() + 1];
        *coeffs.last_mut().unwrap() = scalar;
        for (point, len) in points.into_iter().zip(2..) {
            buf = scalar;
            for idx in (0..coeffs.len() - 1).rev().take(len) {
                buf = coeffs[idx] - buf * point;
                mem::swap(&mut buf, &mut coeffs[idx]);
            }
        }
        Self::monomial(coeffs)
    }

    pub fn basis(&self) -> UnivariateBasis {
        self.basis
    }

    pub fn evaluate(&self, x: &F) -> F {
        assert_eq!(self.basis, Monomial);

        let num_threads = num_threads();
        if self.coeffs().len() < num_threads {
            return horner(&self.coeffs, x);
        }

        let chunk_size = div_ceil(self.coeffs().len(), num_threads);
        let mut results = vec![F::ZERO; num_threads];
        parallelize_iter(
            results
                .iter_mut()
                .zip(self.coeffs.chunks(chunk_size))
                .zip(powers(x.pow_vartime([chunk_size as u64]))),
            |((result, coeffs), scalar)| *result = horner(coeffs, x) * scalar,
        );
        results.iter().fold(F::ZERO, |acc, result| acc + result)
    }

    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert_eq!(self.basis, Monomial);

        match (self.is_empty(), divisor.is_empty()) {
            (_, true) => unreachable!(),
            (true, _) => (Self::zero(), Self::zero()),
            (_, _) => {
                if self.degree() < divisor.degree() {
                    return (Self::zero(), self.clone());
                }

                let mut quotient = vec![F::ZERO; self.degree() - divisor.degree() + 1];
                let mut remainder = self.clone();
                let divisor_leading_inv = divisor.coeffs().last().unwrap().invert().unwrap();
                while remainder.degree() >= divisor.degree() {
                    let quotient_coeff = divisor_leading_inv * remainder.coeffs().last().unwrap();
                    let degree = remainder.degree() - divisor.degree();
                    quotient[degree] = quotient_coeff;
                    for (idx, coeff) in divisor.coeffs.iter().enumerate() {
                        remainder[degree + idx] -= &(quotient_coeff * coeff);
                    }
                    remainder.truncate_leading_zeros();
                }
                (Self::monomial(quotient), remainder)
            }
        }
    }

    fn truncate_leading_zeros(&mut self) {
        assert_eq!(self.basis, Monomial);

        self.coeffs.truncate(
            self.coeffs
                .iter()
                .rev()
                .position(|coeff| !coeff.is_zero_vartime())
                .map(|num_zeros| self.coeffs.len() - num_zeros)
                .unwrap_or_default(),
        );
    }
}

impl<F: Field> Neg for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn neg(mut self) -> Self::Output {
        self.coeffs.iter_mut().for_each(|coeff| *coeff = -*coeff);
        self
    }
}

impl<F: Field, P: Borrow<UnivariatePolynomial<F>>> Add<P> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn add(self, rhs: P) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<F: Field, P: Borrow<UnivariatePolynomial<F>>> AddAssign<P> for UnivariatePolynomial<F> {
    fn add_assign(&mut self, rhs: P) {
        let rhs = rhs.borrow();
        assert_eq!(self.basis, rhs.basis);

        match self.basis {
            Monomial => match self.degree().cmp(&rhs.degree()) {
                Less => {
                    parallelize(&mut self.coeffs, |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs += rhs;
                        }
                    });
                    self.coeffs
                        .extend(rhs[self.coeffs().len()..].iter().cloned());
                }
                ord @ (Greater | Equal) => {
                    parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs += rhs;
                        }
                    });
                    if matches!(ord, Equal) {
                        self.truncate_leading_zeros();
                    }
                }
            },
            Lagrange => {
                assert_eq!(self.coeffs.len(), rhs.coeffs.len());

                parallelize(&mut self.coeffs, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs += rhs;
                    }
                });
            }
        }
    }
}

impl<F: Field, BF: Borrow<F>, P: Borrow<UnivariatePolynomial<F>>> AddAssign<(BF, P)>
    for UnivariatePolynomial<F>
{
    fn add_assign(&mut self, (scalar, rhs): (BF, P)) {
        let (scalar, rhs) = (scalar.borrow(), rhs.borrow());
        assert_eq!(self.basis, rhs.basis);

        if scalar == &F::ONE {
            *self += rhs;
        } else if scalar == &-F::ONE {
            *self -= rhs;
        } else if scalar != &F::ZERO {
            match self.basis {
                Monomial => match self.degree().cmp(&rhs.degree()) {
                    Less => {
                        parallelize(&mut self.coeffs, |(lhs, start)| {
                            let scalar = *scalar;
                            for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                                *lhs += scalar * rhs;
                            }
                        });
                        self.coeffs
                            .extend(rhs[self.coeffs().len()..].iter().map(|rhs| *rhs * scalar));
                    }
                    ord @ (Greater | Equal) => {
                        parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                            let scalar = *scalar;
                            for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                                *lhs += scalar * rhs;
                            }
                        });
                        if matches!(ord, Equal) {
                            self.truncate_leading_zeros();
                        }
                    }
                },
                Lagrange => {
                    assert_eq!(self.coeffs.len(), rhs.coeffs.len());

                    parallelize(&mut self.coeffs, |(lhs, start)| {
                        let scalar = *scalar;
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs += scalar * rhs;
                        }
                    });
                }
            }
        }
    }
}

impl<F: Field, P: Borrow<UnivariatePolynomial<F>>> Sub<P> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn sub(self, rhs: P) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output -= rhs;
        output
    }
}

impl<F: Field, P: Borrow<UnivariatePolynomial<F>>> SubAssign<P> for UnivariatePolynomial<F> {
    fn sub_assign(&mut self, rhs: P) {
        let rhs = rhs.borrow();
        assert_eq!(self.basis, rhs.basis);

        match self.basis {
            Monomial => match self.degree().cmp(&rhs.degree()) {
                Less => {
                    parallelize(&mut self.coeffs, |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs -= rhs;
                        }
                    });
                    self.coeffs
                        .extend(rhs[self.coeffs().len()..].iter().cloned().map(Neg::neg));
                }
                ord @ (Greater | Equal) => {
                    parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs -= rhs;
                        }
                    });
                    if matches!(ord, Equal) {
                        self.truncate_leading_zeros();
                    }
                }
            },
            Lagrange => {
                assert_eq!(self.coeffs.len(), rhs.coeffs.len());

                parallelize(&mut self.coeffs, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs -= rhs;
                    }
                });
            }
        }
    }
}

impl<F: Field, BF: Borrow<F>, P: Borrow<UnivariatePolynomial<F>>> SubAssign<(BF, P)>
    for UnivariatePolynomial<F>
{
    fn sub_assign(&mut self, (scalar, rhs): (BF, P)) {
        *self += (-*scalar.borrow(), rhs);
    }
}

impl<F: Field, BF: Borrow<F>> Mul<BF> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn mul(self, rhs: BF) -> UnivariatePolynomial<F> {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<F: Field, BF: Borrow<F>> MulAssign<BF> for UnivariatePolynomial<F> {
    fn mul_assign(&mut self, rhs: BF) {
        let rhs = rhs.borrow();
        if rhs == &F::ZERO {
            match self.basis {
                Monomial => self.coeffs.clear(),
                Lagrange => self.coeffs.fill(F::ZERO),
            }
        } else if rhs != &F::ONE {
            parallelize(&mut self.coeffs, |(lhs, _)| {
                for lhs in lhs.iter_mut() {
                    *lhs *= rhs;
                }
            });
        }
    }
}

impl<F: Field, P: Borrow<UnivariatePolynomial<F>>> Sum<P> for UnivariatePolynomial<F> {
    fn sum<I: Iterator<Item = P>>(mut iter: I) -> UnivariatePolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => lhs.borrow() + rhs.borrow(),
            (Some(lhs), None) => return lhs.borrow().clone(),
            _ => return Self::zero(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += poly.borrow();
            acc
        })
    }
}

impl<F: Field, BF: Borrow<F>, P: Borrow<UnivariatePolynomial<F>>> Sum<(BF, P)>
    for UnivariatePolynomial<F>
{
    fn sum<I: Iterator<Item = (BF, P)>>(mut iter: I) -> UnivariatePolynomial<F> {
        let init = match iter.next() {
            Some((scalar, poly)) => {
                let mut poly = poly.borrow().clone();
                poly *= scalar.borrow();
                poly
            }
            _ => return Self::zero(),
        };
        iter.fold(init, |mut acc, (scalar, poly)| {
            acc += (scalar.borrow(), poly.borrow());
            acc
        })
    }
}

impl_index!(UnivariatePolynomial, coeffs);
