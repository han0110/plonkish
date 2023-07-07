use crate::{
    poly::Polynomial,
    util::{
        arithmetic::{div_ceil, horner, powers, Field},
        impl_index,
        parallel::{num_threads, parallelize, parallelize_iter},
        Deserialize, Itertools, Serialize,
    },
};
use rand::RngCore;
use std::{
    borrow::Borrow,
    cmp::Ordering::{Equal, Greater, Less},
    fmt::Debug,
    iter::{self, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait Basis: Clone + Copy + Debug {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoefficientBasis;

impl Basis for CoefficientBasis {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnivariatePolynomial<F, B> {
    values: Vec<F>,
    _marker: PhantomData<B>,
}

impl<F> Default for UnivariatePolynomial<F, CoefficientBasis> {
    fn default() -> Self {
        UnivariatePolynomial::zero()
    }
}

impl<F, B> UnivariatePolynomial<F, B> {
    pub const fn zero() -> Self {
        Self {
            values: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }
}

impl<F> UnivariatePolynomial<F, CoefficientBasis> {
    pub fn coeffs(&self) -> &[F] {
        self.values.as_slice()
    }

    pub fn degree(&self) -> usize {
        self.coeffs().len().checked_sub(1).unwrap_or_default()
    }
}

impl<F: Field> Polynomial<F> for UnivariatePolynomial<F, CoefficientBasis> {
    type Point = F;

    fn from_evals(_: Vec<F>) -> Self {
        unimplemented!()
    }

    fn into_evals(self) -> Vec<F> {
        unimplemented!()
    }

    fn evals(&self) -> &[F] {
        unimplemented!()
    }

    fn evaluate(&self, point: &Self::Point) -> F {
        UnivariatePolynomial::evaluate(self, point)
    }
}

impl<F: Field> UnivariatePolynomial<F, CoefficientBasis> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut poly = Self {
            values: coeffs,
            _marker: PhantomData,
        };
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

    pub fn basis<'a>(points: impl IntoIterator<Item = &'a F>, weight: F) -> Self {
        let points = points.into_iter().collect_vec();
        assert!(!points.is_empty());

        let mut buf;
        let mut basis = vec![F::ZERO; points.len() + 1];
        *basis.last_mut().unwrap() = weight;
        for (point, len) in points.into_iter().zip(2..) {
            buf = weight;
            for idx in (0..basis.len() - 1).rev().take(len) {
                buf = basis[idx] - buf * point;
                std::mem::swap(&mut buf, &mut basis[idx]);
            }
        }
        Self::new(basis)
    }

    pub fn evaluate(&self, x: &F) -> F {
        let num_threads = num_threads();
        if self.coeffs().len() < num_threads {
            return horner(&self.values, x);
        }

        let chunk_size = div_ceil(self.coeffs().len(), num_threads);
        let mut results = vec![F::ZERO; num_threads];
        parallelize_iter(
            results
                .iter_mut()
                .zip(self.values.chunks(chunk_size))
                .zip(powers(x.pow_vartime([chunk_size as u64]))),
            |((result, coeffs), scalar)| *result = horner(coeffs, x) * scalar,
        );
        results.iter().fold(F::ZERO, |acc, result| acc + result)
    }

    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        match (self.is_zero(), divisor.is_zero()) {
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
                    for (idx, coeff) in divisor.values.iter().enumerate() {
                        remainder[degree + idx] -= &(quotient_coeff * coeff);
                    }
                    remainder.truncate_leading_zeros();
                }
                (Self::new(quotient), remainder)
            }
        }
    }

    fn truncate_leading_zeros(&mut self) {
        self.values.truncate(
            self.values
                .iter()
                .rev()
                .position(|coeff| !coeff.is_zero_vartime())
                .map(|num_zeros| self.values.len() - num_zeros)
                .unwrap_or_default(),
        );
    }
}

impl<F: Field, B: Basis> Neg for UnivariatePolynomial<F, B> {
    type Output = UnivariatePolynomial<F, B>;

    fn neg(mut self) -> Self::Output {
        self.values.iter_mut().for_each(|coeff| *coeff = -*coeff);
        self
    }
}

impl<'lhs, 'rhs, F: Field> Add<&'rhs UnivariatePolynomial<F, CoefficientBasis>>
    for &'lhs UnivariatePolynomial<F, CoefficientBasis>
{
    type Output = UnivariatePolynomial<F, CoefficientBasis>;

    fn add(
        self,
        rhs: &'rhs UnivariatePolynomial<F, CoefficientBasis>,
    ) -> UnivariatePolynomial<F, CoefficientBasis> {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<'rhs, F: Field> AddAssign<&'rhs UnivariatePolynomial<F, CoefficientBasis>>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn add_assign(&mut self, rhs: &'rhs UnivariatePolynomial<F, CoefficientBasis>) {
        match self.degree().cmp(&rhs.degree()) {
            Less => {
                parallelize(&mut self.values, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs += rhs;
                    }
                });
                self.values
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
        }
    }
}

impl<'rhs, F: Field> AddAssign<(&'rhs F, &'rhs UnivariatePolynomial<F, CoefficientBasis>)>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn add_assign(
        &mut self,
        (scalar, rhs): (&'rhs F, &'rhs UnivariatePolynomial<F, CoefficientBasis>),
    ) {
        if scalar == &F::ONE {
            *self += rhs;
        } else if scalar == &-F::ONE {
            *self -= rhs;
        } else if scalar != &F::ZERO {
            match self.degree().cmp(&rhs.degree()) {
                Less => {
                    parallelize(&mut self.values, |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs += *rhs * scalar;
                        }
                    });
                    self.values
                        .extend(rhs[self.coeffs().len()..].iter().map(|rhs| *rhs * scalar));
                }
                ord @ (Greater | Equal) => {
                    parallelize(&mut self[..rhs.coeffs().len()], |(lhs, start)| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                            *lhs += *rhs * scalar;
                        }
                    });
                    if matches!(ord, Equal) {
                        self.truncate_leading_zeros();
                    }
                }
            }
        }
    }
}

impl<'lhs, 'rhs, F: Field> Sub<&'rhs UnivariatePolynomial<F, CoefficientBasis>>
    for &'lhs UnivariatePolynomial<F, CoefficientBasis>
{
    type Output = UnivariatePolynomial<F, CoefficientBasis>;

    fn sub(
        self,
        rhs: &'rhs UnivariatePolynomial<F, CoefficientBasis>,
    ) -> UnivariatePolynomial<F, CoefficientBasis> {
        let mut output = self.clone();
        output -= rhs;
        output
    }
}

impl<'rhs, F: Field> SubAssign<&'rhs UnivariatePolynomial<F, CoefficientBasis>>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn sub_assign(&mut self, rhs: &'rhs UnivariatePolynomial<F, CoefficientBasis>) {
        match self.degree().cmp(&rhs.degree()) {
            Less => {
                parallelize(&mut self.values, |(lhs, start)| {
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                        *lhs -= rhs;
                    }
                });
                self.values
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

impl<'lhs, 'rhs, F: Field, B: Basis> Mul<&'rhs F> for &'lhs UnivariatePolynomial<F, B> {
    type Output = UnivariatePolynomial<F, B>;

    fn mul(self, rhs: &'rhs F) -> UnivariatePolynomial<F, B> {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<'rhs, F: Field, B: Basis> MulAssign<&'rhs F> for UnivariatePolynomial<F, B> {
    fn mul_assign(&mut self, rhs: &'rhs F) {
        if rhs == &F::ZERO {
            self.values = Vec::new()
        } else if rhs != &F::ONE {
            parallelize(&mut self.values, |(lhs, _)| {
                for lhs in lhs.iter_mut() {
                    *lhs *= rhs;
                }
            });
        }
    }
}

impl<'a, F: Field> Sum<&'a UnivariatePolynomial<F, CoefficientBasis>>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn sum<I: Iterator<Item = &'a UnivariatePolynomial<F, CoefficientBasis>>>(
        mut iter: I,
    ) -> UnivariatePolynomial<F, CoefficientBasis> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => lhs + rhs,
            (Some(lhs), None) => return lhs.clone(),
            _ => return Self::zero(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += poly;
            acc
        })
    }
}

impl<F: Field> Sum<UnivariatePolynomial<F, CoefficientBasis>>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn sum<I: Iterator<Item = UnivariatePolynomial<F, CoefficientBasis>>>(
        iter: I,
    ) -> UnivariatePolynomial<F, CoefficientBasis> {
        iter.reduce(|mut acc, poly| {
            acc += &poly;
            acc
        })
        .unwrap_or_else(Self::zero)
    }
}

impl<'a, F: Field, P: Borrow<UnivariatePolynomial<F, CoefficientBasis>>> Sum<(&'a F, P)>
    for UnivariatePolynomial<F, CoefficientBasis>
{
    fn sum<I: Iterator<Item = (&'a F, P)>>(
        mut iter: I,
    ) -> UnivariatePolynomial<F, CoefficientBasis> {
        let init = match iter.next() {
            Some((scalar, poly)) => {
                let mut poly = poly.borrow().clone();
                poly *= scalar;
                poly
            }
            _ => return Self::zero(),
        };
        iter.fold(init, |mut acc, (scalar, poly)| {
            acc += (scalar, poly.borrow());
            acc
        })
    }
}

impl_index!(@ UnivariatePolynomial<F, CoefficientBasis>, values);
