use crate::util::Itertools;
use std::{
    collections::BTreeSet,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rotation(pub i32);

impl Rotation {
    pub const fn cur() -> Self {
        Rotation(0)
    }

    pub const fn prev() -> Self {
        Rotation(-1)
    }

    pub const fn next() -> Self {
        Rotation(1)
    }

    pub const fn distance(&self) -> usize {
        self.0.unsigned_abs() as usize
    }
}

impl From<i32> for Rotation {
    fn from(rotation: i32) -> Self {
        Self(rotation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Query {
    poly: usize,
    rotation: Rotation,
}

impl Query {
    pub fn new(poly: usize, rotation: Rotation) -> Self {
        Self { poly, rotation }
    }

    pub fn poly(&self) -> usize {
        self.poly
    }

    pub fn rotation(&self) -> Rotation {
        self.rotation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommonPolynomial {
    Lagrange(i32),
    EqXY(usize),
    Identity(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expression<F> {
    Constant(F),
    CommonPolynomial(CommonPolynomial),
    Polynomial(Query),
    Challenge(usize),
    Negated(Box<Expression<F>>),
    Sum(Box<Expression<F>>, Box<Expression<F>>),
    Product(Box<Expression<F>>, Box<Expression<F>>),
    Scaled(Box<Expression<F>>, F),
    DistributePowers(Vec<Expression<F>>, Box<Expression<F>>),
}

impl<F: Clone> Expression<F> {
    pub fn lagrange(i: i32) -> Self {
        Expression::CommonPolynomial(CommonPolynomial::Lagrange(i))
    }

    pub fn eq_xy(idx: usize) -> Self {
        Expression::CommonPolynomial(CommonPolynomial::EqXY(idx))
    }

    pub fn identity(idx: usize) -> Self {
        Expression::CommonPolynomial(CommonPolynomial::Identity(idx))
    }

    pub fn distribute_powers<'a>(
        exprs: impl IntoIterator<Item = &'a Self> + 'a,
        base: &Self,
    ) -> Self
    where
        F: 'a,
    {
        Expression::DistributePowers(
            exprs.into_iter().cloned().collect_vec(),
            base.clone().into(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T: Clone>(
        &self,
        constant: &impl Fn(F) -> T,
        common_poly: &impl Fn(CommonPolynomial) -> T,
        poly: &impl Fn(Query) -> T,
        challenge: &impl Fn(usize) -> T,
        negated: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
        scaled: &impl Fn(T, F) -> T,
    ) -> T {
        let evaluate = |expr: &Expression<F>| {
            expr.evaluate(
                constant,
                common_poly,
                poly,
                challenge,
                negated,
                sum,
                product,
                scaled,
            )
        };
        match self {
            Expression::Constant(scalar) => constant(scalar.clone()),
            Expression::CommonPolynomial(poly) => common_poly(*poly),
            Expression::Polynomial(query) => poly(*query),
            Expression::Challenge(index) => challenge(*index),
            Expression::Negated(a) => {
                let a = evaluate(a);
                negated(a)
            }
            Expression::Sum(a, b) => {
                let a = evaluate(a);
                let b = evaluate(b);
                sum(a, b)
            }
            Expression::Product(a, b) => {
                let a = evaluate(a);
                let b = evaluate(b);
                product(a, b)
            }
            Expression::Scaled(a, scalar) => {
                let a = evaluate(a);
                scaled(a, scalar.clone())
            }
            Expression::DistributePowers(exprs, scalar) => {
                assert!(!exprs.is_empty());
                if exprs.len() == 1 {
                    return evaluate(exprs.first().unwrap());
                }
                let mut exprs = exprs.iter();
                let first = evaluate(exprs.next().unwrap());
                let scalar = evaluate(scalar);
                exprs.fold(first, |acc, expr| {
                    sum(product(acc, scalar.clone()), evaluate(expr))
                })
            }
        }
    }

    pub fn degree(&self) -> usize {
        self.evaluate(
            &|_| 0,
            &|_| 1,
            &|_| 1,
            &|_| 0,
            &|a| a,
            &|a, b| a.max(b),
            &|a, b| a + b,
            &|a, _| a,
        )
    }

    pub fn used_langrange(&self) -> BTreeSet<i32> {
        self.used_primitive(
            &|poly| match poly {
                CommonPolynomial::Lagrange(i) => i.into(),
                _ => None,
            },
            &|_| None,
        )
    }

    pub fn used_identity(&self) -> BTreeSet<usize> {
        self.used_primitive(
            &|poly| match poly {
                CommonPolynomial::Identity(idx) => idx.into(),
                _ => None,
            },
            &|_| None,
        )
    }

    pub fn used_query(&self) -> BTreeSet<Query> {
        self.used_primitive(&|_| None, &|query| query.into())
    }

    pub fn used_poly(&self) -> BTreeSet<usize> {
        self.used_primitive(&|_| None, &|query| query.poly().into())
    }

    pub fn used_rotation(&self) -> BTreeSet<Rotation> {
        self.used_primitive(&|_| None, &|query| query.rotation().into())
    }

    pub fn max_used_rotation_distance(&self) -> usize {
        self.used_rotation()
            .into_iter()
            .map(|rotation| rotation.0.unsigned_abs())
            .max()
            .unwrap_or_default() as usize
    }

    pub fn used_challenge(&self) -> BTreeSet<usize> {
        self.evaluate(
            &|_| None,
            &|_| None,
            &|_| None,
            &|challenge| Some(BTreeSet::from([challenge])),
            &|a| a,
            &merge_left_right,
            &merge_left_right,
            &|a, _| a,
        )
        .unwrap_or_default()
    }

    fn used_primitive<T: Clone + Ord>(
        &self,
        common_poly: &impl Fn(CommonPolynomial) -> Option<T>,
        poly: &impl Fn(Query) -> Option<T>,
    ) -> BTreeSet<T> {
        self.evaluate(
            &|_| None,
            &|poly| common_poly(poly).map(|t| BTreeSet::from([t])),
            &|query| poly(query).map(|t| BTreeSet::from([t])),
            &|_| None,
            &|a| a,
            &merge_left_right,
            &merge_left_right,
            &|a, _| a,
        )
        .unwrap_or_default()
    }
}

impl<F: Clone> From<Query> for Expression<F> {
    fn from(query: Query) -> Self {
        Self::Polynomial(query)
    }
}

impl<F: Clone> From<CommonPolynomial> for Expression<F> {
    fn from(common_poly: CommonPolynomial) -> Self {
        Self::CommonPolynomial(common_poly)
    }
}

macro_rules! impl_expression_ops {
    ($trait:ident, $op:ident, $variant:ident, $rhs:ty, $rhs_expr:expr) => {
        impl<F: Clone> $trait<$rhs> for Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                Expression::$variant((self).into(), $rhs_expr(rhs).into())
            }
        }
        impl<F: Clone> $trait<$rhs> for &Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_expr(rhs).into())
            }
        }
        impl<F: Clone> $trait<&$rhs> for Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: &$rhs) -> Self::Output {
                Expression::$variant((self).into(), $rhs_expr(rhs.clone()).into())
            }
        }
        impl<F: Clone> $trait<&$rhs> for &Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: &$rhs) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_expr(rhs.clone()).into())
            }
        }
    };
}

impl_expression_ops!(Mul, mul, Product, Expression<F>, std::convert::identity);
impl_expression_ops!(Mul, mul, Scaled, F, std::convert::identity);
impl_expression_ops!(Add, add, Sum, Expression<F>, std::convert::identity);
impl_expression_ops!(Sub, sub, Sum, Expression<F>, Neg::neg);

impl<F: Clone> Neg for Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<F: Clone> Neg for &Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        Expression::Negated(Box::new(self.clone()))
    }
}

impl<F: Clone + Default> Sum for Expression<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item)
            .unwrap_or_else(|| Expression::Constant(F::default()))
    }
}

impl<F: Clone + Default> Product for Expression<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc * item)
            .unwrap_or_else(|| Expression::Constant(F::default()))
    }
}

fn merge_left_right<T: Ord>(
    lhs: Option<BTreeSet<T>>,
    rhs: Option<BTreeSet<T>>,
) -> Option<BTreeSet<T>> {
    match (lhs, rhs) {
        (Some(lhs), None) | (None, Some(lhs)) => Some(lhs),
        (Some(mut lhs), Some(rhs)) => {
            lhs.extend(rhs);
            Some(lhs)
        }
        _ => None,
    }
}
