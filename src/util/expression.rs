use crate::util::Itertools;
use std::{
    cmp::max,
    collections::BTreeSet,
    fmt::Debug,
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rotation(pub i32);

impl Rotation {
    pub fn cur() -> Self {
        Rotation(0)
    }

    pub fn prev() -> Self {
        Rotation(-1)
    }

    pub fn next() -> Self {
        Rotation(1)
    }
}

impl From<i32> for Rotation {
    fn from(rotation: i32) -> Self {
        Self(rotation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Query {
    index: usize,
    poly: usize,
    rotation: Rotation,
}

impl Query {
    pub fn new(index: usize, poly: usize, rotation: impl Into<Rotation>) -> Self {
        Self {
            index,
            poly,
            rotation: rotation.into(),
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn poly(&self) -> usize {
        self.poly
    }

    pub fn rotation(&self) -> Rotation {
        self.rotation
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CommonPolynomial {
    Lagrange(i32),
    EqXY(usize),
}

#[derive(Clone, Debug)]
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

    pub fn random_linear_combine<'a>(
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
        match self {
            Expression::Constant(_) => 0,
            Expression::CommonPolynomial(_) => 1,
            Expression::Polynomial(_) => 1,
            Expression::Challenge(_) => 0,
            Expression::Negated(a) => a.degree(),
            Expression::Sum(a, b) => max(a.degree(), b.degree()),
            Expression::Product(a, b) => a.degree() + b.degree(),
            Expression::Scaled(a, _) => a.degree(),
            Expression::DistributePowers(a, b) => a
                .iter()
                .chain(Some(b.as_ref()))
                .map(Self::degree)
                .max()
                .unwrap_or_default(),
        }
    }

    pub fn used_langrange(&self) -> BTreeSet<i32> {
        self.evaluate(
            &|_| None,
            &|poly| match poly {
                CommonPolynomial::Lagrange(i) => Some(BTreeSet::from_iter([i])),
                CommonPolynomial::EqXY(_) => None,
            },
            &|_| None,
            &|_| None,
            &|a| a,
            &merge_left_right,
            &merge_left_right,
            &|a, _| a,
        )
        .unwrap_or_default()
    }

    pub fn used_query(&self) -> BTreeSet<Query> {
        self.evaluate(
            &|_| None,
            &|_| None,
            &|query| Some(BTreeSet::from_iter([query])),
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
        Expression::Negated(Box::new(self))
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
