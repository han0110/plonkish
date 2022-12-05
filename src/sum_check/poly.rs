use crate::{
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{product, BooleanHypercube, Field, PrimeField},
        expression::{CommonPolynomial, Expression, Query},
        Itertools,
    },
};
use std::{collections::HashMap, iter};

#[derive(Debug)]
pub struct VirtualPolynomialInfo<F> {
    num_vars: usize,
    expression: Expression<F>,
}

impl<F: Field> VirtualPolynomialInfo<F> {
    pub fn new(num_vars: usize, expression: Expression<F>) -> Self {
        Self {
            num_vars,
            expression,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn expression(&self) -> &Expression<F> {
        &self.expression
    }

    pub fn degree(&self) -> usize {
        self.expression.degree()
    }

    pub fn sample_points(&self) -> Vec<usize> {
        (0..self.degree() + 1).collect()
    }

    pub fn evaluate(
        &self,
        evals: &HashMap<Query, F>,
        challenges: &[F],
        ys: &[Vec<F>],
        x: &[F],
    ) -> F {
        let idx_map = BooleanHypercube::new(self.num_vars).idx_map();
        let lagranges = self
            .expression()
            .used_langrange()
            .into_iter()
            .map(|i| {
                let b = idx_map[i.rem_euclid(self.num_vars() as i32) as usize];
                (i, lagrange_eval(x, b as usize))
            })
            .collect::<HashMap<_, _>>();
        let eq_xys = ys.iter().map(|y| eq_xy_eval(x, y)).collect_vec();
        self.expression().evaluate(
            &|scalar| scalar,
            &|poly| match poly {
                CommonPolynomial::Lagrange(i) => lagranges[&i],
                CommonPolynomial::EqXY(idx) => eq_xys[idx],
            },
            &|query| evals[&query],
            &|idx| challenges[idx],
            &|scalar| -scalar,
            &|lhs, rhs| lhs + &rhs,
            &|lhs, rhs| lhs * &rhs,
            &|value, scalar| scalar * value,
        )
    }
}

#[derive(Debug)]
pub struct VirtualPolynomial<'a, F> {
    pub(crate) info: &'a VirtualPolynomialInfo<F>,
    pub(crate) polys: Vec<&'a MultilinearPolynomial<F>>,
    pub(crate) challenges: Vec<F>,
    pub(crate) ys: Vec<Vec<F>>,
}

impl<'a, F: PrimeField> VirtualPolynomial<'a, F> {
    pub fn new(
        info: &'a VirtualPolynomialInfo<F>,
        polys: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
        challenges: Vec<F>,
        ys: Vec<Vec<F>>,
    ) -> Self {
        Self {
            info,
            polys: polys.into_iter().collect(),
            challenges,
            ys,
        }
    }
}

pub fn lagrange_eval<F: Field>(x: &[F], b: usize) -> F {
    assert!(!x.is_empty());

    product(
        x.iter()
            .zip(iter::successors(Some(1), |b| Some(b << 1)))
            .map(
                |(x_i, mask)| {
                    if b & mask == 0 {
                        F::one() - x_i
                    } else {
                        *x_i
                    }
                },
            ),
    )
}

pub fn eq_xy_eval<F: Field>(x: &[F], y: &[F]) -> F {
    assert!(!x.is_empty());
    assert_eq!(x.len(), y.len());

    product(
        x.iter()
            .zip(y)
            .map(|(x_i, y_i)| (*x_i * y_i).double() + F::one() - x_i - y_i),
    )
}
