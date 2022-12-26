use crate::{
    util::{
        arithmetic::{inner_product, product, BatchInvert, BooleanHypercube, PrimeField},
        expression::{CommonPolynomial, Expression, Query},
        BitIndex, Itertools,
    },
    Error,
};
use std::{collections::HashMap, iter};

#[derive(Clone, Debug)]
pub struct VirtualPolynomialInfo<F> {
    expression: Expression<F>,
}

impl<F: PrimeField> VirtualPolynomialInfo<F> {
    pub fn new(expression: Expression<F>) -> Self {
        Self { expression }
    }

    pub fn expression(&self) -> &Expression<F> {
        &self.expression
    }

    pub fn degree(&self) -> usize {
        self.expression.degree()
    }

    pub fn sample_points(&self) -> Vec<u64> {
        (0..self.degree() as u64 + 1).collect()
    }

    pub fn evaluate(
        &self,
        num_vars: usize,
        evals: &HashMap<Query, F>,
        challenges: &[F],
        ys: &[&[F]],
        x: &[F],
    ) -> F {
        assert!(num_vars > 0 && self.expression.max_used_rotation_distance() <= num_vars);
        let lagranges = {
            let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
            self.expression()
                .used_langrange()
                .into_iter()
                .map(|i| {
                    let b = bh[i.rem_euclid(1 << num_vars as i32) as usize];
                    (i, lagrange_eval(x, b))
                })
                .collect::<HashMap<_, _>>()
        };
        let eq_xys = ys.iter().map(|y| eq_xy_eval(x, y)).collect_vec();
        let identity = identity_eval(x);
        self.expression().evaluate(
            &|scalar| scalar,
            &|poly| match poly {
                CommonPolynomial::Lagrange(i) => lagranges[&i],
                CommonPolynomial::EqXY(idx) => eq_xys[idx],
                CommonPolynomial::Identity(idx) => F::from((idx << num_vars) as u64) + identity,
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

pub fn lagrange_eval<F: PrimeField>(x: &[F], b: usize) -> F {
    assert!(!x.is_empty());

    product(x.iter().enumerate().map(
        |(idx, x_i)| {
            if b.nth_bit(idx) {
                *x_i
            } else {
                F::one() - x_i
            }
        },
    ))
}

pub fn eq_xy_eval<F: PrimeField>(x: &[F], y: &[F]) -> F {
    assert!(!x.is_empty());
    assert_eq!(x.len(), y.len());

    product(
        x.iter()
            .zip(y)
            .map(|(x_i, y_i)| (*x_i * y_i).double() + F::one() - x_i - y_i),
    )
}

fn identity_eval<F: PrimeField>(x: &[F]) -> F {
    inner_product(x, &(0..x.len()).map(|idx| F::from(1 << idx)).collect_vec())
}

pub fn verify_consistency<F: PrimeField>(
    virtual_poly_info: &VirtualPolynomialInfo<F>,
    sum: F,
    rounds: &[(Vec<F>, F)],
) -> Result<(F, Vec<F>), Error> {
    let (evals, challenges) = rounds
        .iter()
        .map(|(evals, challenge)| (evals.as_slice(), *challenge))
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let challenge_evals =
        evaluate_at_challenge(&virtual_poly_info.sample_points(), &evals, &challenges);
    for (idx, ((sample_evals, _), expected_sum)) in rounds
        .iter()
        .zip(iter::once(&sum).chain(challenge_evals.iter()))
        .enumerate()
    {
        let sum = sample_evals[0] + sample_evals[1];
        if sum != *expected_sum {
            let msg = if idx == 0 {
                format!("Expect sum {expected_sum:?} but get {sum:?}")
            } else {
                format!("Consistency failure at round {idx}")
            };
            return Err(Error::InvalidSumcheck(msg));
        }
    }
    Ok((*challenge_evals.last().unwrap(), challenges))
}

pub(super) fn evaluate_at_challenge<F: PrimeField>(
    points: &[u64],
    evals: &[&[F]],
    challenges: &[F],
) -> Vec<F> {
    let points = points.iter().cloned().map(F::from).collect_vec();
    let weights = barycentric_weights(&points);
    let (coeffs, sum_invs) = {
        let mut coeffs = challenges
            .iter()
            .map(|challenge| points.iter().map(|point| *challenge - point).collect_vec())
            .collect_vec();
        coeffs
            .iter_mut()
            .flat_map(|coeffs| coeffs.iter_mut())
            .batch_invert();
        coeffs.iter_mut().for_each(|coeffs| {
            coeffs
                .iter_mut()
                .zip(weights.iter())
                .for_each(|(coeff, weight)| {
                    *coeff *= weight;
                });
        });
        let mut sum_invs = coeffs
            .iter()
            .map(|coeffs| coeffs.iter().fold(F::zero(), |sum, coeff| sum + coeff))
            .collect_vec();
        sum_invs.iter_mut().batch_invert();
        (coeffs, sum_invs)
    };
    evals
        .iter()
        .zip(&coeffs)
        .zip(&sum_invs)
        .map(|((evals, coeffs), sum_inv)| {
            coeffs
                .iter()
                .zip(evals.iter())
                .map(|(coeff, eval)| *coeff * eval)
                .reduce(|acc, item| acc + &item)
                .unwrap_or_default()
                * sum_inv
        })
        .collect()
}

fn barycentric_weights<F: PrimeField>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .map(|(_, point_i)| (*point_j - point_i))
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(|| F::one())
        })
        .collect_vec();
    weights.iter_mut().batch_invert();
    weights
}
