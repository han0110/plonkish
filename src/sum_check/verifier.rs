use crate::{
    sum_check::VirtualPolynomialInfo,
    util::{
        arithmetic::{BatchInvert, PrimeField},
        parallelize, Itertools,
    },
    Error,
};
use std::iter;

pub fn consistency_check<F: PrimeField>(
    virtual_poly_info: &VirtualPolynomialInfo<F>,
    rounds: &[(Vec<F>, F)],
    sum: F,
) -> Result<F, Error> {
    let challenge_evals = evaluate_at_challenge(&virtual_poly_info.sample_points(), rounds);
    for (idx, ((sample_evals, _), expected_sum)) in rounds
        .iter()
        .zip(iter::once(&sum).chain(challenge_evals.iter()))
        .enumerate()
    {
        if sample_evals[0] + sample_evals[1] != *expected_sum {
            return Err(Error::InvalidSumcheck(format!(
                "Consistency check failure at round {idx}"
            )));
        }
    }
    Ok(*challenge_evals.last().unwrap())
}

fn evaluate_at_challenge<F: PrimeField>(points: &[usize], rounds: &[(Vec<F>, F)]) -> Vec<F> {
    let points = points
        .iter()
        .map(|point| F::from(*point as u64))
        .collect_vec();
    let weights = barycentric_weights(&points);
    let mut challenge_evals = vec![F::zero(); rounds.len()];
    parallelize(&mut challenge_evals, |(challenge_evals, start)| {
        let (coeffs, sum_invs) = {
            let mut coeffs = rounds[start..start + challenge_evals.len()]
                .iter()
                .map(|(_, challenge)| points.iter().map(|point| *challenge - point).collect_vec())
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
        for (((challenge_eval, (sample_evals, _)), coeffs), sum_inv) in challenge_evals
            .iter_mut()
            .zip(&rounds[start..])
            .zip(&coeffs)
            .zip(&sum_invs)
        {
            *challenge_eval = coeffs
                .iter()
                .zip(sample_evals)
                .map(|(coeff, sample_eval)| *coeff * sample_eval)
                .reduce(|acc, item| acc + &item)
                .unwrap_or_default()
                * sum_inv;
        }
    });
    challenge_evals
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
