use crate::{
    backend::{
        hyperplonk::{
            folding::sangria::{SangriaProverParam, SangriaVerifierParam},
            preprocessor::permutation_constraints,
            HyperPlonk,
        },
        PlonkishBackend, PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, PrimeField},
        chain,
        expression::{
            relaxed::{cross_term_expressions, products, relaxed_expression},
            Expression, Query, Rotation,
        },
        Itertools,
    },
    Error,
};
use std::{
    array,
    borrow::Cow,
    collections::{BTreeSet, HashSet},
    hash::Hash,
    iter,
};

pub(crate) fn batch_size<F: PrimeField>(circuit_info: &PlonkishCircuitInfo<F>) -> usize {
    let num_lookups = circuit_info.lookups.len();
    let num_permutation_polys = circuit_info.permutation_polys().len();
    chain![
        [circuit_info.preprocess_polys.len() + circuit_info.permutation_polys().len()],
        circuit_info.num_witness_polys.clone(),
        [num_lookups],
        [2 * num_lookups + div_ceil(num_permutation_polys, max_degree(circuit_info, None) - 1)],
        [1],
    ]
    .sum()
}

#[allow(clippy::type_complexity)]
pub(super) fn preprocess<F, Pcs>(
    param: &Pcs::Param,
    circuit_info: &PlonkishCircuitInfo<F>,
) -> Result<(SangriaProverParam<F, Pcs>, SangriaVerifierParam<F, Pcs>), Error>
where
    F: PrimeField + Ord + Hash,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    let challenge_offset = circuit_info.num_challenges.iter().sum::<usize>();
    let max_lookup_width = circuit_info.lookups.iter().map(Vec::len).max().unwrap_or(0);
    let num_theta_primes = max_lookup_width.checked_sub(1).unwrap_or_default();
    let theta_primes = (challenge_offset..)
        .take(num_theta_primes)
        .map(Expression::<F>::Challenge)
        .collect_vec();
    let beta_prime = &Expression::<F>::Challenge(challenge_offset + num_theta_primes);

    let (lookup_constraints, lookup_zero_checks) =
        lookup_constraints(circuit_info, &theta_primes, beta_prime);

    let max_degree = iter::empty()
        .chain(circuit_info.constraints.iter())
        .chain(lookup_constraints.iter())
        .map(Expression::degree)
        .chain(circuit_info.max_degree)
        .chain(Some(2))
        .max()
        .unwrap();

    let permutation_polys = circuit_info.permutation_polys();
    let preprocess_polys = iter::empty()
        .chain((circuit_info.num_instances.len()..).take(circuit_info.preprocess_polys.len()))
        .chain((circuit_info.num_poly()..).take(permutation_polys.len()))
        .collect();

    let num_constraints = circuit_info.constraints.len() + lookup_constraints.len();
    let num_alpha_primes = num_constraints.checked_sub(1).unwrap_or_default();

    let products = {
        let mut constraints = iter::empty()
            .chain(circuit_info.constraints.iter())
            .chain(lookup_constraints.iter())
            .collect_vec();
        let folding_degrees = constraints
            .iter()
            .map(|constraint| folding_degree(&preprocess_polys, constraint))
            .enumerate()
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .collect_vec();
        if let &[a, b, ..] = &folding_degrees[..] {
            if a.1 != b.1 {
                constraints.swap(0, a.0);
            }
        }
        let constraint = iter::empty()
            .chain(constraints.first().cloned().cloned())
            .chain(
                constraints
                    .into_iter()
                    .skip(1)
                    .zip((challenge_offset + num_theta_primes + 1..).map(Expression::Challenge))
                    .map(|(constraint, challenge)| constraint * challenge),
            )
            .sum();
        products(&preprocess_polys, &constraint)
    };

    let num_witness_polys = circuit_info.num_witness_polys.iter().sum::<usize>();
    let witness_poly_offset =
        circuit_info.num_instances.len() + circuit_info.preprocess_polys.len();
    let internal_witness_poly_offset =
        witness_poly_offset + num_witness_polys + permutation_polys.len();

    let folding_polys = iter::empty()
        .chain(0..circuit_info.num_instances.len())
        .chain((witness_poly_offset..).take(num_witness_polys))
        .chain((internal_witness_poly_offset..).take(3 * circuit_info.lookups.len()))
        .collect::<BTreeSet<_>>();
    let num_folding_wintess_polys = num_witness_polys + 3 * circuit_info.lookups.len();
    let num_folding_challenges = challenge_offset + num_theta_primes + 1 + num_alpha_primes;

    let cross_term_expressions = cross_term_expressions(
        circuit_info.num_instances.len(),
        circuit_info.preprocess_polys.len(),
        folding_polys,
        num_folding_challenges,
        &products,
    );
    let num_cross_terms = cross_term_expressions.len();

    let [beta, gamma, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(num_folding_challenges + idx));
    let (num_chunks, permutation_constraints) = permutation_constraints(
        circuit_info,
        max_degree,
        beta,
        gamma,
        3 * circuit_info.lookups.len(),
    );

    let relexed_constraint = {
        let u = num_folding_challenges + 3;
        let e_poly = circuit_info.num_poly()
            + permutation_polys.len()
            + circuit_info.lookups.len() * 3
            + num_chunks;
        relaxed_expression(&products, u)
            - Expression::Polynomial(Query::new(e_poly, Rotation::cur()))
    };
    let zero_check_on_every_row = Expression::distribute_powers(
        iter::empty()
            .chain(Some(&relexed_constraint))
            .chain(permutation_constraints.iter()),
        alpha,
    ) * Expression::eq_xy(0);
    let zero_check_expression = Expression::distribute_powers(
        iter::empty()
            .chain(lookup_zero_checks.iter())
            .chain(Some(&zero_check_on_every_row)),
        alpha,
    );

    let (mut pp, mut vp) = HyperPlonk::preprocess(param, circuit_info)?;
    let (pcs_pp, pcs_vp) = Pcs::trim(param, 1 << circuit_info.k, batch_size(circuit_info))?;
    pp.pcs = pcs_pp;
    vp.pcs = pcs_vp;

    Ok((
        SangriaProverParam {
            pp,
            num_theta_primes,
            num_alpha_primes,
            num_folding_wintess_polys,
            num_folding_challenges,
            cross_term_expressions,
            zero_check_expression: zero_check_expression.clone(),
        },
        SangriaVerifierParam {
            vp,
            num_theta_primes,
            num_alpha_primes,
            num_folding_wintess_polys,
            num_folding_challenges,
            num_cross_terms,
            zero_check_expression,
        },
    ))
}

pub(crate) fn max_degree<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    lookup_constraints: Option<&[Expression<F>]>,
) -> usize {
    let lookup_constraints = lookup_constraints.map(Cow::Borrowed).unwrap_or_else(|| {
        let n = circuit_info.lookups.iter().map(Vec::len).max().unwrap_or(1);
        let dummy_challenges = vec![Expression::zero(); n];
        Cow::Owned(
            self::lookup_constraints(circuit_info, &dummy_challenges, &dummy_challenges[0]).0,
        )
    });
    iter::empty()
        .chain(circuit_info.constraints.iter().map(Expression::degree))
        .chain(lookup_constraints.iter().map(Expression::degree))
        .chain(circuit_info.max_degree)
        .chain(Some(2))
        .max()
        .unwrap()
}

pub(crate) fn lookup_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    theta_primes: &[Expression<F>],
    beta_prime: &Expression<F>,
) -> (Vec<Expression<F>>, Vec<Expression<F>>) {
    let one = &Expression::one();
    let m_offset = circuit_info.num_poly() + circuit_info.permutation_polys().len();
    let h_offset = m_offset + circuit_info.lookups.len();
    let constraints = circuit_info
        .lookups
        .iter()
        .zip(m_offset..)
        .zip((h_offset..).step_by(2))
        .flat_map(|((lookup, m), h)| {
            let [m, h_input, h_table] = &[m, h, h + 1]
                .map(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
            let (inputs, tables) = lookup
                .iter()
                .map(|(input, table)| (input, table))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let [input, table] = &[inputs, tables].map(|exprs| {
                iter::empty()
                    .chain(exprs.first().cloned().cloned())
                    .chain(
                        exprs
                            .into_iter()
                            .skip(1)
                            .zip(theta_primes)
                            .map(|(expr, theta_prime)| expr * theta_prime),
                    )
                    .sum::<Expression<_>>()
            });
            [
                h_input * (input + beta_prime) - one,
                h_table * (table + beta_prime) - m,
            ]
        })
        .collect_vec();
    let sum_check = (h_offset..)
        .step_by(2)
        .take(circuit_info.lookups.len())
        .map(|h| {
            let [h_input, h_table] = &[h, h + 1]
                .map(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
            h_input - h_table
        })
        .collect_vec();
    (constraints, sum_check)
}

pub(crate) fn folding_degree<F: PrimeField>(
    preprocess_polys: &HashSet<usize>,
    expression: &Expression<F>,
) -> usize {
    expression.evaluate(
        &|_| 0,
        &|_| 0,
        &|query| (!preprocess_polys.contains(&query.poly())) as usize,
        &|_| 1,
        &|a| a,
        &|a, b| a.max(b),
        &|a, b| a + b,
        &|a, _| a,
    )
}
