use crate::{
    accumulation::protostar::{
        ProtostarProverParam, ProtostarStrategy,
        ProtostarStrategy::{Compressing, NoCompressing},
        ProtostarVerifierParam,
    },
    backend::{
        hyperplonk::{preprocessor::permutation_constraints, HyperPlonk},
        PlonkishBackend, PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, PrimeField},
        chain,
        expression::{
            relaxed::{cross_term_expressions, products, relaxed_expression, PolynomialSet},
            Expression, Query, Rotation,
        },
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use std::{array, borrow::Cow, collections::BTreeSet, hash::Hash};

pub(crate) fn batch_size<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    strategy: ProtostarStrategy,
) -> usize {
    let num_lookups = circuit_info.lookups.len();
    let num_permutation_polys = circuit_info.permutation_polys().len();
    chain![
        [circuit_info.preprocess_polys.len() + circuit_info.permutation_polys().len()],
        circuit_info.num_witness_polys.clone(),
        [num_lookups],
        match strategy {
            NoCompressing => {
                vec![]
            }
            Compressing => {
                vec![1]
            }
        },
        [2 * num_lookups + div_ceil(num_permutation_polys, max_degree(circuit_info, None) - 1)],
        [1],
    ]
    .sum()
}

#[allow(clippy::type_complexity)]
pub(super) fn preprocess<F, Pcs>(
    param: &Pcs::Param,
    circuit_info: &PlonkishCircuitInfo<F>,
    strategy: ProtostarStrategy,
) -> Result<
    (
        ProtostarProverParam<F, HyperPlonk<Pcs>>,
        ProtostarVerifierParam<F, HyperPlonk<Pcs>>,
    ),
    Error,
>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
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

    let max_degree = max_degree(circuit_info, Some(&lookup_constraints));

    let num_constraints = circuit_info.constraints.len() + lookup_constraints.len();
    let num_alpha_primes = num_constraints.checked_sub(1).unwrap_or_default();

    let witness_poly_offset =
        circuit_info.num_instances.len() + circuit_info.preprocess_polys.len();
    let num_witness_polys = circuit_info.num_witness_polys.iter().sum::<usize>();
    let num_permutation_z_polys = div_ceil(circuit_info.permutation_polys().len(), max_degree - 1);

    let (
        num_builtin_witness_polys,
        alpha_prime_offset,
        cross_term_expressions,
        sum_check,
        zero_check_on_every_row,
    ) = match strategy {
        NoCompressing => {
            let alpha_prime_offset = challenge_offset + num_theta_primes + 1;
            let num_builtin_witness_polys = 3 * circuit_info.lookups.len();
            let builtin_witness_poly_offset =
                witness_poly_offset + num_witness_polys + circuit_info.permutation_polys().len();

            let poly_set = PolynomialSet {
                preprocess: chain![
                    (circuit_info.num_instances.len()..).take(circuit_info.preprocess_polys.len()),
                ]
                .collect(),
                folding: chain![
                    0..circuit_info.num_instances.len(),
                    (witness_poly_offset..).take(num_witness_polys),
                    (builtin_witness_poly_offset..).take(num_builtin_witness_polys),
                ]
                .collect(),
            };

            let products = {
                let mut constraints =
                    chain![circuit_info.constraints.iter(), lookup_constraints.iter()]
                        .collect_vec();
                let folding_degrees = constraints
                    .iter()
                    .map(|constraint| folding_degree(&poly_set.preprocess, constraint))
                    .enumerate()
                    .sorted_by(|a, b| b.1.cmp(&a.1))
                    .collect_vec();
                if let &[a, b, ..] = &folding_degrees[..] {
                    if a.1 != b.1 {
                        constraints.swap(0, a.0);
                    }
                }
                let compressed_constraint = chain![
                    constraints.first().cloned().cloned(),
                    constraints
                        .into_iter()
                        .skip(1)
                        .zip((alpha_prime_offset..).map(Expression::Challenge))
                        .map(|(constraint, challenge)| constraint * challenge),
                ]
                .sum::<Expression<_>>();
                products(&poly_set.preprocess, &compressed_constraint)
            };

            let num_folding_challenges = alpha_prime_offset + num_alpha_primes;
            let cross_term_expressions =
                cross_term_expressions(&poly_set, &products, num_folding_challenges);

            let u = num_folding_challenges;
            let relexed_constraint = {
                let e = builtin_witness_poly_offset
                    + circuit_info.lookups.len() * 3
                    + num_permutation_z_polys;
                relaxed_expression(&products, u)
                    - Expression::Polynomial(Query::new(e, Rotation::cur()))
            };

            (
                num_builtin_witness_polys,
                alpha_prime_offset,
                cross_term_expressions,
                None,
                relexed_constraint,
            )
        }
        Compressing => {
            let zeta = challenge_offset + num_theta_primes + 1;
            let alpha_prime_offset = zeta + 1;
            let num_builtin_witness_polys = 3 * circuit_info.lookups.len() + 1;
            let builtin_witness_poly_offset =
                witness_poly_offset + num_witness_polys + circuit_info.permutation_polys().len();

            let poly_set = PolynomialSet {
                preprocess: (circuit_info.num_instances.len()..)
                    .take(circuit_info.preprocess_polys.len())
                    .collect(),
                folding: chain![
                    0..circuit_info.num_instances.len(),
                    (witness_poly_offset..).take(num_witness_polys),
                    (builtin_witness_poly_offset..).take(num_builtin_witness_polys),
                ]
                .collect(),
            };

            let powers_of_zeta = builtin_witness_poly_offset + circuit_info.lookups.len() * 3;
            let compressed_products = {
                let mut constraints =
                    chain![circuit_info.constraints.iter(), lookup_constraints.iter()]
                        .collect_vec();
                let folding_degrees = constraints
                    .iter()
                    .map(|constraint| folding_degree(&poly_set.preprocess, constraint))
                    .enumerate()
                    .sorted_by(|a, b| b.1.cmp(&a.1))
                    .collect_vec();
                if let &[a, b, ..] = &folding_degrees[..] {
                    if a.1 != b.1 {
                        constraints.swap(0, a.0);
                    }
                }
                let powers_of_zeta =
                    Expression::<F>::Polynomial(Query::new(powers_of_zeta, Rotation::cur()));
                let compressed_constraint = chain![
                    constraints.first().cloned().cloned(),
                    constraints
                        .into_iter()
                        .skip(1)
                        .zip((alpha_prime_offset..).map(Expression::Challenge))
                        .map(|(constraint, challenge)| constraint * challenge),
                ]
                .sum::<Expression<_>>()
                    * powers_of_zeta;
                products(&poly_set.preprocess, &compressed_constraint)
            };
            let powers_of_zeta_constraint = powers_of_zeta_constraint(zeta, powers_of_zeta);
            let zeta_products = products(&poly_set.preprocess, &powers_of_zeta_constraint);

            let num_folding_challenges = alpha_prime_offset + num_alpha_primes;
            let cross_term_expressions =
                cross_term_expressions(&poly_set, &compressed_products, num_folding_challenges);

            let u = num_folding_challenges;
            let relexed_compressed_constraint = relaxed_expression(&compressed_products, u);
            let relexed_zeta_constraint = {
                let e = powers_of_zeta + num_permutation_z_polys + 1;
                relaxed_expression(&zeta_products, u)
                    - Expression::Polynomial(Query::new(e, Rotation::cur()))
            };

            (
                num_builtin_witness_polys,
                alpha_prime_offset,
                cross_term_expressions,
                Some(relexed_compressed_constraint),
                relexed_zeta_constraint,
            )
        }
    };

    let num_folding_witness_polys = num_witness_polys + num_builtin_witness_polys;
    let num_folding_challenges = alpha_prime_offset + num_alpha_primes;

    let [beta, gamma, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(num_folding_challenges + 1 + idx));
    let (_, permutation_constraints) = permutation_constraints(
        circuit_info,
        max_degree,
        beta,
        gamma,
        num_builtin_witness_polys,
    );

    let expression = {
        let zero_check_on_every_row = Expression::distribute_powers(
            chain![[&zero_check_on_every_row], &permutation_constraints],
            alpha,
        ) * Expression::eq_xy(0);
        Expression::distribute_powers(
            chain![
                &sum_check,
                lookup_zero_checks.iter(),
                [&zero_check_on_every_row],
            ],
            alpha,
        )
    };

    let (pp, vp) = {
        let (mut pp, mut vp) = HyperPlonk::preprocess(param, circuit_info)?;
        let batch_size = batch_size(circuit_info, strategy);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, 1 << circuit_info.k, batch_size)?;
        pp.pcs = pcs_pp;
        vp.pcs = pcs_vp;
        pp.num_permutation_z_polys = num_permutation_z_polys;
        vp.num_permutation_z_polys = num_permutation_z_polys;
        pp.expression = expression.clone();
        vp.expression = expression;
        (pp, vp)
    };

    let num_cross_terms = cross_term_expressions.len();

    Ok((
        ProtostarProverParam {
            pp,
            strategy,
            num_theta_primes,
            num_alpha_primes,
            num_folding_witness_polys,
            num_folding_challenges,
            cross_term_expressions,
        },
        ProtostarVerifierParam {
            vp,
            strategy,
            num_theta_primes,
            num_alpha_primes,
            num_folding_witness_polys,
            num_folding_challenges,
            num_cross_terms,
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
    chain![
        circuit_info.constraints.iter().map(Expression::degree),
        lookup_constraints.iter().map(Expression::degree),
        circuit_info.max_degree,
        [2],
    ]
    .max()
    .unwrap()
}

pub(crate) fn folding_degree<F: PrimeField>(
    preprocess_polys: &BTreeSet<usize>,
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
                chain![
                    exprs.first().cloned().cloned(),
                    exprs
                        .into_iter()
                        .skip(1)
                        .zip(theta_primes)
                        .map(|(expr, theta_prime)| expr * theta_prime),
                ]
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

fn powers_of_zeta_constraint<F: PrimeField>(zeta: usize, powers_of_zeta: usize) -> Expression<F> {
    let l_last = &Expression::<F>::lagrange(-1);
    let one = &Expression::one();
    let zeta = &Expression::Challenge(zeta);
    let [powers_of_zeta, powers_of_zeta_next] = &[Rotation::cur(), Rotation::next()]
        .map(|rotation| Expression::Polynomial(Query::new(powers_of_zeta, rotation)));

    powers_of_zeta_next - (l_last + (one - l_last) * powers_of_zeta * zeta)
}
