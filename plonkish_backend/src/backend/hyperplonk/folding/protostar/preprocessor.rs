use crate::{
    backend::{
        hyperplonk::{
            folding::{
                protostar::{ProtostarProverParam, ProtostarVerifierParam},
                sangria::preprocessor::{folding_degree, lookup_constraints},
            },
            preprocessor::permutation_constraints,
            HyperPlonk,
        },
        PlonkishBackend, PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        expression::{
            relaxed::{cross_term_expressions, products, relaxed_expression, PolynomialSet},
            Expression, Query, Rotation,
        },
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use std::{array, hash::Hash, iter};

pub(super) use crate::backend::hyperplonk::folding::sangria::preprocessor::batch_size;

#[allow(clippy::type_complexity)]
pub(super) fn preprocess<F, Pcs>(
    param: &Pcs::Param,
    circuit_info: &PlonkishCircuitInfo<F>,
) -> Result<(ProtostarProverParam<F, Pcs>, ProtostarVerifierParam<F, Pcs>), Error>
where
    F: PrimeField + Ord + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    let challenge_offset = circuit_info.num_challenges.iter().sum::<usize>();
    let max_lookup_width = circuit_info.lookups.iter().map(Vec::len).max().unwrap_or(0);
    let num_theta_primes = max_lookup_width.checked_sub(1).unwrap_or_default();
    let theta_primes = (challenge_offset..)
        .take(num_theta_primes)
        .map(Expression::<F>::Challenge)
        .collect_vec();
    let [beta_prime, zeta] =
        &[0, 1].map(|idx| Expression::<F>::Challenge(challenge_offset + num_theta_primes + idx));
    let alpha_prime_offset = challenge_offset + num_theta_primes + 2;
    let powers_of_zeta = circuit_info.num_poly()
        + circuit_info.permutation_polys().len()
        + circuit_info.lookups.len() * 3;

    let (lookup_constraints, lookup_zero_checks) =
        lookup_constraints(circuit_info, &theta_primes, beta_prime);
    let powers_of_zeta_constraint = powers_of_zeta_constraint(zeta, powers_of_zeta);

    let max_degree = iter::empty()
        .chain(circuit_info.constraints.iter())
        .chain(lookup_constraints.iter())
        .chain(Some(&powers_of_zeta_constraint))
        .map(Expression::degree)
        .chain(circuit_info.max_degree)
        .chain(Some(2))
        .max()
        .unwrap();

    let num_witness_polys = circuit_info.num_witness_polys.iter().sum::<usize>();
    let num_builtin_witness_polys = 3 * circuit_info.lookups.len() + 1;
    let witness_poly_offset =
        circuit_info.num_instances.len() + circuit_info.preprocess_polys.len();
    let builtin_witness_poly_offset =
        witness_poly_offset + num_witness_polys + circuit_info.permutation_polys().len();

    let poly_set = PolynomialSet {
        preprocess: iter::empty()
            .chain((circuit_info.num_instances.len()..).take(circuit_info.preprocess_polys.len()))
            .collect(),
        folding: iter::empty()
            .chain(0..circuit_info.num_instances.len())
            .chain((witness_poly_offset..).take(num_witness_polys))
            .chain((builtin_witness_poly_offset..).take(num_builtin_witness_polys))
            .collect(),
    };

    let num_constraints = circuit_info.constraints.len() + lookup_constraints.len();
    let num_alpha_primes = num_constraints.checked_sub(1).unwrap_or_default();

    let compressed_products = {
        let mut constraints = iter::empty()
            .chain(circuit_info.constraints.iter())
            .chain(lookup_constraints.iter())
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
        let compressed_constraint = iter::empty()
            .chain(constraints.first().cloned().cloned())
            .chain(
                constraints
                    .into_iter()
                    .skip(1)
                    .zip((alpha_prime_offset..).map(Expression::Challenge))
                    .map(|(constraint, challenge)| constraint * challenge),
            )
            .sum::<Expression<_>>()
            * powers_of_zeta;
        products(&poly_set.preprocess, &compressed_constraint)
    };
    let zeta_products = products(&poly_set.preprocess, &powers_of_zeta_constraint);

    let num_folding_witness_polys = num_witness_polys + num_builtin_witness_polys;
    let num_folding_challenges = alpha_prime_offset + num_alpha_primes;

    let compressed_cross_term_expressions =
        cross_term_expressions(&poly_set, &compressed_products, num_folding_challenges);
    let num_compressed_cross_terms = compressed_cross_term_expressions.len();

    let [beta, gamma, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(num_folding_challenges + idx));
    let (num_permutation_z_polys, permutation_constraints) = permutation_constraints(
        circuit_info,
        max_degree,
        beta,
        gamma,
        num_builtin_witness_polys,
    );

    let u = num_folding_challenges + 3;
    let relexed_compressed_constraint = relaxed_expression(&compressed_products, u);
    let relexed_zeta_constraint = {
        let e = powers_of_zeta + num_permutation_z_polys + 1;
        relaxed_expression(&zeta_products, u)
            - Expression::Polynomial(Query::new(e, Rotation::cur()))
    };
    let zero_check_on_every_row = Expression::distribute_powers(
        iter::empty()
            .chain(Some(&relexed_zeta_constraint))
            .chain(&permutation_constraints),
        alpha,
    ) * Expression::eq_xy(0);
    let sum_check_expression = Expression::distribute_powers(
        iter::empty()
            .chain(Some(&relexed_compressed_constraint))
            .chain(lookup_zero_checks.iter())
            .chain(Some(&zero_check_on_every_row)),
        alpha,
    );

    let (pcs_pp, pcs_vp) = Pcs::trim(param, 1 << circuit_info.k, batch_size(circuit_info))?;
    let (mut pp, mut vp) = HyperPlonk::preprocess(param, circuit_info)?;
    pp.num_permutation_z_polys = num_permutation_z_polys;
    vp.num_permutation_z_polys = num_permutation_z_polys;
    pp.pcs = pcs_pp;
    vp.pcs = pcs_vp;

    Ok((
        ProtostarProverParam {
            pp,
            num_theta_primes,
            num_alpha_primes,
            num_folding_witness_polys,
            num_folding_challenges,
            compressed_cross_term_expressions,
            sum_check_expression: sum_check_expression.clone(),
        },
        ProtostarVerifierParam {
            vp,
            num_theta_primes,
            num_alpha_primes,
            num_folding_witness_polys,
            num_folding_challenges,
            num_compressed_cross_terms,
            sum_check_expression,
        },
    ))
}

fn powers_of_zeta_constraint<F: PrimeField>(
    zeta: &Expression<F>,
    powers_of_zeta: usize,
) -> Expression<F> {
    let l_0 = &Expression::<F>::lagrange(0);
    let l_last = &Expression::<F>::lagrange(-1);
    let one = &Expression::one();
    let [powers_of_zeta, powers_of_zeta_next] = &[Rotation::cur(), Rotation::next()]
        .map(|rotation| Expression::Polynomial(Query::new(powers_of_zeta, rotation)));

    powers_of_zeta_next - (l_0 + l_last * zeta + (one - (l_0 + l_last)) * powers_of_zeta * zeta)
}
