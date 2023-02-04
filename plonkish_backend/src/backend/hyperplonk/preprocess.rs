use crate::{
    backend::PlonkishCircuitInfo,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps, PrimeField},
        expression::{CommonPolynomial, Expression, Query, Rotation},
        Itertools,
    },
};
use std::{array, iter, mem};

pub(super) fn compose<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
) -> (usize, Expression<F>) {
    let permutation_polys = circuit_info.permutation_polys();
    let challenge_offset = circuit_info.num_challenges.iter().sum::<usize>();
    let [theta, beta, gamma, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(challenge_offset + idx));
    let l_1 = &Expression::<F>::lagrange(1);
    let one = &Expression::Constant(F::one());
    let lookup_constraints = {
        let permuted_input_offset = circuit_info.num_poly() + permutation_polys.len();
        let z_offset = permuted_input_offset + 2 * circuit_info.lookups.len();
        circuit_info
            .lookups
            .iter()
            .zip((permuted_input_offset..).step_by(2))
            .zip(z_offset..)
            .flat_map(|((lookup, permuted_input), z)| {
                let [permuted_input, permuted_input_prev, permuted_table, z, z_next] = &[
                    Query::new(permuted_input, Rotation::cur()),
                    Query::new(permuted_input, Rotation::prev()),
                    Query::new(permuted_input + 1, Rotation::cur()),
                    Query::new(z, Rotation::cur()),
                    Query::new(z, Rotation::next()),
                ]
                .map(Expression::Polynomial);
                let (inputs, tables) = lookup
                    .iter()
                    .map(|(input, table)| (input, table))
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                let input = Expression::distribute_powers(inputs, theta);
                let table = Expression::distribute_powers(tables, theta);
                [
                    l_1 * (z - one),
                    z * (input + beta) * (table + gamma)
                        - z_next * (permuted_input + beta) * (permuted_table + gamma),
                    (permuted_input - permuted_table) * (permuted_input - permuted_input_prev),
                ]
            })
            .collect_vec()
    };

    let max_degree = iter::empty()
        .chain(circuit_info.constraints.iter())
        .chain(lookup_constraints.iter())
        .map(Expression::degree)
        .chain(circuit_info.max_degree)
        .chain(Some(2))
        .max()
        .unwrap();
    let permutation_constraints = {
        let chunk_size = max_degree - 1;
        let num_chunks = div_ceil(permutation_polys.len(), chunk_size);
        let permutation_offset = circuit_info.num_poly();
        let z_offset =
            permutation_offset + permutation_polys.len() + 3 * circuit_info.lookups.len();
        let polys = permutation_polys
            .iter()
            .map(|idx| Query::new(*idx, Rotation::cur()))
            .map(Expression::<F>::Polynomial)
            .collect_vec();
        let ids = (0..polys.len())
            .map(|idx| Expression::CommonPolynomial(CommonPolynomial::Identity(idx)))
            .collect_vec();
        let permutations = (permutation_offset..)
            .map(|idx| Query::new(idx, Rotation::cur()))
            .map(Expression::<F>::Polynomial)
            .take(permutation_polys.len())
            .collect_vec();
        let zs = (z_offset..)
            .map(|idx| Query::new(idx, Rotation::cur()))
            .map(Expression::<F>::Polynomial)
            .take(num_chunks)
            .collect_vec();
        let z_0_next = Expression::<F>::Polynomial(Query::new(z_offset, Rotation::next()));
        iter::empty()
            .chain(zs.first().map(|z_0| l_1 * (z_0 - one)))
            .chain(
                polys
                    .chunks(chunk_size)
                    .zip(ids.chunks(chunk_size))
                    .zip(permutations.chunks(chunk_size))
                    .zip(zs.iter())
                    .zip(zs.iter().skip(1).chain(Some(&z_0_next)))
                    .map(|((((polys, ids), permutations), z_lhs), z_rhs)| {
                        z_lhs
                            * polys
                                .iter()
                                .zip(ids)
                                .map(|(poly, id)| poly + beta * id + gamma)
                                .product::<Expression<_>>()
                            - z_rhs
                                * polys
                                    .iter()
                                    .zip(permutations)
                                    .map(|(poly, permutation)| poly + beta * permutation + gamma)
                                    .product::<Expression<_>>()
                    }),
            )
            .collect_vec()
    };

    let expression = {
        let constraints = circuit_info
            .constraints
            .iter()
            .chain(lookup_constraints.iter())
            .chain(permutation_constraints.iter())
            .collect_vec();
        let eq = Expression::eq_xy(0);
        Expression::distribute_powers(constraints, alpha) * eq
    };

    (max_degree, expression)
}

pub(super) fn permutation_polys<F: PrimeField>(
    num_vars: usize,
    permutation_polys: &[usize],
    cycles: &[Vec<(usize, usize)>],
) -> Vec<(usize, MultilinearPolynomial<F>)> {
    let poly_index = {
        let mut poly_index = vec![0; permutation_polys.last().map(|poly| 1 + poly).unwrap_or(0)];
        for (idx, poly) in permutation_polys.iter().enumerate() {
            poly_index[*poly] = idx;
        }
        poly_index
    };
    let mut permutations = (0..permutation_polys.len() as u64)
        .map(|idx| {
            steps(F::from(idx << num_vars))
                .take(1 << num_vars)
                .collect_vec()
        })
        .collect_vec();
    for cycle in cycles.iter() {
        let (i0, j0) = cycle[0];
        let mut last = permutations[poly_index[i0]][j0];
        for &(i, j) in cycle.iter().cycle().skip(1).take(cycle.len()) {
            assert_ne!(j, 0);
            mem::swap(&mut permutations[poly_index[i]][j], &mut last);
        }
    }
    permutation_polys
        .iter()
        .cloned()
        .zip(permutations.into_iter().map(MultilinearPolynomial::new))
        .collect()
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::hyperplonk::util::{plonk_expression, plonk_with_lookup_expression},
        util::expression::{Expression, Query, Rotation},
    };
    use halo2_curves::bn256::Fr;
    use std::array;

    #[test]
    fn compose_plonk() {
        let expression = plonk_expression();
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [z, z_next] = &[
                Query::new(12, Rotation::cur()),
                Query::new(12, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, alpha] = &[1, 2, 3].map(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(Expression::identity);
            let l_1 = Expression::<Fr>::lagrange(1);
            let one = Expression::Constant(Fr::one());
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    l_1 * (z - one),
                    (z * ((w_l + beta * id_1 + gamma)
                        * (w_r + beta * id_2 + gamma)
                        * (w_o + beta * id_3 + gamma)))
                        - (z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            Expression::distribute_powers(&constraints, alpha) * eq
        });
    }

    #[test]
    fn compose_plonk_with_lookup() {
        let expression = plonk_with_lookup_expression();
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [lookup_permuted_input, lookup_permuted_input_prev, lookup_permuted_table, lookup_z, lookup_z_next] =
                &[
                    Query::new(16, Rotation::cur()),
                    Query::new(16, Rotation::prev()),
                    Query::new(17, Rotation::cur()),
                    Query::new(18, Rotation::cur()),
                    Query::new(18, Rotation::next()),
                ]
                .map(Expression::Polynomial);
            let [perm_z, perm_z_next] = &[
                Query::new(19, Rotation::cur()),
                Query::new(19, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [theta, beta, gamma, alpha] = &array::from_fn(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(Expression::identity);
            let l_1 = &Expression::<Fr>::lagrange(1);
            let one = &Expression::Constant(Fr::one());
            let lookup_compressed_input =
                Expression::distribute_powers(&[w_l, w_r, w_o].map(|w| q_lookup * w), theta);
            let lookup_compressed_table = Expression::distribute_powers([t_l, t_r, t_o], theta);
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    l_1 * (lookup_z - one),
                    lookup_z * (lookup_compressed_input + beta) * (lookup_compressed_table + gamma)
                        - lookup_z_next
                            * (lookup_permuted_input + beta)
                            * (lookup_permuted_table + gamma),
                    (lookup_permuted_input - lookup_permuted_table)
                        * (lookup_permuted_input - lookup_permuted_input_prev),
                    l_1 * (perm_z - one),
                    (perm_z
                        * ((w_l + beta * id_1 + gamma)
                            * (w_r + beta * id_2 + gamma)
                            * (w_o + beta * id_3 + gamma)))
                        - (perm_z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            Expression::distribute_powers(&constraints, alpha) * eq
        });
    }
}
