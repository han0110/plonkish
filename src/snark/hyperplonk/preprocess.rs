use crate::{
    piop::sum_check::VirtualPolynomialInfo,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        expression::{CommonPolynomial, Expression, Query, Rotation},
        Itertools,
    },
};
use num_integer::Integer;
use std::{array, collections::BTreeSet, iter, mem};

#[derive(Clone, Debug)]
pub struct PlonkishCircuitInfo<F> {
    /// 2^k is the size of the circuit
    pub k: usize,
    /// Number of instnace value in each instance polynomial.
    pub num_instance: Vec<usize>,
    /// Preprocessed polynomials, which has index starts with offset
    /// `num_instance.len()`.
    pub preprocess_polys: Vec<MultilinearPolynomial<F>>,
    /// Number of witness polynoimal in each phase.
    /// Witness polynomial index starts with offset `num_instance.len()` +
    /// `preprocess_polys.len()`.
    pub num_witness_poly: Vec<usize>,
    /// Number of challenge in each phase.
    pub num_challenge: Vec<usize>,
    /// Constraints.
    pub constraints: Vec<Expression<F>>,
    /// Each item inside outer vector repesents an independent vector lookup,
    /// which contains vector of tuples representing the input and table
    /// respectively.
    pub lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    /// Each item inside outer vector repesents an closed permutation cycle,
    /// which contains vetor of tuples representing the polynomial index and
    /// row respectively.
    pub permutations: Vec<Vec<(usize, usize)>>,
    /// Maximum degree of constraints
    pub max_degree: Option<usize>,
}

impl<F: Clone> PlonkishCircuitInfo<F> {
    pub fn is_well_formed(&self) -> bool {
        let num_poly = self.num_poly();
        let num_challenge = self.num_challenge.iter().sum::<usize>();
        let polys = iter::empty()
            .chain(self.expressions().flat_map(Expression::used_poly))
            .chain(self.permutation_polys())
            .collect::<BTreeSet<_>>();
        let challenges = iter::empty()
            .chain(self.expressions().flat_map(Expression::used_challenge))
            .collect::<BTreeSet<_>>();
        // Same amount of phases
        self.num_witness_poly.len() == self.num_challenge.len()
            // Polynomial indices are in range
            && (polys.is_empty() || *polys.last().unwrap() < num_poly)
            // Challenge indices are in range
            && (challenges.is_empty() || *challenges.last().unwrap() < num_challenge)
            // Every constraint has degree less equal than `max_degree`
            && self
                .max_degree
                .map(|max_degree| {
                    !self
                        .constraints
                        .iter()
                        .any(|constraint| constraint.degree() > max_degree)
                })
                .unwrap_or(true)
    }

    pub fn num_poly(&self) -> usize {
        self.num_instance.len()
            + self.preprocess_polys.len()
            + self.num_witness_poly.iter().sum::<usize>()
    }

    pub fn permutation_polys(&self) -> Vec<usize> {
        self.permutations
            .iter()
            .flat_map(|cycle| cycle.iter().map(|(poly, _)| *poly))
            .unique()
            .sorted()
            .collect()
    }

    pub fn expressions(&self) -> impl Iterator<Item = &Expression<F>> {
        iter::empty().chain(self.constraints.iter()).chain(
            self.lookups
                .iter()
                .flat_map(|lookup| lookup.iter().flat_map(|(input, table)| [input, table])),
        )
    }
}

pub(super) fn compose<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
) -> (usize, VirtualPolynomialInfo<F>) {
    let permutation_polys = circuit_info.permutation_polys();
    let challenge_offset = circuit_info.num_challenge.iter().sum::<usize>();
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
        .max()
        .unwrap_or(2);
    let permutation_constraints = {
        let chunk_size = max_degree - 1;
        let num_chunk = Integer::div_ceil(&permutation_polys.len(), &chunk_size);
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
            .take(num_chunk)
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

    let virtual_poly_info = {
        let constraints = circuit_info
            .constraints
            .iter()
            .chain(lookup_constraints.iter())
            .chain(permutation_constraints.iter())
            .collect_vec();
        let eq = Expression::eq_xy(0);
        VirtualPolynomialInfo::new(Expression::distribute_powers(constraints, alpha) * eq)
    };

    (max_degree, virtual_poly_info)
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
            (idx << num_vars..(idx + 1) << num_vars)
                .map(F::from)
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
        piop::sum_check::VirtualPolynomialInfo,
        poly::multilinear::MultilinearPolynomial,
        snark::hyperplonk::preprocess::{compose, PlonkishCircuitInfo},
        util::{
            arithmetic::PrimeField,
            expression::{Expression, Query, Rotation},
        },
    };
    use halo2_curves::bn256::Fr;
    use std::array;

    #[test]
    fn test_preprocess_plonk() {
        let virtual_poly_info = plonk_virtual_poly_info();
        assert_eq!(virtual_poly_info.expression(), &{
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
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi,
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
    fn test_preprocess_plonk_with_lookup() {
        let virtual_poly_info = plonk_with_lookup_virtual_poly_info();
        assert_eq!(virtual_poly_info.expression(), &{
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
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi,
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

    pub(crate) fn plonk_circuit_info<F: PrimeField>(
        num_vars: usize,
        num_instance: usize,
        preprocess_polys: [MultilinearPolynomial<F>; 5],
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<F> {
        let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, Rotation::cur())).map(Expression::Polynomial);
        PlonkishCircuitInfo {
            k: num_vars,
            num_instance: vec![num_instance],
            preprocess_polys: preprocess_polys.to_vec(),
            num_witness_poly: vec![3],
            num_challenge: vec![0],
            constraints: vec![q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi],
            lookups: Vec::new(),
            permutations,
            max_degree: Some(4),
        }
    }

    pub(crate) fn plonk_virtual_poly_info<F: PrimeField>() -> VirtualPolynomialInfo<F> {
        let circuit_info = plonk_circuit_info(
            0,
            0,
            Default::default(),
            vec![vec![(6, 1)], vec![(7, 1)], vec![(8, 1)]],
        );
        let (max_degree, virtual_poly_info) = compose(&circuit_info);
        assert_eq!(max_degree, 4);
        virtual_poly_info
    }

    pub(crate) fn plonk_with_lookup_circuit_info<F: PrimeField>(
        num_vars: usize,
        num_instance: usize,
        preprocess_polys: [MultilinearPolynomial<F>; 9],
        permutations: Vec<Vec<(usize, usize)>>,
    ) -> PlonkishCircuitInfo<F> {
        let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, Rotation::cur())).map(Expression::Polynomial);
        PlonkishCircuitInfo {
            k: num_vars,
            num_instance: vec![num_instance],
            preprocess_polys: preprocess_polys.to_vec(),
            num_witness_poly: vec![3],
            num_challenge: vec![0],
            constraints: vec![q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi],
            lookups: vec![vec![
                (q_lookup * w_l, t_l.clone()),
                (q_lookup * w_r, t_r.clone()),
                (q_lookup * w_o, t_o.clone()),
            ]],
            permutations,
            max_degree: Some(4),
        }
    }

    pub(crate) fn plonk_with_lookup_virtual_poly_info<F: PrimeField>() -> VirtualPolynomialInfo<F> {
        let circuit_info = plonk_with_lookup_circuit_info(
            0,
            0,
            Default::default(),
            vec![vec![(10, 1)], vec![(11, 1)], vec![(12, 1)]],
        );
        let (max_degree, virtual_poly_info) = compose(&circuit_info);
        assert_eq!(max_degree, 4);
        virtual_poly_info
    }
}
