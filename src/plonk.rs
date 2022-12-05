#[cfg(test)]
pub(crate) mod test {
    use crate::{
        poly::multilinear::MultilinearPolynomial,
        util::{
            arithmetic::{BatchInvert, BooleanHypercube, Field, PrimeField},
            expression::{Expression, Query, Rotation},
            test::rand_array,
            Itertools,
        },
    };
    use num_integer::Integer;
    use rand::RngCore;
    use std::{array, hash::Hash, iter, mem};

    pub fn hyperplonk_expression<F: Field>() -> Expression<F> {
        let [q_l, q_r, q_m, q_o, q_c, id_1, id_2, id_3, s_1, s_2, s_3, pi, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, poly, Rotation::cur()))
                .map(Expression::Polynomial);
        let [z, z_next] = &[
            Query::new(15, 15, Rotation::cur()),
            Query::new(16, 15, Rotation::next()),
        ]
        .map(Expression::Polynomial);
        let [beta, gamma, alpha] = &[0, 1, 2].map(Expression::<F>::Challenge);
        let l_1 = Expression::lagrange(1);
        let one = Expression::Constant(F::one());
        let gates = {
            vec![
                q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi,
                ((w_l + beta * id_1 + gamma)
                    * (w_r + beta * id_2 + gamma)
                    * (w_o + beta * id_3 + gamma))
                    * z
                    - ((w_l + beta * s_1 + gamma)
                        * (w_r + beta * s_2 + gamma)
                        * (w_o + beta * s_3 + gamma))
                        * z_next,
                (z - one) * l_1,
            ]
        };
        let eq = Expression::eq_xy(0);
        Expression::random_linear_combine(&gates, alpha) * eq
    }

    pub fn hyperplonk_plus_expression<F: Field>() -> Expression<F> {
        let [q_l, q_r, q_m, q_o, q_c, q_lookup, id_1, id_2, id_3, s_1, s_2, s_3, t_l, t_r, t_o, pi, w_l, w_r, w_o] =
            &array::from_fn(|poly| Query::new(poly, poly, Rotation::cur()))
                .map(Expression::Polynomial);
        let [t_l_next, t_r_next, t_o_next] = &[(19, 12), (20, 13), (21, 14)]
            .map(|(idx, poly)| Query::new(idx, poly, Rotation::next()))
            .map(Expression::Polynomial);
        let [h_1, h_1_next, h_2] = &[
            Query::new(22, 19, Rotation::cur()),
            Query::new(23, 19, Rotation::next()),
            Query::new(24, 20, Rotation::cur()),
        ]
        .map(Expression::Polynomial);
        let [z_perm, z_perm_next] = &[
            Query::new(25, 21, Rotation::cur()),
            Query::new(26, 21, Rotation::next()),
        ]
        .map(Expression::Polynomial);
        let [z_lookup, z_lookup_next] = &[
            Query::new(27, 22, Rotation::cur()),
            Query::new(28, 22, Rotation::next()),
        ]
        .map(Expression::Polynomial);
        let [theta, beta, gamma, alpha] = &[0, 1, 2, 3].map(Expression::<F>::Challenge);
        let l_1 = &Expression::lagrange(1);
        let one = &Expression::Constant(F::one());
        let gates = {
            vec![
                q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c - pi,
                ((w_l + beta * id_1 + gamma)
                    * (w_r + beta * id_2 + gamma)
                    * (w_o + beta * id_3 + gamma))
                    * z_perm
                    - ((w_l + beta * s_1 + gamma)
                        * (w_r + beta * s_2 + gamma)
                        * (w_o + beta * s_3 + gamma))
                        * z_perm_next,
                {
                    let f = q_lookup * Expression::random_linear_combine([w_l, w_r, w_o], theta);
                    let t = Expression::random_linear_combine([t_l, t_r, t_o], theta);
                    let t_next =
                        Expression::random_linear_combine([t_l_next, t_r_next, t_o_next], theta);
                    ((one + beta) * f + gamma) * (t + beta * t_next + gamma) * z_lookup
                        - (h_1 + beta * h_2 + gamma)
                            * (h_2 + beta * h_1_next + gamma)
                            * z_lookup_next
                },
                (z_perm - one) * l_1,
                (z_lookup - one) * l_1,
            ]
        };
        let eq = Expression::eq_xy(0);
        Expression::random_linear_combine(&gates, alpha) * eq
    }

    pub fn rand_hyperplonk_assignments<F: Field>(
        num_vars: usize,
        id: impl Fn(usize) -> F,
        mut rng: impl RngCore,
    ) -> ([MultilinearPolynomial<F>; 16], [F; 3]) {
        let size = 1 << num_vars;
        let mut polys = [(); 9].map(|_| vec![F::zero(); size]);
        let [id_1, id_2, id_3] = array::from_fn(|idx| {
            (idx << num_vars..(idx + 1) << num_vars)
                .map(&id)
                .collect_vec()
        });
        let [beta, gamma, alpha] = [(); 3].map(|_| F::random(&mut rng));

        let mut cycles = Vec::new();
        for idx in 1..size {
            let [w_l, w_r, q_c] = if rng.next_u32().is_even() && idx != 1 {
                let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                    let mut idx = rng.next_u32() as usize % idx;
                    if idx == 0 {
                        idx += 1;
                    }
                    (rng.next_u32() as usize % 3, idx)
                });
                copy(&mut cycles, l_copy_idx, (0, idx));
                copy(&mut cycles, r_copy_idx, (1, idx));
                [
                    polys[6 + l_copy_idx.0][l_copy_idx.1],
                    polys[6 + r_copy_idx.0][r_copy_idx.1],
                    F::zero(),
                ]
            } else {
                rand_array(&mut rng)
            };
            let values = if rng.next_u32().is_even() {
                vec![
                    (0, F::one()),
                    (1, F::one()),
                    (3, -F::one()),
                    (4, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l + w_r + q_c),
                ]
            } else {
                vec![
                    (2, F::one()),
                    (3, -F::one()),
                    (4, q_c),
                    (6, w_l),
                    (7, w_r),
                    (8, w_l * w_r + q_c),
                ]
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }
        let [s_1, s_2, s_3] = sigmas([&id_1, &id_2, &id_3], &cycles);
        let z = perm_grand_product(
            &[&polys[6], &polys[7], &polys[8]],
            &[&id_1, &id_2, &id_3],
            &[&s_1, &s_2, &s_3],
            beta,
            gamma,
        );

        let [q_l, q_r, q_m, q_o, q_c, pi, w_l, w_r, w_o] = polys;
        let idx_map = BooleanHypercube::new(num_vars).idx_map();
        let polys = [
            q_l, q_r, q_m, q_o, q_c, id_1, id_2, id_3, s_1, s_2, s_3, pi, w_l, w_r, w_o, z,
        ]
        .map(|poly| (0..size).map(|idx| poly[idx_map[idx]]).collect())
        .map(MultilinearPolynomial::new);

        (polys, [beta, gamma, alpha])
    }

    pub fn rand_hyperplonk_plus_assignments<F: PrimeField + Hash>(
        num_vars: usize,
        id: impl Fn(usize) -> F,
        mut rng: impl RngCore,
    ) -> ([MultilinearPolynomial<F>; 23], [F; 4]) {
        let size = 1 << num_vars;
        let mut polys = [(); 10].map(|_| vec![F::zero(); size]);
        let [id_1, id_2, id_3] = array::from_fn(|idx| {
            (idx << num_vars..(idx + 1) << num_vars)
                .map(&id)
                .collect_vec()
        });
        let n_bit_xor = 1u64 << ((num_vars >> 1) - num_vars.is_even() as usize);
        let (t_l, t_r, t_o) = iter::once((F::zero(), F::zero(), F::zero()))
            .chain(
                (0..n_bit_xor)
                    .cartesian_product(0..n_bit_xor)
                    .map(|(lhs, rhs)| (F::from(lhs), F::from(rhs), F::from(lhs ^ rhs))),
            )
            .chain(iter::repeat_with(|| (F::zero(), F::zero(), F::zero())))
            .take(1 << num_vars)
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
        let [theta, beta, gamma, alpha] = [(); 4].map(|_| F::random(&mut rng));

        let mut cycles = Vec::new();
        for idx in 1..size {
            let use_copy = rng.next_u32().is_even() && idx != 1;
            let [w_l, w_r, q_c] = if use_copy {
                let [l_copy_idx, r_copy_idx] = [(); 2].map(|_| {
                    let mut idx = rng.next_u32() as usize % idx;
                    if idx == 0 {
                        idx += 1;
                    }
                    (rng.next_u32() as usize % 3, idx)
                });
                copy(&mut cycles, l_copy_idx, (0, idx));
                copy(&mut cycles, r_copy_idx, (1, idx));
                [
                    polys[7 + l_copy_idx.0][l_copy_idx.1],
                    polys[7 + r_copy_idx.0][r_copy_idx.1],
                    F::zero(),
                ]
            } else {
                rand_array(&mut rng)
            };
            let values = match (use_copy, rng.next_u32().is_even()) {
                (true, true) => {
                    vec![
                        (0, F::one()),
                        (1, F::one()),
                        (3, -F::one()),
                        (4, q_c),
                        (7, w_l),
                        (8, w_r),
                        (9, w_l + w_r + q_c),
                    ]
                }
                (true, false) => {
                    vec![
                        (2, F::one()),
                        (3, -F::one()),
                        (4, q_c),
                        (7, w_l),
                        (8, w_r),
                        (9, w_l * w_r + q_c),
                    ]
                }
                (false, _) => {
                    let [w_l, w_r] = array::from_fn(|_| (rng.next_u32() % n_bit_xor as u32) as u64);
                    vec![
                        (5, F::one()),
                        (7, F::from(w_l)),
                        (8, F::from(w_r)),
                        (9, F::from(w_l ^ w_r)),
                    ]
                }
            };
            for (poly, value) in values {
                polys[poly][idx] = value;
            }
        }
        let [s_1, s_2, s_3] = sigmas([&id_1, &id_2, &id_3], &cycles);
        let z_perm = perm_grand_product(
            &[&polys[7], &polys[8], &polys[9]],
            &[&id_1, &id_2, &id_3],
            &[&s_1, &s_2, &s_3],
            beta,
            gamma,
        );
        let f = polys[5]
            .iter()
            .zip(polys[7].iter())
            .zip(polys[8].iter())
            .zip(polys[9].iter())
            .map(|(((q_lookup, w_l), w_r), w_o)| (theta * (theta * w_l + w_r) + w_o) * q_lookup)
            .collect_vec();
        let t = t_l
            .iter()
            .zip(t_r.iter())
            .zip(t_o.iter())
            .map(|((t_l, t_r), t_o)| theta * (theta * t_l + t_r) + t_o)
            .collect_vec();
        let [h_1, h_2] = ordered_multiset(&t, [&f]);
        let z_lookup = lookup_grand_product(&t, &[&f], &[&h_1, &h_2], beta, gamma);

        let [q_l, q_r, q_m, q_o, q_c, q_lookup, pi, w_l, w_r, w_o] = polys;
        let idx_map = BooleanHypercube::new(num_vars).idx_map();
        let polys = [
            q_l, q_r, q_m, q_o, q_c, q_lookup, id_1, id_2, id_3, s_1, s_2, s_3, t_l, t_r, t_o, pi,
            w_l, w_r, w_o, h_1, h_2, z_perm, z_lookup,
        ]
        .map(|poly| (0..size).map(|idx| poly[idx_map[idx]]).collect())
        .map(MultilinearPolynomial::new);

        (polys, [theta, beta, gamma, alpha])
    }

    fn copy(cycles: &mut Vec<Vec<(usize, usize)>>, lhs: (usize, usize), rhs: (usize, usize)) {
        if let Some(pos) = cycles.iter().position(|cycle| cycle.contains(&lhs)) {
            cycles[pos].push(rhs);
        } else {
            cycles.push(vec![lhs, rhs]);
        }
    }

    fn sigmas<F: Field, const N: usize>(
        ids: [&[F]; N],
        cycles: &[Vec<(usize, usize)>],
    ) -> [Vec<F>; N] {
        let mut sigmas = ids.map(|id| id.to_vec());
        for cycle in cycles.iter() {
            let (i0, j0) = cycle[0];
            let mut last = sigmas[i0][j0];
            for &(i, j) in cycle.iter().cycle().skip(1).take(cycle.len()) {
                mem::swap(&mut sigmas[i][j], &mut last);
            }
        }
        sigmas
    }

    pub fn ordered_multiset<F: Field + Hash, const N: usize, const N_PLUS_ONE: usize>(
        t: &[F],
        f: [&[F]; N],
    ) -> [Vec<F>; N_PLUS_ONE] {
        assert_eq!(N + 1, N_PLUS_ONE);
        assert!(!f.iter().any(|f| f.len() != t.len()));

        let mut h_item_count = f
            .iter()
            .flat_map(|f| f.iter().skip(1))
            .chain(t.iter().skip(1))
            .counts();

        let n = f.len() + 1;
        let mut hs = array::from_fn(|_| Vec::with_capacity(t.len()));
        hs.iter_mut().for_each(|h| h.push(F::zero()));
        let mut idx = 0;
        for &item in t.iter().skip(1) {
            let count = h_item_count.get_mut(&item).unwrap();

            let (q, r) = count.div_rem(&n);
            if q > 0 {
                for h in hs.iter_mut() {
                    h.extend(vec![item; q]);
                }
            }

            let idx_next = idx + r;
            if idx_next < n {
                for h in hs[idx..idx_next].iter_mut() {
                    h.push(item);
                }
                idx = idx_next
            } else {
                for h in hs[idx..].iter_mut() {
                    h.push(item);
                }
                idx = idx_next - n;
                for h in hs[..idx].iter_mut() {
                    h.push(item);
                }
            }

            *count = 0;
        }

        assert!(!hs.iter().any(|h| h.len() != t.len()));

        hs
    }

    fn perm_grand_product<F: Field>(
        w: &[&[F]],
        id: &[&[F]],
        s: &[&[F]],
        beta: F,
        gamma: F,
    ) -> Vec<F> {
        let numer = w
            .iter()
            .zip(id.iter())
            .map(|(w, id)| {
                w.iter()
                    .zip(id.iter())
                    .skip(1)
                    .map(|(w, id)| beta * id + w)
                    .collect_vec()
            })
            .collect_vec();
        let denom = w
            .iter()
            .zip(s.iter())
            .map(|(w, s)| {
                w.iter()
                    .zip(s.iter())
                    .skip(1)
                    .map(|(w, s)| beta * s + w)
                    .collect_vec()
            })
            .collect_vec();
        iter::once(F::zero())
            .chain(grand_product(&numer, &denom, gamma))
            .collect()
    }

    fn lookup_grand_product<F: Field>(
        t: &[F],
        f: &[&[F]],
        h: &[&[F]],
        beta: F,
        gamma: F,
    ) -> Vec<F> {
        let numer = f
            .iter()
            .map(|f| {
                f.iter()
                    .skip(1)
                    .map(|f| (F::one() + beta) * f)
                    .collect_vec()
            })
            .chain(Some(
                t.iter()
                    .skip(1)
                    .zip(t.iter().skip(1).cycle().skip(1))
                    .map(|(t, t_next)| beta * t_next + t)
                    .collect_vec(),
            ))
            .collect_vec();
        let denom = h
            .iter()
            .zip(
                h.iter()
                    .skip(1)
                    .map(|h| h.iter().cycle().skip(0))
                    .chain(iter::once(h.first().unwrap().iter().cycle().skip(1))),
            )
            .map(|(h, h_next)| {
                h.iter()
                    .zip(h_next)
                    .skip(1)
                    .map(|(h, h_next)| beta * h_next + h)
                    .collect_vec()
            })
            .collect_vec();
        iter::once(F::zero())
            .chain(grand_product(&numer, &denom, gamma))
            .collect()
    }

    fn grand_product<F: Field>(numer: &[Vec<F>], denom: &[Vec<F>], gamma: F) -> Vec<F> {
        let mut products = vec![F::one(); denom[0].len()];
        for denom in denom.iter() {
            for (product, value) in products.iter_mut().zip(denom.iter()) {
                *product *= gamma + value;
            }
        }
        products.batch_invert();
        for numer in numer.iter() {
            for (product, value) in products.iter_mut().zip(numer.iter()) {
                *product *= gamma + value;
            }
        }
        let mut z = Vec::with_capacity(products.len());
        z.push(F::one());
        for (idx, product) in products[..products.len() - 1].iter().enumerate() {
            z.push(z[idx] * product);
        }
        assert_eq!(*z.last().unwrap() * products.last().unwrap(), F::one());
        z
    }
}
