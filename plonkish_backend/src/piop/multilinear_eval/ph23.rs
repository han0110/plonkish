//! Implementation of section 5.1 of [PH23].
//!
//! [PH23]: https://eprint.iacr.org/2023/1284.pdf

use crate::util::{
    arithmetic::{powers, squares, BatchInvert, WithSmallOrderMulGroup},
    expression::{evaluator::quotient::Radix2Domain, Query},
    izip,
    parallel::parallelize,
    Itertools,
};

pub mod additive;

pub fn s_polys<F: WithSmallOrderMulGroup<3>>(num_vars: usize) -> Vec<Vec<F>> {
    let domain = Radix2Domain::<F>::new(num_vars, 2);
    let vanishing = {
        let coset_scalar = match domain.n() % 3 {
            1 => domain.zeta(),
            2 => domain.zeta_inv(),
            _ => unreachable!(),
        };
        powers(domain.extended_omega().pow([domain.n() as u64]))
            .map(|value| coset_scalar * value - F::ONE)
            .take(1 << (domain.extended_k() - domain.k()))
            .collect_vec()
    };
    let omegas = powers(domain.extended_omega())
        .take(domain.extended_n())
        .collect_vec();
    let mut s_polys = vec![vec![F::ZERO; domain.extended_n()]; domain.k()];
    parallelize(&mut s_polys, |(s_polys, start)| {
        izip!(s_polys, start..).for_each(|(s_polys, idx)| {
            let exponent = 1 << idx;
            let offset = match exponent % 3 {
                1 => domain.zeta(),
                2 => domain.zeta_inv(),
                _ => unreachable!(),
            };
            izip!((0..).step_by(exponent), s_polys.iter_mut()).for_each(|(idx, value)| {
                *value = offset * omegas[idx % domain.extended_n()] - F::ONE
            });
            s_polys.batch_invert();
            izip!(s_polys.iter_mut(), vanishing.iter().cycle())
                .for_each(|(denom, numer)| *denom *= numer);
        })
    });
    s_polys
}

fn s_evals<F: WithSmallOrderMulGroup<3>>(
    domain: &Radix2Domain<F>,
    poly: usize,
    x: F,
) -> Vec<(Query, F)> {
    let iter = &mut squares(x).map(|square_of_x| square_of_x - F::ONE);
    let mut s_denom_evals = iter.take(domain.k()).collect_vec();
    let vanishing_eval = iter.next().unwrap();
    s_denom_evals.batch_invert();
    let s_evals = s_denom_evals.iter().map(|denom| vanishing_eval * denom);
    izip!((poly..).map(|poly| (poly, 0).into()), s_evals).collect()
}

fn vanishing_eval<F: WithSmallOrderMulGroup<3>>(domain: &Radix2Domain<F>, x: F) -> F {
    x.pow([domain.n() as u64]) - F::ONE
}
