use crate::{
    accumulation::protostar::ProtostarAccumulator,
    backend::hyperplonk::prover::instance_polys,
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, sum, BatchInvert, PrimeField},
        chain,
        expression::{
            evaluator::hadamard::HadamardEvaluator,
            rotate::{BinaryField, Rotatable},
            Expression, Rotation,
        },
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        Itertools,
    },
};
use std::{borrow::Cow, hash::Hash};

pub(crate) fn lookup_h_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    m_polys: &[MultilinearPolynomial<F>],
    beta: &F,
) -> Vec<[MultilinearPolynomial<F>; 2]> {
    compressed_polys
        .iter()
        .zip(m_polys.iter())
        .map(|(compressed_polys, m_poly)| lookup_h_poly(compressed_polys, m_poly, beta))
        .collect()
}

fn lookup_h_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
    m_poly: &MultilinearPolynomial<F>,
    beta: &F,
) -> [MultilinearPolynomial<F>; 2] {
    let [input, table] = compressed_polys;
    let mut h_input = vec![F::ZERO; 1 << input.num_vars()];
    let mut h_table = vec![F::ZERO; 1 << input.num_vars()];

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, input) in h_input.iter_mut().zip(input[start..].iter()) {
            *h_input = *beta + input;
        }
    });
    parallelize(&mut h_table, |(h_table, start)| {
        for (h_table, table) in h_table.iter_mut().zip(table[start..].iter()) {
            *h_table = *beta + table;
        }
    });

    let chunk_size = div_ceil(2 * h_input.len(), num_threads());
    parallelize_iter(
        chain![
            h_input.chunks_mut(chunk_size),
            h_table.chunks_mut(chunk_size)
        ],
        |h| {
            h.batch_invert();
        },
    );

    parallelize(&mut h_table, |(h_table, start)| {
        for (h_table, m) in h_table.iter_mut().zip(m_poly[start..].iter()) {
            *h_table *= m;
        }
    });

    if cfg!(feature = "sanity-check") {
        assert_eq!(sum::<F>(&h_input), sum::<F>(&h_table));
    }

    [
        MultilinearPolynomial::new(h_input),
        MultilinearPolynomial::new(h_table),
    ]
}

pub(super) fn powers_of_zeta_poly<F: PrimeField>(
    num_vars: usize,
    zeta: F,
) -> MultilinearPolynomial<F> {
    let powers_of_zeta = chain![[F::ZERO], powers(zeta)]
        .take(1 << num_vars)
        .collect_vec();
    let nth_map = BinaryField::new(num_vars).nth_map();
    MultilinearPolynomial::new(par_map_collect(&nth_map, |b| powers_of_zeta[*b]))
}

pub(crate) fn evaluate_cross_term_polys<F, Pcs>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    accumulator: &ProtostarAccumulator<F, Pcs>,
    incoming: &ProtostarAccumulator<F, Pcs>,
) -> Vec<MultilinearPolynomial<F>>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    if cross_term_expressions.is_empty() {
        return Vec::new();
    }

    let num_cross_terms = cross_term_expressions.len();
    if accumulator.instance.u.is_zero_vartime() {
        return vec![MultilinearPolynomial::new(vec![F::ZERO; 1 << num_vars]); num_cross_terms];
    }

    let ev = init_hadamard_evaluator(
        cross_term_expressions,
        num_vars,
        preprocess_polys,
        accumulator,
        incoming,
    );

    let size = 1 << ev.num_vars;
    let chunk_size = div_ceil(size, num_threads());

    let mut outputs = vec![F::ZERO; num_cross_terms * size];
    parallelize_iter(
        outputs
            .chunks_mut(chunk_size * num_cross_terms)
            .zip((0..).step_by(chunk_size)),
        |(outputs, start)| {
            let mut data = ev.cache();
            let bs = start..(start + chunk_size).min(size);
            izip!(bs, outputs.chunks_mut(num_cross_terms))
                .for_each(|(b, outputs)| ev.evaluate(outputs, &mut data, b));
        },
    );

    (0..num_cross_terms)
        .map(|offset| par_map_collect(0..size, |idx| outputs[idx * num_cross_terms + offset]))
        .map(MultilinearPolynomial::new)
        .collect_vec()
}

pub(super) fn evaluate_compressed_cross_term_sums<F, Pcs>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    accumulator: &ProtostarAccumulator<F, Pcs>,
    incoming: &ProtostarAccumulator<F, Pcs>,
) -> Vec<F>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    if cross_term_expressions.is_empty() {
        return Vec::new();
    }

    let num_cross_terms = cross_term_expressions.len();
    if accumulator.instance.u.is_zero_vartime() {
        return vec![F::ZERO; num_cross_terms];
    }

    let ev = init_hadamard_evaluator(
        cross_term_expressions,
        num_vars,
        preprocess_polys,
        accumulator,
        incoming,
    );

    let size = 1 << ev.num_vars;
    let num_threads = num_threads();
    let chunk_size = div_ceil(size, num_threads);

    let mut partial_sums = vec![vec![F::ZERO; num_cross_terms]; num_threads];
    parallelize_iter(
        partial_sums.iter_mut().zip((0..).step_by(chunk_size)),
        |(partial_sums, start)| {
            let mut data = ev.cache();
            (start..(start + chunk_size).min(size))
                .for_each(|b| ev.evaluate_and_sum(partial_sums, &mut data, b))
        },
    );

    partial_sums
        .into_iter()
        .reduce(|mut sums, partial_sums| {
            izip_eq!(&mut sums, &partial_sums).for_each(|(sum, partial_sum)| *sum += partial_sum);
            sums
        })
        .unwrap()
}

pub(crate) fn evaluate_zeta_cross_term_poly<F, Pcs>(
    num_vars: usize,
    zeta_nth_back: usize,
    accumulator: &ProtostarAccumulator<F, Pcs>,
    incoming: &ProtostarAccumulator<F, Pcs>,
) -> MultilinearPolynomial<F>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    if accumulator.instance.u.is_zero_vartime() {
        return MultilinearPolynomial::new(vec![F::ZERO; 1 << num_vars]);
    }

    let [(acc_pow, acc_zeta, acc_u), (incoming_pow, incoming_zeta, incoming_u)] =
        [accumulator, incoming].map(|witness| {
            let pow = witness.witness_polys.last().unwrap();
            let zeta = witness
                .instance
                .challenges
                .iter()
                .nth_back(zeta_nth_back)
                .unwrap();
            (pow, zeta, witness.instance.u)
        });
    assert_eq!(incoming_u, F::ONE);

    let size = 1 << num_vars;
    let mut cross_term = vec![F::ZERO; size];

    let bf = BinaryField::new(num_vars);
    let next_map = bf.rotation_map(Rotation::next());
    parallelize(&mut cross_term, |(cross_term, start)| {
        cross_term
            .iter_mut()
            .zip(start..)
            .for_each(|(cross_term, b)| {
                *cross_term = acc_pow[next_map[b]] + acc_u * incoming_pow[next_map[b]]
                    - (acc_pow[b] * incoming_zeta + incoming_pow[b] * acc_zeta);
            })
    });
    let b_last = bf.rotate(1, Rotation::prev());
    cross_term[b_last] +=
        acc_pow[b_last] * incoming_zeta + incoming_pow[b_last] * acc_zeta - acc_u.double();

    MultilinearPolynomial::new(cross_term)
}

fn init_hadamard_evaluator<'a, F, Pcs>(
    expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &'a [MultilinearPolynomial<F>],
    accumulator: &'a ProtostarAccumulator<F, Pcs>,
    incoming: &'a ProtostarAccumulator<F, Pcs>,
) -> HadamardEvaluator<'a, F, BinaryField>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    assert!(!expressions.is_empty());

    let accumulator_instance_polys =
        instance_polys::<_, BinaryField>(num_vars, &accumulator.instance.instances);
    let incoming_instance_polys =
        instance_polys::<_, BinaryField>(num_vars, &incoming.instance.instances);
    let polys = chain![
        chain![preprocess_polys].map(|poly| Cow::Borrowed(poly.evals())),
        chain![accumulator_instance_polys].map(|poly| Cow::Owned(poly.into_evals())),
        chain![&accumulator.witness_polys].map(|poly| Cow::Borrowed(poly.evals())),
        chain![incoming_instance_polys].map(|poly| Cow::Owned(poly.into_evals())),
        chain![&incoming.witness_polys].map(|poly| Cow::Borrowed(poly.evals())),
    ]
    .collect_vec();
    let challenges = chain![
        accumulator.instance.challenges.iter().cloned(),
        [accumulator.instance.u],
        incoming.instance.challenges.iter().cloned(),
        [incoming.instance.u],
    ]
    .collect_vec();

    let expressions = expressions
        .iter()
        .map(|expression| {
            expression
                .simplified(Some(&challenges))
                .unwrap_or_else(Expression::zero)
        })
        .collect_vec();

    HadamardEvaluator::new(num_vars, &expressions, polys)
}
