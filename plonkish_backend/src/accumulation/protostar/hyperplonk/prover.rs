use crate::{
    accumulation::protostar::ProtostarAccumulator,
    backend::hyperplonk::prover::instance_polys,
    pcs::PolynomialCommitmentScheme,
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, powers, sum, BatchInvert, BooleanHypercube, PrimeField},
        expression::{evaluator::ExpressionRegistry, Expression, Rotation},
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        Itertools,
    },
};
use std::{borrow::Cow, hash::Hash, iter::{self, repeat}};

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
        iter::empty()
            .chain(h_input.chunks_mut(chunk_size))
            .chain(h_table.chunks_mut(chunk_size)),
        |h| {
            h.iter_mut().batch_invert();
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

pub(crate) fn powers_of_zeta_poly<F: PrimeField>(
    num_vars: usize,
    zeta: F,
) -> MultilinearPolynomial<F> {
    let powers_of_zeta: Vec<F> = powers(zeta).take(1 << num_vars).collect_vec();
    let nth_map = BooleanHypercube::new(num_vars).nth_map();
    MultilinearPolynomial::new(par_map_collect(&nth_map, |b| powers_of_zeta[*b]))
}

pub(crate) fn powers_of_zeta_sum_check_poly_hi<F: PrimeField>(
    num_vars: usize,
    zeta_pow_lsqrt: F,
) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
    let nth_map = BooleanHypercube::new(num_vars).nth_map();
    let powers_of_zeta_hi = powers(zeta_pow_lsqrt).take(1 << num_vars).collect_vec();
    let powers_of_zeta_hi_poly = MultilinearPolynomial::new(par_map_collect(&nth_map, |b| powers_of_zeta_hi[*b]));
    ( powers_of_zeta_hi_poly.clone(),
        MultilinearPolynomial::new(powers_of_zeta_hi_poly.iter().flat_map(|zeta| repeat(zeta).take(1 << num_vars)).cloned().collect_vec()))
}

pub(crate) fn powers_of_zeta_sum_check_poly_lo<F: PrimeField>(
    num_vars: usize,
    zeta: F,
) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
    let nth_map = BooleanHypercube::new(num_vars).nth_map();
    let powers_of_zeta_lo = powers(zeta).take(1 << num_vars).collect_vec();
    let powers_of_zeta_lo_poly = MultilinearPolynomial::new(par_map_collect(&nth_map, |b| powers_of_zeta_lo[*b]));
    ( powers_of_zeta_lo_poly.clone(),
        MultilinearPolynomial::new(repeat(powers_of_zeta_lo_poly.clone().into_evals()).take(1 << num_vars).flatten().collect_vec()))
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

    let ev = init_hadamard_evaluator(
        cross_term_expressions,
        num_vars,
        preprocess_polys,
        accumulator,
        incoming,
    );

    let size = 1 << ev.num_vars;
    let chunk_size = div_ceil(size, num_threads());
    let num_cross_terms = ev.reg.indexed_outputs().len();

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

pub(crate) fn evaluate_compressed_cross_term_sums<F, Pcs>(
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
    let num_cross_terms = ev.reg.indexed_outputs().len();

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

    let bh = BooleanHypercube::new(num_vars);
    let next_map = bh.rotation_map(Rotation::next());
    parallelize(&mut cross_term, |(cross_term, start)| {
        cross_term
            .iter_mut()
            .zip(start..)
            .for_each(|(cross_term, b)| {
                *cross_term = acc_pow[next_map[b]] + acc_u * incoming_pow[next_map[b]]
                     - (acc_pow[b] * incoming_zeta + incoming_pow[b] * acc_zeta);
            })
    });
    let b_0 = 0;
    let b_last = bh.rotate(1, Rotation::prev());
    cross_term[b_0] += acc_pow[b_0] * incoming_zeta + incoming_pow[b_0] * acc_zeta - acc_u.double();
    cross_term[b_last] += acc_pow[b_last] * incoming_zeta + incoming_pow[b_last] * acc_zeta
        - acc_u * incoming_zeta
        - acc_zeta;

    MultilinearPolynomial::new(cross_term)
}

pub(crate) fn evaluate_zeta_root_cross_term_poly<F, Pcs>(
    num_vars: usize,
    zeta_nth_back: usize,
    accumulator: &ProtostarAccumulator<F, Pcs>,
    incoming: &ProtostarAccumulator<F, Pcs>,
) -> MultilinearPolynomial<F>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    let [(acc_pow_lo, acc_pow_hi, acc_zeta, acc_zeta_sqrt, acc_u), 
        (incoming_pow_lo, incoming_pow_hi, incoming_zeta, incoming_zeta_sqrt, incoming_u)] =
        [accumulator, incoming]
        .map(|witness| {
            let pow_lo = witness.witness_polys[witness.witness_polys.len() - 2].clone().into_evals().iter()
                .cloned()
                .take(1 << num_vars)
                .collect_vec();

            let pow_hi = witness.witness_polys.last().unwrap().clone().into_evals().iter()
                .step_by(1 << num_vars)
                .cloned()
                .collect_vec();

            let zeta = witness
                .instance
                .challenges
                .iter()
                .nth_back(zeta_nth_back + 1)
                .unwrap();

            let zeta_sqrt = witness
                .instance
                .challenges
                .iter()
                .nth_back(zeta_nth_back)
                .unwrap();

            (pow_lo, pow_hi, zeta, zeta_sqrt, witness.instance.u)
        });
    assert_eq!(incoming_u, F::ONE);

    let size = 1 << num_vars;
    let b_0 = 0;
    let bh = BooleanHypercube::new(num_vars);
    let next_map = bh.rotation_map(Rotation::next());
    let b_last = bh.rotate(1, Rotation::prev());

    let mut cross_term_lo = vec![F::ZERO; size];
    parallelize(&mut cross_term_lo, |(cross_term_lo, start)| {
        cross_term_lo
            .iter_mut()
            .zip(start..)
            .for_each(|(cross_term_lo, b)| {
                *cross_term_lo = acc_pow_lo[next_map[b]] + acc_u * incoming_pow_lo[next_map[b]]
                     - (acc_pow_lo[b] * incoming_zeta + incoming_pow_lo[b] * acc_zeta);
            })
    });
    cross_term_lo[b_0] += acc_pow_lo[b_0] * incoming_zeta + incoming_pow_lo[b_0] * acc_zeta - acc_u.double();
    cross_term_lo[b_last] += acc_pow_lo[b_last] * incoming_zeta + incoming_pow_lo[b_last] * acc_zeta
        - acc_u * incoming_zeta
        - acc_zeta;

    let mut cross_term_hi = vec![F::ZERO; size];
    parallelize(&mut cross_term_hi, |(cross_term_hi, start)| {
        cross_term_hi
            .iter_mut()
            .zip(start..)
            .for_each(|(cross_term_hi, b)| {
                *cross_term_hi = acc_pow_hi[next_map[b]] + acc_u * incoming_pow_hi[next_map[b]]
                     - (acc_pow_hi[b] * incoming_zeta_sqrt + incoming_pow_hi[b] * acc_zeta_sqrt);
            })
    });
    cross_term_hi[b_0] += acc_pow_hi[b_0] * incoming_zeta_sqrt + incoming_pow_hi[b_0] * acc_zeta_sqrt - acc_u.double();
    cross_term_hi[b_last] += acc_pow_hi[b_last] * incoming_zeta_sqrt + incoming_pow_hi[b_last] * acc_zeta_sqrt
        - acc_u * incoming_zeta_sqrt
        - acc_zeta_sqrt;

    let cross_term = [cross_term_lo, cross_term_hi].concat();
    MultilinearPolynomial::new(cross_term)
}

fn init_hadamard_evaluator<'a, F, Pcs>(
    expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &'a [MultilinearPolynomial<F>],
    accumulator: &'a ProtostarAccumulator<F, Pcs>,
    incoming: &'a ProtostarAccumulator<F, Pcs>,
) -> HadamardEvaluator<'a, F>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    assert!(!expressions.is_empty());

    let acc_instance_polys = instance_polys(num_vars, &accumulator.instance.instances);
    let incoming_instance_polys = instance_polys(num_vars, &incoming.instance.instances);
    let polys = iter::empty()
        .chain(preprocess_polys.iter().map(Cow::Borrowed))
        .chain(acc_instance_polys.into_iter().map(Cow::Owned))
        .chain(accumulator.witness_polys.iter().map(Cow::Borrowed))
        .chain(incoming_instance_polys.into_iter().map(Cow::Owned))
        .chain(incoming.witness_polys.iter().map(Cow::Borrowed))
        .collect_vec();
    let challenges = iter::empty()
        .chain(accumulator.instance.challenges.iter().cloned())
        .chain(Some(accumulator.instance.u))
        .chain(incoming.instance.challenges.iter().cloned())
        .chain(Some(incoming.instance.u))
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

#[derive(Clone, Debug)]
pub(crate) struct HadamardEvaluator<'a, F: PrimeField> {
    pub(crate) num_vars: usize,
    pub(crate) reg: ExpressionRegistry<F>,
    lagranges: Vec<usize>,
    polys: Vec<Cow<'a, MultilinearPolynomial<F>>>,
}

impl<'a, F: PrimeField> HadamardEvaluator<'a, F> {
    pub(crate) fn new(
        num_vars: usize,
        expressions: &[Expression<F>],
        polys: Vec<Cow<'a, MultilinearPolynomial<F>>>,
    ) -> Self {
        let mut reg = ExpressionRegistry::new();
        for expression in expressions.iter() {
            reg.register(expression);
        }
        assert!(reg.eq_xys().is_empty());

        let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
        let lagranges = reg
            .lagranges()
            .iter()
            .map(|i| bh[i.rem_euclid(1 << num_vars) as usize])
            .collect_vec();

        Self {
            num_vars,
            reg,
            lagranges,
            polys,
        }
    }

    pub(crate) fn cache(&self) -> Vec<F> {
        self.reg.cache()
    }

    pub(crate) fn evaluate(&self, evals: &mut [F], cache: &mut [F], b: usize) {
        self.evaluate_calculations(cache, b);
        izip_eq!(evals, self.reg.indexed_outputs()).for_each(|(eval, idx)| *eval = cache[*idx])
    }

    pub(crate) fn evaluate_and_sum(&self, sums: &mut [F], cache: &mut [F], b: usize) {
        self.evaluate_calculations(cache, b);
        izip_eq!(sums, self.reg.indexed_outputs()).for_each(|(sum, idx)| *sum += cache[*idx])
    }

    fn evaluate_calculations(&self, cache: &mut [F], b: usize) {
        let bh = BooleanHypercube::new(self.num_vars);
        if self.reg.has_identity() {
            cache[self.reg.offsets().identity()] = F::from(b as u64);
        }
        cache[self.reg.offsets().lagranges()..]
            .iter_mut()
            .zip(&self.lagranges)
            .for_each(|(value, i)| *value = if &b == i { F::ONE } else { F::ZERO });
        cache[self.reg.offsets().polys()..]
            .iter_mut()
            .zip(self.reg.polys())
            .for_each(|(value, (query, _))| {
                *value = self.polys[query.poly()][bh.rotate(b, query.rotation())]
            });
        self.reg
            .indexed_calculations()
            .iter()
            .zip(self.reg.offsets().calculations()..)
            .for_each(|(calculation, idx)| calculation.calculate(cache, idx));
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        poly::{
            multilinear::{rotation_eval, zip_self, MultilinearPolynomial, test::{fix_vars}},
            Polynomial,
        },
        util::{
            arithmetic::{BooleanHypercube, Field, powers},
            expression::Rotation,
            test::rand_vec,
            Itertools,
        }, accumulation::protostar::hyperplonk::prover::{powers_of_zeta_poly, powers_of_zeta_sum_check_poly_lo, powers_of_zeta_sum_check_poly_hi},
    };
    use halo2_curves::bn256::Fr;
    use rand::{rngs::OsRng, RngCore};
    use core::num;
    use std::{iter::{self, repeat}, ops::Mul};

    #[test]
    fn eval_zeta_sqrt() {
   
        let rand_x_i = || match OsRng.next_u32() % 3 {
            0 => Fr::zero(),
            1 => Fr::one(),
            2 => Fr::random(OsRng),
            _ => unreachable!(),
        };

            let mut rng = OsRng;
            let num_vars = 8;
            let zeta = Fr::random(OsRng);
            let l_sqrt = 1 << (num_vars/2);
            let zeta_pow_lsqrt = zeta.pow(&[l_sqrt as u64]);

            // committed powers_of_zeta_polys
            let powers_of_zeta_root_lo_poly = powers_of_zeta_poly(num_vars/2, zeta);
            let powers_of_zeta_root_hi_poly = powers_of_zeta_poly(num_vars/2, zeta_pow_lsqrt);

            // sum_check witness polys
            let powers_of_zeta_sum_check_lo = MultilinearPolynomial::new(repeat(powers_of_zeta_root_lo_poly.clone().into_evals()).take(1 << num_vars/2).flatten().collect_vec());
            let powers_of_zeta_sum_check_hi = MultilinearPolynomial::new(powers_of_zeta_root_hi_poly.iter().flat_map(|zeta| repeat(zeta).take(1 << num_vars/2)).cloned().collect_vec());
            
            // full num_vars
            let x = iter::repeat_with(rand_x_i).take(num_vars).collect_vec();
            // half num_vars used to fix the variables in the sum_check witness polys
            let x_rand = iter::repeat_with(rand_x_i).take(num_vars/2).collect_vec();
            // split the query point x into two halves
            let (x_lo, x_hi) = x.split_at(num_vars/2);

            // evaluation of the sum_check witness polys with half of num_vars fixed
            let eval_lo_fix_vars = powers_of_zeta_sum_check_lo.fix_last_vars(&x_rand).evaluate(&x_lo);   
            let eval_hi_fix_vars = MultilinearPolynomial::new(fix_vars(powers_of_zeta_sum_check_hi.evals(), &x_rand)).evaluate(&x_hi); 

            assert_eq!(powers_of_zeta_root_lo_poly.evaluate(&x_lo), eval_lo_fix_vars);
            assert_eq!(powers_of_zeta_root_hi_poly.evaluate(&x_hi), eval_hi_fix_vars);

    }
}