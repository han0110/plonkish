use crate::{
    backend::hyperplonk::{folding::sangria::verifier::SangriaInstance, prover::instance_polys},
    pcs::{AdditiveCommitment, Polynomial},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, sum, BatchInvert, BooleanHypercube, PrimeField},
        chain,
        expression::{evaluator::ExpressionRegistry, Expression},
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        Itertools,
    },
};
use std::{borrow::Cow, hash::Hash, iter};

#[derive(Debug)]
pub(crate) struct SangriaWitness<F, C, P> {
    pub(crate) instance: SangriaInstance<F, C>,
    pub(crate) witness_polys: Vec<P>,
    pub(crate) e_poly: P,
}

impl<F, C, P> SangriaWitness<F, C, P>
where
    F: PrimeField,
    C: Default,
    P: Polynomial<F>,
{
    pub(crate) fn init(
        k: usize,
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        let zero_poly = P::from_evals(vec![F::ZERO; 1 << k]);
        Self {
            instance: SangriaInstance::init(num_instances, num_witness_polys, num_challenges),
            witness_polys: iter::repeat_with(|| zero_poly.clone())
                .take(num_witness_polys)
                .collect(),
            e_poly: zero_poly,
        }
    }

    pub(crate) fn from_committed(
        k: usize,
        instances: &[&[F]],
        witness_polys: impl IntoIterator<Item = P>,
        witness_comms: impl IntoIterator<Item = C>,
        challenges: Vec<F>,
    ) -> Self {
        Self {
            instance: SangriaInstance::from_committed(instances, witness_comms, challenges),
            witness_polys: witness_polys.into_iter().collect(),
            e_poly: P::from_evals(vec![F::ZERO; 1 << k]),
        }
    }
}

impl<F, C, P> SangriaWitness<F, C, P>
where
    F: PrimeField,
    P: Polynomial<F>,
    C: AdditiveCommitment<F>,
{
    pub(crate) fn fold(
        &mut self,
        rhs: &Self,
        cross_term_polys: &[P],
        cross_term_comms: &[C],
        r: &F,
    ) {
        self.instance.fold(&rhs.instance, cross_term_comms, r);
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), chain![cross_term_polys, [&rhs.e_poly]])
            .for_each(|(power_of_r, poly)| self.e_poly += (&power_of_r, poly));
    }
}

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

pub(crate) fn evaluate_cross_term<F: PrimeField, C>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &SangriaWitness<F, C, MultilinearPolynomial<F>>,
    incoming: &SangriaWitness<F, C, MultilinearPolynomial<F>>,
) -> Vec<MultilinearPolynomial<F>> {
    if cross_term_expressions.is_empty() {
        return Vec::new();
    }

    let ev = init_hadamard_evaluator(
        cross_term_expressions,
        num_vars,
        preprocess_polys,
        folded,
        incoming,
    );
    evaluate_cross_term_inner(&ev)
}

pub(crate) fn evaluate_cross_term_inner<F: PrimeField>(
    ev: &HadamardEvaluator<F>,
) -> Vec<MultilinearPolynomial<F>> {
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

fn init_hadamard_evaluator<'a, F: PrimeField, C>(
    expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &'a [MultilinearPolynomial<F>],
    folded: &'a SangriaWitness<F, C, MultilinearPolynomial<F>>,
    incoming: &'a SangriaWitness<F, C, MultilinearPolynomial<F>>,
) -> HadamardEvaluator<'a, F> {
    assert!(!expressions.is_empty());

    let folded_instance_polys = instance_polys(num_vars, &folded.instance.instances);
    let incoming_instance_polys = instance_polys(num_vars, &incoming.instance.instances);
    let polys = iter::empty()
        .chain(preprocess_polys.iter().map(Cow::Borrowed))
        .chain(folded_instance_polys.into_iter().map(Cow::Owned))
        .chain(folded.witness_polys.iter().map(Cow::Borrowed))
        .chain(incoming_instance_polys.into_iter().map(Cow::Owned))
        .chain(incoming.witness_polys.iter().map(Cow::Borrowed))
        .collect_vec();
    let challenges = iter::empty()
        .chain(folded.instance.challenges.iter().cloned())
        .chain(Some(folded.instance.u))
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
