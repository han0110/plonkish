use crate::{
    backend::hyperplonk::{folding::sangria::verifier::SangriaInstance, prover::instance_polys},
    pcs::{AdditiveCommitment, Polynomial, PolynomialCommitmentScheme},
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
use std::{hash::Hash, iter};

#[derive(Debug)]
pub(crate) struct SangriaWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) instance: SangriaInstance<F, Pcs>,
    pub(crate) witness_polys: Vec<Pcs::Polynomial>,
    pub(crate) e_poly: Pcs::Polynomial,
}

impl<F, Pcs> SangriaWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) fn init(
        k: usize,
        num_instances: &[usize],
        num_witness_polys: usize,
        num_challenges: usize,
    ) -> Self {
        let zero_poly = Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]);
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
        witness_polys: impl IntoIterator<Item = Pcs::Polynomial>,
        witness_comms: impl IntoIterator<Item = Pcs::Commitment>,
        challenges: Vec<F>,
    ) -> Self {
        Self {
            instance: SangriaInstance::from_committed(instances, witness_comms, challenges),
            witness_polys: witness_polys.into_iter().collect(),
            e_poly: Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]),
        }
    }
}

impl<F, Pcs> SangriaWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: AdditiveCommitment<F>,
{
    pub(crate) fn fold(
        &mut self,
        rhs: &Self,
        cross_term_polys: &[Pcs::Polynomial],
        cross_term_comms: &[Pcs::Commitment],
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

pub(crate) fn evaluate_cross_term<F, Pcs>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &SangriaWitness<F, Pcs>,
    incoming: &SangriaWitness<F, Pcs>,
) -> Vec<MultilinearPolynomial<F>>
where
    F: PrimeField + Ord,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    if cross_term_expressions.is_empty() {
        return Vec::new();
    }

    let folded_instance_polys = instance_polys(num_vars, &folded.instance.instances);
    let incoming_instance_polys = instance_polys(num_vars, &incoming.instance.instances);
    let polys = iter::empty()
        .chain(preprocess_polys)
        .chain(&folded_instance_polys)
        .chain(&folded.witness_polys)
        .chain(&incoming_instance_polys)
        .chain(&incoming.witness_polys)
        .collect_vec();
    let challenges = iter::empty()
        .chain(folded.instance.challenges.iter().cloned())
        .chain(Some(folded.instance.u))
        .chain(incoming.instance.challenges.iter().cloned())
        .chain(Some(incoming.instance.u))
        .collect_vec();

    let cross_term_expressions = cross_term_expressions
        .iter()
        .map(|expression| {
            expression
                .simplified(Some(&challenges))
                .unwrap_or_else(Expression::zero)
        })
        .collect_vec();

    let ev = HadamardEvaluator::new(num_vars, &cross_term_expressions);
    let size = 1 << num_vars;
    let chunk_size = div_ceil(size, num_threads());
    let num_cross_terms = cross_term_expressions.len();

    let mut outputs = vec![F::ZERO; num_cross_terms << num_vars];
    parallelize_iter(
        outputs
            .chunks_mut(chunk_size * num_cross_terms)
            .zip((0..).step_by(chunk_size)),
        |(outputs, start)| {
            let mut data = ev.cache();
            let bs = start..(start + chunk_size).min(size);
            for (b, outputs) in bs.zip(outputs.chunks_mut(num_cross_terms)) {
                ev.evaluate(outputs, &mut data, polys.as_slice(), b);
            }
        },
    );

    (0..num_cross_terms)
        .map(|offset| par_map_collect(0..size, |idx| outputs[idx * num_cross_terms + offset]))
        .map(MultilinearPolynomial::new)
        .collect_vec()
}

#[derive(Clone, Debug)]
pub(crate) struct HadamardEvaluator<F: PrimeField> {
    num_vars: usize,
    reg: ExpressionRegistry<F>,
    lagranges: Vec<usize>,
}

impl<F: PrimeField> HadamardEvaluator<F> {
    pub(crate) fn new(num_vars: usize, expressions: &[Expression<F>]) -> Self {
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
        }
    }

    pub(crate) fn cache(&self) -> Vec<F> {
        self.reg.cache()
    }

    pub(crate) fn evaluate(
        &self,
        evals: &mut [F],
        cache: &mut [F],
        polys: &[&MultilinearPolynomial<F>],
        b: usize,
    ) {
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
                *value = polys[query.poly()][bh.rotate(b, query.rotation())]
            });
        self.reg
            .indexed_calculations()
            .iter()
            .zip(self.reg.offsets().calculations()..)
            .for_each(|(calculation, idx)| calculation.calculate(cache, idx));
        evals
            .iter_mut()
            .zip(self.reg.indexed_outputs())
            .for_each(|(eval, idx)| *eval = cache[*idx])
    }
}
