use crate::{
    backend::hyperplonk::{
        folding::{protostar::verifier::ProtostarInstance, sangria::prover::HadamardEvaluator},
        prover::instance_polys,
    },
    pcs::{AdditiveCommitment, Polynomial, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, BooleanHypercube, PrimeField},
        expression::{evaluator::ExpressionRegistry, Expression},
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize_iter},
        Itertools,
    },
};
use std::{iter, slice};

pub(super) use crate::backend::hyperplonk::folding::sangria::prover::lookup_h_polys;

#[derive(Debug)]
pub(crate) struct ProtostarWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) instance: ProtostarInstance<F, Pcs>,
    pub(crate) witness_polys: Vec<Pcs::Polynomial>,
    pub(crate) zeta_e_poly: Pcs::Polynomial,
}

impl<F, Pcs> ProtostarWitness<F, Pcs>
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
            instance: ProtostarInstance::init(num_instances, num_witness_polys, num_challenges),
            witness_polys: iter::repeat_with(|| zero_poly.clone())
                .take(num_witness_polys)
                .collect(),
            zeta_e_poly: zero_poly,
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
            instance: ProtostarInstance::from_committed(instances, witness_comms, challenges),
            witness_polys: witness_polys.into_iter().collect(),
            zeta_e_poly: Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]),
        }
    }
}

impl<F, Pcs> ProtostarWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
    Pcs::Commitment: AdditiveCommitment<F>,
{
    pub(crate) fn fold(
        &mut self,
        rhs: &Self,
        compressed_cross_term_sums: &[F],
        zeta_cross_term_poly: &Pcs::Polynomial,
        zeta_cross_term_comm: &Pcs::Commitment,
        r: &F,
    ) {
        self.instance.fold(
            &rhs.instance,
            compressed_cross_term_sums,
            zeta_cross_term_comm,
            r,
        );
        izip_eq!(&mut self.witness_polys, &rhs.witness_polys)
            .for_each(|(lhs, rhs)| *lhs += (r, rhs));
        izip!(powers(*r).skip(1), [zeta_cross_term_poly, &rhs.zeta_e_poly])
            .for_each(|(power_of_r, poly)| self.zeta_e_poly += (&power_of_r, poly));
    }
}

pub(super) fn powers_of_zeta_poly<F: PrimeField>(
    num_vars: usize,
    zeta: F,
) -> MultilinearPolynomial<F> {
    let powers_of_zeta = powers(zeta).take(1 << num_vars).collect_vec();
    let nth_map = BooleanHypercube::new(num_vars).nth_map();
    MultilinearPolynomial::new(par_map_collect(&nth_map, |b| powers_of_zeta[*b]))
}

pub(super) fn evaluate_cross_term_sum<F, Pcs>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &ProtostarWitness<F, Pcs>,
    incoming: &ProtostarWitness<F, Pcs>,
) -> Vec<F>
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

    let ev = HadamardSumEvaluator::new(num_vars, &cross_term_expressions);
    let size = 1 << num_vars;
    let num_threads = num_threads();
    let chunk_size = div_ceil(size, num_threads);
    let num_cross_terms = cross_term_expressions.len();

    let mut partial_sums = vec![vec![F::ZERO; num_cross_terms]; num_threads];
    parallelize_iter(
        partial_sums.iter_mut().zip((0..).step_by(chunk_size)),
        |(partial_sums, start)| {
            let mut data = ev.cache();
            for b in start..(start + chunk_size).min(size) {
                ev.evaluate(partial_sums, &mut data, polys.as_slice(), b);
            }
        },
    );

    partial_sums
        .into_iter()
        .reduce(|mut sums, partial_sums| {
            sums.iter_mut()
                .zip(&partial_sums)
                .for_each(|(sum, partial_sum)| *sum += partial_sum);
            sums
        })
        .unwrap()
}

#[derive(Clone, Debug)]
pub(crate) struct HadamardSumEvaluator<F: PrimeField> {
    num_vars: usize,
    reg: ExpressionRegistry<F>,
    lagranges: Vec<usize>,
}

impl<F: PrimeField> HadamardSumEvaluator<F> {
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
        sum: &mut [F],
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
        sum.iter_mut()
            .zip(self.reg.indexed_outputs())
            .for_each(|(sum, idx)| *sum += cache[*idx])
    }
}

pub(crate) fn evaluate_zeta_cross_term<F, Pcs>(
    zeta_cross_term_expression: &Expression<F>,
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &ProtostarWitness<F, Pcs>,
    incoming: &ProtostarWitness<F, Pcs>,
) -> MultilinearPolynomial<F>
where
    F: PrimeField + Ord,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
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

    let zeta_cross_term_expression = zeta_cross_term_expression
        .simplified(Some(&challenges))
        .unwrap_or_else(Expression::zero);

    let ev = HadamardEvaluator::new(num_vars, &[zeta_cross_term_expression]);
    let size = 1 << num_vars;
    let chunk_size = div_ceil(size, num_threads());

    let mut outputs = vec![F::ZERO; 1 << num_vars];
    parallelize_iter(
        outputs
            .chunks_mut(chunk_size)
            .zip((0..).step_by(chunk_size)),
        |(outputs, start)| {
            let mut data = ev.cache();
            let bs = start..(start + chunk_size).min(size);
            for (b, output) in bs.zip(outputs.iter_mut()) {
                ev.evaluate(slice::from_mut(output), &mut data, polys.as_slice(), b);
            }
        },
    );

    MultilinearPolynomial::new(outputs)
}
