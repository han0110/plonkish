use crate::{
    backend::hyperplonk::{
        folding::{protostar::verifier::ProtostarInstance, sangria::prover::HadamardEvaluator},
        prover::instance_polys,
    },
    pcs::{AdditiveCommitment, Polynomial, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, PrimeField},
        chain,
        expression::Expression,
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize_iter},
        Itertools,
    },
};
use std::iter;

pub(super) use crate::backend::hyperplonk::folding::sangria::prover::lookup_h_polys;

#[derive(Debug)]
pub(crate) struct ProtostarWitness<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) instance: ProtostarInstance<F, Pcs>,
    pub(crate) witness_polys: Vec<Pcs::Polynomial>,
    pub(crate) e_poly: Pcs::Polynomial,
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
            instance: ProtostarInstance::from_committed(instances, witness_comms, challenges),
            witness_polys: witness_polys.into_iter().collect(),
            e_poly: Pcs::Polynomial::from_evals(vec![F::ZERO; 1 << k]),
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

pub(super) fn evaluate_cross_term<F, Pcs>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &ProtostarWitness<F, Pcs>,
    incoming: &ProtostarWitness<F, Pcs>,
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
