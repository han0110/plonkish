use crate::{
    backend::hyperplonk::{
        folding::{protostar::verifier::ProtostarInstance, sangria::prover::HadamardEvaluator},
        prover::instance_polys,
    },
    pcs::{AdditiveCommitment, Polynomial},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, BooleanHypercube, PrimeField},
        expression::{Expression, Rotation},
        izip, izip_eq,
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        Itertools,
    },
};
use std::{borrow::Cow, iter};

pub(super) use crate::backend::hyperplonk::folding::sangria::prover::lookup_h_polys;

#[derive(Clone, Debug)]
pub(crate) struct ProtostarWitness<F, C, P> {
    pub(crate) instance: ProtostarInstance<F, C>,
    pub(crate) witness_polys: Vec<P>,
    pub(crate) zeta_e_poly: P,
}

impl<F, C, P> ProtostarWitness<F, C, P>
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
        witness_polys: impl IntoIterator<Item = P>,
        witness_comms: impl IntoIterator<Item = C>,
        challenges: Vec<F>,
    ) -> Self {
        Self {
            instance: ProtostarInstance::from_committed(instances, witness_comms, challenges),
            witness_polys: witness_polys.into_iter().collect(),
            zeta_e_poly: P::from_evals(vec![F::ZERO; 1 << k]),
        }
    }
}

impl<F, C, P> ProtostarWitness<F, C, P>
where
    F: PrimeField,
    C: AdditiveCommitment<F>,
    P: Polynomial<F>,
{
    pub(crate) fn fold(
        &mut self,
        rhs: &Self,
        compressed_cross_term_sums: &[F],
        zeta_cross_term_poly: &P,
        zeta_cross_term_comm: &C,
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

pub(super) fn evaluate_compressed_cross_term_sum<F, C>(
    cross_term_expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &[MultilinearPolynomial<F>],
    folded: &ProtostarWitness<F, C, MultilinearPolynomial<F>>,
    incoming: &ProtostarWitness<F, C, MultilinearPolynomial<F>>,
) -> Vec<F>
where
    F: PrimeField + Ord,
{
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
    evaluate_cross_term_sum_inner(&ev)
}

pub(crate) fn evaluate_zeta_cross_term<F: PrimeField, C>(
    num_vars: usize,
    zeta_nth_back: usize,
    folded: &ProtostarWitness<F, C, MultilinearPolynomial<F>>,
    incoming: &ProtostarWitness<F, C, MultilinearPolynomial<F>>,
) -> MultilinearPolynomial<F> {
    let [(folded_pow, folded_zeta, folded_u), (incoming_pow, incoming_zeta, incoming_u)] =
        [folded, incoming].map(|witness| {
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
                *cross_term = folded_pow[next_map[b]] + folded_u * incoming_pow[next_map[b]]
                    - (folded_pow[b] * incoming_zeta + incoming_pow[b] * folded_zeta);
            })
    });
    let b_0 = 0;
    let b_last = bh.rotate(1, Rotation::prev());
    cross_term[b_0] +=
        folded_pow[b_0] * incoming_zeta + incoming_pow[b_0] * folded_zeta - folded_u.double();
    cross_term[b_last] += folded_pow[b_last] * incoming_zeta + incoming_pow[b_last] * folded_zeta
        - folded_u * incoming_zeta
        - folded_zeta;

    MultilinearPolynomial::new(cross_term)
}

fn init_hadamard_evaluator<'a, F: PrimeField, C>(
    expressions: &[Expression<F>],
    num_vars: usize,
    preprocess_polys: &'a [MultilinearPolynomial<F>],
    folded: &'a ProtostarWitness<F, C, MultilinearPolynomial<F>>,
    incoming: &'a ProtostarWitness<F, C, MultilinearPolynomial<F>>,
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

fn evaluate_cross_term_sum_inner<F: PrimeField>(ev: &HadamardEvaluator<F>) -> Vec<F> {
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
