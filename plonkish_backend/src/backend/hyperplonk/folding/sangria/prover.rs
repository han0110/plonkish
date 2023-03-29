use crate::{
    backend::hyperplonk::{folding::sangria::verifier::SangriaInstance, prover::instance_polys},
    pcs::{AdditiveCommitment, Polynomial, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, powers, sum, BatchInvert, BooleanHypercube, PrimeField},
        chain,
        expression::{CommonPolynomial, Expression, Query},
        izip, izip_eq,
        parallel::{num_threads, parallelize, parallelize_iter},
        Itertools,
    },
};
use std::{collections::BTreeSet, hash::Hash, iter, ops::Deref};

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

pub(super) fn lookup_h_polys<F: PrimeField + Hash>(
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

pub fn lookup_h_poly<F: PrimeField + Hash>(
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

pub(super) fn evaluate_cross_term<F, Pcs>(
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

    let mut cross_term_polys = vec![
        MultilinearPolynomial::new(vec![F::ZERO; 1 << num_vars]);
        cross_term_expressions.len()
    ];

    let (denses, sparses) = cross_term_expressions
        .iter()
        .map(|expr| split_sparse(expr, &polys))
        .unzip::<_, _, Vec<_>, Vec<_>>();
    let ev = GraphEvaluator::new(num_vars, &denses, false);
    let sparse_evs = sparses
        .iter()
        .map(|sparse| {
            sparse
                .iter()
                .map(|sparse| GraphEvaluator::new(num_vars, &[sparse.clone()], true))
                .collect_vec()
        })
        .collect_vec();

    let size = 1 << num_vars;
    let chunk_size = div_ceil(size, num_threads());
    let num_chunks = div_ceil(size, chunk_size);
    let mut chunks = cross_term_polys
        .iter_mut()
        .map(|poly| poly[..].chunks_mut(chunk_size))
        .fold(
            iter::repeat_with(Vec::new).take(num_chunks).collect_vec(),
            |mut acc, chunks| {
                for (acc, chunk) in acc.iter_mut().zip(chunks) {
                    acc.push(chunk);
                }
                acc
            },
        );
    parallelize_iter(
        chunks.iter_mut().zip((0..).step_by(chunk_size)),
        |(chunk, start)| {
            let bs = start..(start + chunk_size).min(size);
            let mut data = ev.data();
            bs.enumerate().for_each(|(idx, b)| {
                ev.evaluate(
                    chunk.iter_mut().map(|poly| &mut poly[idx]),
                    &mut data,
                    polys.as_slice(),
                    b,
                );
            })
        },
    );

    for (cross_term_poly, evs) in cross_term_polys.iter_mut().zip(sparse_evs) {
        for ev in evs {
            let mut data = ev.data();
            for b in ev.sparse_bs(&polys).unwrap() {
                ev.evaluate(
                    Some(&mut cross_term_poly[b]).into_iter(),
                    &mut data,
                    &polys,
                    b,
                );
            }
        }
    }

    cross_term_polys
}

// TODO: Refactor into mod
#[derive(Clone, Debug, Default)]
struct GraphEvaluator<F: PrimeField> {
    num_vars: usize,
    bh: Vec<usize>,
    constants: Vec<F>,
    lagranges: Vec<usize>,
    identities: Vec<usize>,
    queries: Vec<Query>,
    calculations: Vec<Calculation<ValueSource>>,
    indexed_calculations: Vec<Calculation<usize>>,
    results: Vec<usize>,
    offsets: (usize, usize, usize, usize),
    sparse: Option<Expression<F>>,
}

impl<F: PrimeField> GraphEvaluator<F> {
    fn new(num_vars: usize, expressions: &[Expression<F>], is_sparse: bool) -> Self {
        let mut ev = Self {
            num_vars,
            bh: BooleanHypercube::new(num_vars).iter().collect(),
            constants: vec![F::ZERO, F::ONE, F::ONE.double()],
            ..Default::default()
        };

        let results = expressions
            .iter()
            .map(|expression| {
                let simplified = expression.simplified(None).unwrap_or_else(Expression::zero);
                ev.register_expression(&simplified)
            })
            .collect_vec();
        ev.offsets.0 = ev.constants.len();
        ev.offsets.1 = ev.offsets.0 + ev.lagranges.len();
        ev.offsets.2 = ev.offsets.1 + ev.identities.len();
        ev.offsets.3 = ev.offsets.2 + ev.queries.len();
        ev.indexed_calculations = ev
            .calculations
            .iter()
            .map(|calculation| calculation.indexed(&ev.offsets))
            .collect();
        ev.results = results
            .iter()
            .map(|result| result.indexed(&ev.offsets))
            .collect();

        if is_sparse {
            assert_eq!(expressions.len(), 1);
            ev.sparse = Some(expressions[0].clone())
        }

        ev
    }

    fn register<T: Eq + Clone>(
        &mut self,
        field: impl FnOnce(&mut Self) -> &mut Vec<T>,
        item: &T,
    ) -> usize {
        let field = field(self);
        if let Some(idx) = field.iter().position(|lhs| lhs == item) {
            idx
        } else {
            let idx = field.len();
            field.push(item.clone());
            idx
        }
    }

    fn register_constant(&mut self, constant: &F) -> ValueSource {
        ValueSource::Constant(self.register(|ev| &mut ev.constants, constant))
    }

    fn register_lagrange(&mut self, i: i32) -> ValueSource {
        let i = self.bh[i.rem_euclid(1 << self.num_vars) as usize];
        ValueSource::Lagrange(self.register(|ev| &mut ev.lagranges, &i))
    }

    fn register_identity(&mut self, idx: usize) -> ValueSource {
        ValueSource::Identity(self.register(|ev| &mut ev.identities, &idx))
    }

    fn register_query(&mut self, query: &Query) -> ValueSource {
        ValueSource::Poly(self.register(|ev| &mut ev.queries, query))
    }

    fn register_calculation(&mut self, calculation: Calculation<ValueSource>) -> ValueSource {
        ValueSource::Calculation(self.register(|ev| &mut ev.calculations, &calculation))
    }

    fn register_expression(&mut self, expr: &Expression<F>) -> ValueSource {
        match expr {
            Expression::Constant(constant) => self.register_constant(constant),
            Expression::CommonPolynomial(poly) => match poly {
                CommonPolynomial::Lagrange(i) => self.register_lagrange(*i),
                CommonPolynomial::Identity(idx) => self.register_identity(*idx),
                CommonPolynomial::EqXY(_) => unreachable!(),
            },
            Expression::Polynomial(query) => self.register_query(query),
            Expression::Challenge(_) => unreachable!(),
            Expression::Negated(value) => {
                if let Expression::Constant(constant) = value.deref() {
                    self.register_constant(&-*constant)
                } else {
                    let value = self.register_expression(value);
                    if let ValueSource::Constant(idx) = value {
                        self.register_constant(&-self.constants[idx])
                    } else {
                        self.register_calculation(Calculation::Negate(value))
                    }
                }
            }
            Expression::Sum(lhs, rhs) => match (lhs.deref(), rhs.deref()) {
                (minuend, Expression::Negated(subtrahend))
                | (Expression::Negated(subtrahend), minuend) => {
                    let minuend = self.register_expression(minuend);
                    let subtrahend = self.register_expression(subtrahend);
                    match (minuend, subtrahend) {
                        (ValueSource::Constant(minuend), ValueSource::Constant(subtrahend)) => self
                            .register_constant(
                                &(self.constants[minuend] - &self.constants[subtrahend]),
                            ),
                        (ValueSource::Constant(0), _) => {
                            self.register_calculation(Calculation::Negate(subtrahend))
                        }
                        (_, ValueSource::Constant(0)) => minuend,
                        _ => self.register_calculation(Calculation::Sub(minuend, subtrahend)),
                    }
                }
                _ => {
                    let lhs = self.register_expression(lhs);
                    let rhs = self.register_expression(rhs);
                    match (lhs, rhs) {
                        (ValueSource::Constant(lhs), ValueSource::Constant(rhs)) => {
                            self.register_constant(&(self.constants[lhs] + &self.constants[rhs]))
                        }
                        (ValueSource::Constant(0), other) | (other, ValueSource::Constant(0)) => {
                            other
                        }
                        _ => {
                            if lhs <= rhs {
                                self.register_calculation(Calculation::Add(lhs, rhs))
                            } else {
                                self.register_calculation(Calculation::Add(rhs, lhs))
                            }
                        }
                    }
                }
            },
            Expression::Product(lhs, rhs) => {
                let lhs = self.register_expression(lhs);
                let rhs = self.register_expression(rhs);
                match (lhs, rhs) {
                    (ValueSource::Constant(0), _) | (_, ValueSource::Constant(0)) => {
                        ValueSource::Constant(0)
                    }
                    (ValueSource::Constant(1), other) | (other, ValueSource::Constant(1)) => other,
                    (ValueSource::Constant(2), other) | (other, ValueSource::Constant(2)) => {
                        self.register_calculation(Calculation::Add(other, other))
                    }
                    (lhs, rhs) => {
                        if lhs <= rhs {
                            self.register_calculation(Calculation::Mul(lhs, rhs))
                        } else {
                            self.register_calculation(Calculation::Mul(rhs, lhs))
                        }
                    }
                }
            }
            Expression::Scaled(value, scalar) => {
                if scalar == &F::ZERO {
                    ValueSource::Constant(0)
                } else if scalar == &F::ONE {
                    self.register_expression(value)
                } else {
                    let value = self.register_expression(value);
                    let scalar = self.register_constant(scalar);
                    self.register_calculation(Calculation::Mul(value, scalar))
                }
            }
            Expression::DistributePowers(_, _) => unreachable!(),
        }
    }

    fn sparse_bs(&self, _: &[&MultilinearPolynomial<F>]) -> Option<Vec<usize>> {
        self.sparse.as_ref().map(|sparse| {
            sparse
                .evaluate(
                    &|_| None,
                    &|poly| match poly {
                        CommonPolynomial::Lagrange(i) => {
                            Some(vec![self.bh[i.rem_euclid(1 << self.num_vars) as usize]])
                        }
                        CommonPolynomial::Identity(_) => unimplemented!(),
                        _ => None,
                    },
                    &|_| None,
                    &|_| None,
                    &|bs| bs,
                    &|lhs, rhs| match (lhs, rhs) {
                        (None, None) => None,
                        (Some(bs), None) | (None, Some(bs)) => Some(bs),
                        (Some(mut lhs), Some(rhs)) => {
                            lhs.extend(rhs);
                            Some(lhs)
                        }
                    },
                    &|lhs, rhs| match (lhs, rhs) {
                        (None, None) => None,
                        (Some(bs), None) | (None, Some(bs)) => Some(bs),
                        (Some(lhs), Some(rhs)) => Some(
                            BTreeSet::from_iter(lhs)
                                .intersection(&BTreeSet::from_iter(rhs))
                                .cloned()
                                .collect(),
                        ),
                    },
                    &|bs, _| bs,
                )
                .unwrap()
        })
    }

    fn data(&self) -> EvaluatorData<F> {
        let mut data = EvaluatorData {
            calculations: vec![F::ZERO; self.offsets.3 + self.calculations.len()],
        };
        data.calculations[..self.constants.len()].clone_from_slice(&self.constants);
        data
    }

    fn evaluate<'a>(
        &self,
        evals: impl Iterator<Item = &'a mut F>,
        data: &mut EvaluatorData<F>,
        polys: &[&MultilinearPolynomial<F>],
        b: usize,
    ) {
        let bh = BooleanHypercube::new(self.num_vars);
        data.calculations[self.offsets.0..]
            .iter_mut()
            .zip(self.lagranges.iter())
            .for_each(|(value, i)| *value = if &b == i { F::ONE } else { F::ZERO });
        data.calculations[self.offsets.1..]
            .iter_mut()
            .zip(self.identities.iter())
            .for_each(|(value, idx)| *value = F::from(((self.num_vars << idx) + b) as u64));
        data.calculations[self.offsets.2..]
            .iter_mut()
            .zip(self.queries.iter())
            .for_each(|(value, query)| {
                *value = polys[query.poly()][bh.rotate(b, query.rotation())]
            });
        self.indexed_calculations
            .iter()
            .zip(self.offsets.3..)
            .for_each(|(calculation, idx)| calculation.calculate(&mut data.calculations, idx));
        evals
            .zip(self.results.iter())
            .for_each(|(eval, idx)| *eval += data.calculations[*idx])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ValueSource {
    Constant(usize),
    Lagrange(usize),
    Identity(usize),
    Poly(usize),
    Calculation(usize),
}

impl ValueSource {
    fn indexed(&self, offsets: &(usize, usize, usize, usize)) -> usize {
        use ValueSource::*;
        match self {
            Constant(idx) => *idx,
            Lagrange(idx) => offsets.0 + idx,
            Identity(idx) => offsets.1 + idx,
            Poly(idx) => offsets.2 + idx,
            Calculation(idx) => offsets.3 + idx,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Calculation<T> {
    Negate(T),
    Add(T, T),
    Sub(T, T),
    Mul(T, T),
}

impl Calculation<ValueSource> {
    fn indexed(&self, offsets: &(usize, usize, usize, usize)) -> Calculation<usize> {
        use Calculation::*;
        match self {
            Negate(value) => Negate(value.indexed(offsets)),
            Add(lhs, rhs) => Add(lhs.indexed(offsets), rhs.indexed(offsets)),
            Sub(lhs, rhs) => Sub(lhs.indexed(offsets), rhs.indexed(offsets)),
            Mul(lhs, rhs) => Mul(lhs.indexed(offsets), rhs.indexed(offsets)),
        }
    }
}

impl Calculation<usize> {
    fn calculate<F: PrimeField>(&self, data: &mut [F], idx: usize) {
        use Calculation::*;
        data[idx] = match self {
            Negate(value) => -data[*value],
            Add(lhs, rhs) => data[*lhs] + &data[*rhs],
            Sub(lhs, rhs) => data[*lhs] - &data[*rhs],
            Mul(lhs, rhs) => data[*lhs] * &data[*rhs],
        };
    }
}

#[derive(Debug)]
struct EvaluatorData<F: PrimeField> {
    calculations: Vec<F>,
}

fn split_sparse<F: PrimeField>(
    expression: &Expression<F>,
    _: &[&MultilinearPolynomial<F>],
) -> (Expression<F>, Vec<Expression<F>>) {
    expression.evaluate(
        &|constant| (Expression::Constant(constant), Vec::new()),
        &|poly| {
            if matches!(poly, CommonPolynomial::Lagrange(_)) {
                (Expression::zero(), vec![Expression::CommonPolynomial(poly)])
            } else {
                (poly.into(), Vec::new())
            }
        },
        &|query| {
            // TODO: Recognize sparse selectors
            (query.into(), Vec::new())
        },
        &|_| unreachable!(),
        &|(dense, sparse)| (-dense, sparse.iter().map(|sparse| -sparse).collect()),
        &|(lhs_dense, lhs_sparse), (rhs_dense, rhs_sparse)| {
            (lhs_dense + rhs_dense, [lhs_sparse, rhs_sparse].concat())
        },
        &|(lhs_dense, lhs_sparse), (rhs_dense, rhs_sparse)| match (
            lhs_dense, lhs_sparse, rhs_dense, rhs_sparse,
        ) {
            (lhs_dense, sparse, rhs_dense, empty) | (rhs_dense, empty, lhs_dense, sparse)
                if empty.is_empty() =>
            {
                let sparse = sparse.iter().map(|sparse| sparse * &rhs_dense).collect();
                (lhs_dense * &rhs_dense, sparse)
            }
            (lhs_dense, lhs_sparse, rhs_dense, rhs_sparse) => {
                let lhs = lhs_dense + lhs_sparse.into_iter().sum::<Expression<_>>();
                let rhs = rhs_dense + rhs_sparse.into_iter().sum::<Expression<_>>();
                (lhs * rhs, Vec::new())
            }
        },
        &|(dense, sparse), scalar| {
            let sparse = sparse.iter().map(|sparse| sparse * scalar).collect();
            (dense * scalar, sparse)
        },
    )
}
