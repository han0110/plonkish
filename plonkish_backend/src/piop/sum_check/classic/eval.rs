use crate::{
    piop::sum_check::classic::{ClassicSumCheckProver, ClassicSumCheckRoundMessage, ProverState},
    util::{
        arithmetic::{
            barycentric_interpolate, barycentric_weights, div_ceil, steps, BooleanHypercube,
            PrimeField,
        },
        expression::{CommonPolynomial, Expression, Query, Rotation},
        impl_index,
        parallel::{num_threads, parallelize_iter},
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
    },
    Error,
};
use num_integer::Integer;
use std::{
    collections::BTreeSet,
    fmt::Debug,
    iter,
    ops::{AddAssign, Deref},
};

#[derive(Clone, Debug)]
pub struct Evaluations<F>(Vec<F>);

impl<F: PrimeField> Evaluations<F> {
    fn new(degree: usize) -> Self {
        Self(vec![F::zero(); degree + 1])
    }

    fn points(degree: usize) -> Vec<F> {
        steps(F::zero()).take(degree + 1).collect()
    }
}

impl<F: PrimeField> ClassicSumCheckRoundMessage<F> for Evaluations<F> {
    type Auxiliary = (Vec<F>, Vec<F>);

    fn write(&self, transcript: &mut impl FieldTranscriptWrite<F>) -> Result<(), Error> {
        transcript.write_field_elements(&self.0)
    }

    fn read(degree: usize, transcript: &mut impl FieldTranscriptRead<F>) -> Result<Self, Error> {
        transcript.read_field_elements(degree + 1).map(Self)
    }

    fn sum(&self) -> F {
        self[0] + self[1]
    }

    fn auxiliary(degree: usize) -> Self::Auxiliary {
        let points = Self::points(degree);
        (barycentric_weights(&points), points)
    }

    fn evaluate(&self, (weights, points): &Self::Auxiliary, challenge: &F) -> F {
        barycentric_interpolate(weights, points, &self.0, challenge)
    }
}

impl<'rhs, F: PrimeField> AddAssign<&'rhs Evaluations<F>> for Evaluations<F> {
    fn add_assign(&mut self, rhs: &'rhs Evaluations<F>) {
        zip_for_each!(
            (self.0.iter_mut(), rhs.0.iter()),
            (|(lhs, rhs)| *lhs += rhs)
        );
    }
}

impl_index!(Evaluations, 0);

#[derive(Clone, Debug)]
pub struct EvaluationsProver<F: PrimeField, const IS_ZERO_CHECK: bool>(
    Vec<GraphEvaluator<F, IS_ZERO_CHECK>>,
);

impl<F, const IS_ZERO_CHECK: bool> ClassicSumCheckProver<F> for EvaluationsProver<F, IS_ZERO_CHECK>
where
    F: PrimeField,
{
    type RoundMessage = Evaluations<F>;

    fn new(state: &ProverState<F>) -> Self {
        let (dense, sparse) = split_sparse(state);
        Self(
            iter::empty()
                .chain(Some((&dense, false)))
                .chain(sparse.iter().zip(iter::repeat(true)))
                .map(|(expression, is_sparse)| {
                    GraphEvaluator::new(state.num_vars, state.challenges, expression, is_sparse)
                })
                .collect(),
        )
    }

    fn prove_round(&self, state: &ProverState<F>) -> Evaluations<F> {
        if state.round > 0 {
            self.evals::<false>(state)
        } else {
            self.evals::<true>(state)
        }
    }
}

impl<F: PrimeField, const IS_ZERO_CHECK: bool> EvaluationsProver<F, IS_ZERO_CHECK> {
    fn evals<const IS_FIRST_ROUND: bool>(&self, state: &ProverState<F>) -> Evaluations<F> {
        let mut evals = Evaluations::new(state.degree);

        let size = state.size();
        let chunk_size = div_ceil(size, num_threads());
        let mut partials = vec![Evaluations::new(state.degree); div_ceil(size, chunk_size)];
        for ev in self.0.iter() {
            if let Some(sparse_bs) = ev.sparse_bs(state) {
                let mut data = ev.data(state, 0);
                sparse_bs.into_iter().for_each(|b| {
                    ev.evaluate::<IS_FIRST_ROUND>(&mut partials[0], &mut data, state, b)
                })
            } else {
                parallelize_iter(
                    partials.iter_mut().zip((0..).step_by(chunk_size)),
                    |(partials, start)| {
                        let bs = start..(start + chunk_size).min(size);
                        let mut data = ev.data(state, start);
                        bs.for_each(|b| {
                            ev.evaluate::<IS_FIRST_ROUND>(partials, &mut data, state, b)
                        })
                    },
                );
            }
        }
        partials.iter().for_each(|partials| evals += partials);

        if cfg!(feature = "sanity-check") && IS_ZERO_CHECK && IS_FIRST_ROUND {
            assert_eq!(evals[1], F::zero());
        }

        evals[0] = state.sum - evals[1];
        evals
    }
}

#[derive(Clone, Debug, Default)]
struct GraphEvaluator<F: PrimeField, const IS_ZERO_CHECK: bool> {
    num_vars: usize,
    constants: Vec<F>,
    lagranges: Vec<i32>,
    identitys: Vec<usize>,
    eq_xys: Vec<usize>,
    rotations: Vec<(Rotation, usize)>,
    polys: Vec<(usize, usize)>,
    calculations: Vec<Calculation<ValueSource>>,
    indexed_calculations: Vec<Calculation<usize>>,
    offsets: (usize, usize, usize, usize, usize),
    sparse: Option<Expression<F>>,
}

impl<F: PrimeField, const IS_ZERO_CHECK: bool> GraphEvaluator<F, IS_ZERO_CHECK> {
    fn new(num_vars: usize, challenges: &[F], expression: &Expression<F>, is_sparse: bool) -> Self {
        let mut ev = Self {
            num_vars,
            constants: vec![F::zero(), F::one(), F::one().double()],
            rotations: vec![(Rotation(0), num_vars)],
            ..Default::default()
        };

        let (scalar, expression) = extract_scalar(expression, challenges);
        let expression = expression
            .map(|expression| expression * scalar)
            .unwrap_or_else(|| Expression::Constant(scalar));
        ev.register_expression(&expression);
        ev.offsets.0 = ev.constants.len();
        ev.offsets.1 = ev.offsets.0 + ev.lagranges.len();
        ev.offsets.2 = ev.offsets.1 + ev.identitys.len();
        ev.offsets.3 = ev.offsets.2 + ev.eq_xys.len();
        ev.offsets.4 = ev.offsets.3 + ev.polys.len();
        ev.indexed_calculations = ev
            .calculations
            .iter()
            .map(|calculation| calculation.indexed(&ev.offsets))
            .collect();

        if is_sparse {
            ev.sparse = Some(expression)
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
        ValueSource::Lagrange(self.register(|ev| &mut ev.lagranges, &i))
    }

    fn register_identity(&mut self, idx: usize) -> ValueSource {
        ValueSource::Identity(self.register(|ev| &mut ev.identitys, &idx))
    }

    fn register_eq_xy(&mut self, idx: usize) -> ValueSource {
        ValueSource::EqXY(self.register(|ev| &mut ev.eq_xys, &idx))
    }

    fn register_rotation(&mut self, rotation: Rotation) -> usize {
        let rotated_poly = (rotation.0 + self.num_vars as i32) as usize;
        self.register(|ev| &mut ev.rotations, &(rotation, rotated_poly))
    }

    fn register_poly_eval(&mut self, query: &Query) -> ValueSource {
        let rotation = self.register_rotation(query.rotation());
        ValueSource::Poly(self.register(|ev| &mut ev.polys, &(query.poly(), rotation)))
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
                CommonPolynomial::EqXY(idx) => self.register_eq_xy(*idx),
            },
            Expression::Polynomial(query) => self.register_poly_eval(query),
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
                if scalar == &F::zero() {
                    ValueSource::Constant(0)
                } else if scalar == &F::one() {
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

    fn sparse_bs(&self, state: &ProverState<F>) -> Option<Vec<usize>> {
        self.sparse.as_ref().map(|sparse| {
            sparse
                .evaluate(
                    &|_| None,
                    &|poly| match poly {
                        CommonPolynomial::Lagrange(i) => Some(vec![state.lagranges[&i].0 >> 1]),
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

    fn data(&self, state: &ProverState<F>, start: usize) -> EvaluatorData<F> {
        let mut data = EvaluatorData {
            bs: vec![(0, 0); self.rotations.len()],
            lagrange_steps: vec![F::zero(); self.lagranges.len()],
            identity_step_first: F::from(1 << (state.round + 1))
                - F::from(((state.degree - 1) << state.round) as u64),
            identity_step: F::from(1 << state.round),
            eq_xy_steps: vec![F::zero(); self.eq_xys.len()],
            poly_steps: vec![F::zero(); self.polys.len()],
            calculations: vec![F::zero(); self.offsets.4 + self.calculations.len()],
        };
        data.calculations[..self.constants.len()].clone_from_slice(&self.constants);
        data.calculations[self.offsets.1..]
            .iter_mut()
            .zip(self.identitys.iter())
            .for_each(|(eval, idx)| {
                *eval = state.identities[*idx]
                    + F::from((1 << state.round) + (start << (state.round + 1)) as u64)
                    - data.identity_step_first;
            });
        data
    }

    fn evaluate_polys_next<const IS_FIRST_ROUND: bool, const IS_FIRST_POINT: bool>(
        &self,
        data: &mut EvaluatorData<F>,
        state: &ProverState<F>,
        b: usize,
    ) {
        macro_rules! calculation {
            (lagranges) => {
                data.calculations[self.offsets.0..]
            };
            (identities) => {
                data.calculations[self.offsets.1..]
            };
            (eq_xys) => {
                data.calculations[self.offsets.2..]
            };
            (polys) => {
                data.calculations[self.offsets.3..]
            };
        }

        if IS_FIRST_ROUND && IS_FIRST_POINT {
            let bh = BooleanHypercube::new(self.num_vars);
            zip_for_each!(
                (data.bs.iter_mut(), self.rotations.iter()),
                (|(bs, (rotation, _))| {
                    let [b_0, b_1] = [b << 1, (b << 1) + 1].map(|b| bh.rotate(b, *rotation));
                    *bs = (b_0, b_1);
                })
            );
        }

        if IS_FIRST_POINT {
            let (b_0, b_1) = if IS_FIRST_ROUND {
                data.bs[0]
            } else {
                (b << 1, (b << 1) + 1)
            };
            zip_for_each!(
                (
                    calculation!(lagranges).iter_mut(),
                    data.lagrange_steps.iter_mut(),
                    self.lagranges.iter()
                ),
                (|((eval, step), i)| {
                    let lagrange = &state.lagranges[i];
                    if b == lagrange.0 >> 1 {
                        if lagrange.0.is_even() {
                            *step = -lagrange.1;
                        } else {
                            *eval = lagrange.1;
                            *step = lagrange.1;
                        }
                    } else {
                        *eval = F::zero();
                        *step = F::zero();
                    }
                })
            );
            calculation!(identities)
                .iter_mut()
                .for_each(|eval| *eval += data.identity_step_first);
            zip_for_each!(
                (calculation!(eq_xys).iter_mut(), self.eq_xys.iter()),
                (|(eval, idx)| *eval = state.eq_xys[*idx][b_1])
            );
            zip_for_each!(
                (data.eq_xy_steps.iter_mut(), self.eq_xys.iter()),
                (|(step, idx)| *step = state.eq_xys[*idx][b_1] - &state.eq_xys[*idx][b_0])
            );
            zip_for_each!(
                (calculation!(polys).iter_mut(), self.polys.iter()),
                (|(eval, (poly, rotation))| if IS_FIRST_ROUND {
                    *eval = state.polys[*poly][self.num_vars][data.bs[*rotation].1]
                } else {
                    *eval = state.polys[*poly][self.rotations[*rotation].1][b_1]
                })
            );
            zip_for_each!(
                (data.poly_steps.iter_mut(), self.polys.iter()),
                (|(step, (poly, rotation))| if IS_FIRST_ROUND {
                    let poly = &state.polys[*poly][self.num_vars];
                    let (b_0, b_1) = data.bs[*rotation];
                    *step = poly[b_1] - poly[b_0]
                } else {
                    let poly = &state.polys[*poly][self.rotations[*rotation].1];
                    *step = poly[b_1] - poly[b_0]
                })
            );
        } else {
            zip_for_each!(
                [
                    (calculation!(lagranges).iter_mut(), &data.lagrange_steps),
                    (calculation!(eq_xys).iter_mut(), &data.eq_xy_steps),
                    (calculation!(polys).iter_mut(), &data.poly_steps)
                ],
                (|(eval, step)| *eval += step)
            );
            calculation!(identities)
                .iter_mut()
                .take(self.identitys.len())
                .for_each(|eval| *eval += &data.identity_step);
        }
    }

    fn evaluate_next<const IS_FIRST_ROUND: bool, const IS_FIRST_POINT: bool>(
        &self,
        eval: &mut F,
        state: &ProverState<F>,
        data: &mut EvaluatorData<F>,
        b: usize,
    ) {
        self.evaluate_polys_next::<IS_FIRST_ROUND, IS_FIRST_POINT>(data, state, b);

        if !cfg!(feature = "sanity-check") && IS_ZERO_CHECK && IS_FIRST_ROUND && IS_FIRST_POINT {
            return;
        }

        for (calculation, idx) in self.indexed_calculations.iter().zip(self.offsets.4..) {
            calculation.calculate(&mut data.calculations, idx);
        }
        *eval += data.calculations.last().unwrap();
    }

    fn evaluate<const IS_FIRST_ROUND: bool>(
        &self,
        evals: &mut Evaluations<F>,
        data: &mut EvaluatorData<F>,
        state: &ProverState<F>,
        b: usize,
    ) {
        assert!(evals.0.len() > 2);

        self.evaluate_next::<IS_FIRST_ROUND, true>(&mut evals[1], state, data, b);
        for eval in evals[2..].iter_mut() {
            self.evaluate_next::<IS_FIRST_ROUND, false>(eval, state, data, b);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ValueSource {
    Constant(usize),
    Lagrange(usize),
    Identity(usize),
    EqXY(usize),
    Poly(usize),
    Calculation(usize),
}

impl ValueSource {
    fn indexed(&self, offsets: &(usize, usize, usize, usize, usize)) -> usize {
        match self {
            ValueSource::Constant(idx) => *idx,
            ValueSource::Lagrange(idx) => offsets.0 + idx,
            ValueSource::Identity(idx) => offsets.1 + idx,
            ValueSource::EqXY(idx) => offsets.2 + idx,
            ValueSource::Poly(idx) => offsets.3 + idx,
            ValueSource::Calculation(idx) => offsets.4 + idx,
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
    fn indexed(&self, offsets: &(usize, usize, usize, usize, usize)) -> Calculation<usize> {
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
        data[idx] = match self {
            Calculation::Negate(value) => -data[*value],
            Calculation::Add(lhs, rhs) => data[*lhs] + &data[*rhs],
            Calculation::Sub(lhs, rhs) => data[*lhs] - &data[*rhs],
            Calculation::Mul(lhs, rhs) => data[*lhs] * &data[*rhs],
        };
    }
}

#[derive(Debug)]
struct EvaluatorData<F: PrimeField> {
    bs: Vec<(usize, usize)>,
    lagrange_steps: Vec<F>,
    identity_step_first: F,
    identity_step: F,
    eq_xy_steps: Vec<F>,
    poly_steps: Vec<F>,
    calculations: Vec<F>,
}

fn split_sparse<F: PrimeField>(state: &ProverState<F>) -> (Expression<F>, Vec<Expression<F>>) {
    state.expression.evaluate(
        &|constant| (Expression::Constant(constant), Vec::new()),
        &|poly| match poly {
            CommonPolynomial::Lagrange(_) => (
                Expression::Constant(F::zero()),
                vec![Expression::<F>::CommonPolynomial(poly)],
            ),
            _ => (Expression::CommonPolynomial(poly), Vec::new()),
        },
        &|query| {
            // TODO: Recognize sparse selectors
            (Expression::Polynomial(query), Vec::new())
        },
        &|challenge| (Expression::Challenge(challenge), Vec::new()),
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
                (
                    lhs_dense * &rhs_dense,
                    sparse.iter().map(|sparse| sparse * &rhs_dense).collect(),
                )
            }
            (lhs_dense, lhs_sparse, rhs_dense, rhs_sparse) => (
                iter::once(lhs_dense)
                    .chain(lhs_sparse)
                    .sum::<Expression<_>>()
                    * iter::once(rhs_dense)
                        .chain(rhs_sparse)
                        .sum::<Expression<_>>(),
                Vec::new(),
            ),
        },
        &|(dense, sparse), scalar| {
            (
                dense * scalar,
                sparse.iter().map(|sparse| sparse * scalar).collect(),
            )
        },
    )
}

fn extract_scalar<F: PrimeField>(
    expression: &Expression<F>,
    challenges: &[F],
) -> (F, Option<Expression<F>>) {
    expression.evaluate(
        &|constant| (constant, None),
        &|poly| (F::one(), Some(Expression::CommonPolynomial(poly))),
        &|query| (F::one(), Some(Expression::Polynomial(query))),
        &|challenge| (challenges[challenge], None),
        &|(scalar, expression)| match expression {
            Some(expression) if scalar == F::one() => (F::one(), Some(-expression)),
            _ => (-scalar, expression),
        },
        &|lhs, rhs| match (lhs, rhs) {
            ((lhs, None), (rhs, None)) => (lhs + rhs, None),
            ((scalar, _), other) | (other, (scalar, _)) if scalar == F::zero() => other,
            (lhs, rhs) => {
                let [lhs, rhs] = [lhs, rhs].map(|(scalar, expression)| {
                    if scalar == F::one() {
                        expression
                    } else {
                        expression
                            .map(|expression| expression * scalar)
                            .or_else(|| Some(Expression::Constant(scalar)))
                    }
                });
                (
                    F::one(),
                    match (lhs, rhs) {
                        (Some(lhs), Some(rhs)) => Some(lhs + rhs),
                        (Some(e), None) | (None, Some(e)) => Some(e),
                        (None, None) => None,
                    },
                )
            }
        },
        &|lhs, rhs| match (lhs, rhs) {
            ((lhs, _), (rhs, _)) if lhs == F::zero() || rhs == F::zero() => (F::zero(), None),
            ((lhs, None), (rhs, None)) => (lhs * rhs, None),
            ((lhs, None), (rhs, Some(expression))) | ((rhs, Some(expression)), (lhs, None)) => {
                (lhs * rhs, Some(expression))
            }
            ((lhs_scalar, Some(lhs_expression)), (rhs_scalar, Some(rhs_expression))) => (
                lhs_scalar * rhs_scalar,
                Some(lhs_expression * rhs_expression),
            ),
        },
        &|(lhs, expression), rhs| (lhs * rhs, expression),
    )
}

macro_rules! zip_for_each {
    ([$(($lhs:expr, $rhs:expr)),*], $fn:tt) => {
        $(
            #[allow(unused_parens)]
            $lhs.zip($rhs).for_each($fn);
        )*
    };
    (($lhs:expr, $rhs:expr), $fn:tt) => {
        #[allow(unused_parens)]
        $lhs.zip($rhs).for_each($fn);
    };
    (($lhs:expr, $mhs:expr, $rhs:expr), $fn:tt) => {
        #[allow(unused_parens)]
        $lhs.zip($mhs).zip($rhs).for_each($fn);
    };
}

use zip_for_each;

#[cfg(test)]
mod test {
    use crate::piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        test::tests,
    };

    tests!(ClassicSumCheck<EvaluationsProver<Fr, true>>);
}
