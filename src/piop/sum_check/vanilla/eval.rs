use crate::{
    piop::sum_check::vanilla::{ProverState, VanillaSumCheckProver, VanillaSumCheckRoundMessage},
    util::{
        arithmetic::{
            barycentric_interpolate, barycentric_weights, div_ceil, BooleanHypercube, PrimeField,
        },
        expression::{CommonPolynomial, Expression, Query, Rotation},
        impl_index,
        parallel::{num_threads, parallelize_iter},
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use num_integer::Integer;
use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{AddAssign, Deref},
};

#[derive(Clone, Debug)]
pub struct Evaluations<F>(Vec<F>);

impl<F: PrimeField> Evaluations<F> {
    fn points(degree: usize) -> Vec<F> {
        (0..degree as u64 + 1).map_into().collect_vec()
    }
}

impl<F: PrimeField> VanillaSumCheckRoundMessage<F> for Evaluations<F> {
    type Auxiliary = (Vec<F>, Vec<F>);

    fn write(&self, transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error> {
        for eval in self.0.iter().copied() {
            transcript.write_scalar(eval)?;
        }
        Ok(())
    }

    fn read(degree: usize, transcript: &mut impl TranscriptRead<F>) -> Result<Self, Error> {
        transcript.read_n_scalars(degree + 1).map(Self)
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
    GraphEvaluator<F, IS_ZERO_CHECK>,
);

impl<F, const IS_ZERO_CHECK: bool> VanillaSumCheckProver<F> for EvaluationsProver<F, IS_ZERO_CHECK>
where
    F: PrimeField,
{
    type RoundMessage = Evaluations<F>;

    fn new(state: &ProverState<F>) -> Self {
        Self(GraphEvaluator::new(state))
    }

    fn prove_round<'a>(&self, state: &ProverState<'a, F>) -> Evaluations<F> {
        self.evals(state)
    }
}

impl<F: PrimeField, const IS_ZERO_CHECK: bool> EvaluationsProver<F, IS_ZERO_CHECK> {
    fn evals<'a>(&self, state: &ProverState<'a, F>) -> Evaluations<F> {
        let num_evals = state.expression.degree() + 1;
        let mut evals = Evaluations(vec![F::zero(); num_evals]);

        let size = state.size();
        let num_threads = num_threads();
        if size < num_threads {
            let bs = 0..size;
            let mut data = self.0.data();
            if state.round > 0 {
                bs.for_each(|b| self.0.evaluate::<false>(&mut evals, &mut data, state, b))
            } else {
                bs.for_each(|b| self.0.evaluate::<true>(&mut evals, &mut data, state, b))
            }
        } else {
            let chunk_size = div_ceil(size, num_threads);
            let mut partials = vec![Evaluations(vec![F::zero(); num_evals]); num_threads];
            parallelize_iter(
                partials.iter_mut().zip((0..).step_by(chunk_size)),
                |(partials, start)| {
                    let bs = start..(start + chunk_size).min(size);
                    let mut data = self.0.data();
                    if state.round > 0 {
                        bs.for_each(|b| self.0.evaluate::<false>(partials, &mut data, state, b))
                    } else {
                        bs.for_each(|b| self.0.evaluate::<true>(partials, &mut data, state, b))
                    }
                },
            );
            partials.iter().for_each(|partials| evals += partials);
        }

        evals[0] = state.sum - evals[1];
        evals
    }
}

#[derive(Clone, Debug, Default)]
struct GraphEvaluator<F: PrimeField, const IS_ZERO_CHECK: bool> {
    num_vars: usize,
    constants: Vec<F>,
    challenges: Vec<F>,
    lagranges: Vec<i32>,
    identitys: Vec<usize>,
    eq_xys: Vec<usize>,
    rotations: Vec<(Rotation, usize)>,
    polys: Vec<(usize, usize)>,
    calculations: Vec<Calculation>,
}

impl<F: PrimeField, const IS_ZERO_CHECK: bool> GraphEvaluator<F, IS_ZERO_CHECK> {
    fn new(state: &ProverState<F>) -> Self {
        let mut ev = Self {
            num_vars: state.num_vars,
            constants: vec![F::from(0), F::from(1), F::from(2)],
            challenges: state.challenges.to_vec(),
            rotations: vec![(Rotation(0), state.num_vars)],
            ..Default::default()
        };
        ev.register_expression(state.expression);
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

    fn register_calculation(&mut self, calculation: Calculation) -> ValueSource {
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
            Expression::Challenge(challenge) => ValueSource::Challenge(*challenge),
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
                            if lhs < rhs {
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
                        self.register_calculation(Calculation::Double(other))
                    }
                    (lhs, rhs) => match lhs.cmp(&rhs) {
                        Ordering::Equal => self.register_calculation(Calculation::Square(lhs)),
                        Ordering::Less => self.register_calculation(Calculation::Mul(lhs, rhs)),
                        Ordering::Greater => self.register_calculation(Calculation::Mul(rhs, lhs)),
                    },
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
            Expression::DistributePowers(values, base) => {
                let values = values
                    .iter()
                    .map(|value| self.register_expression(value))
                    .collect_vec();
                let base = self.register_expression(base);
                self.register_calculation(Calculation::Horner(values, base))
            }
        }
    }

    fn data(&self) -> EvaluatorData<F> {
        EvaluatorData {
            bs: vec![(0, 0); self.rotations.len()],
            lagranges: vec![F::zero(); self.lagranges.len()],
            lagrange_steps: vec![F::zero(); self.lagranges.len()],
            identitys: vec![F::zero(); self.identitys.len()],
            identity_step: F::zero(),
            eq_xys: vec![F::zero(); self.eq_xys.len()],
            eq_xy_steps: vec![F::zero(); self.eq_xys.len()],
            polys: vec![F::zero(); self.polys.len()],
            poly_steps: vec![F::zero(); self.polys.len()],
            calculations: vec![F::zero(); self.calculations.len()],
        }
    }

    fn evaluate_polys_next<const IS_FIRST_ROUND: bool, const IS_FIRST_POINT: bool>(
        &self,
        data: &mut EvaluatorData<F>,
        state: &ProverState<F>,
        b: usize,
    ) {
        if IS_FIRST_ROUND {
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
                    data.lagranges.iter_mut(),
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
            data.identity_step = F::from((1 << state.round) as u64);
            zip_for_each!(
                (data.identitys.iter_mut(), self.identitys.iter()),
                (|(eval, idx)| {
                    *eval = state.identities[*idx]
                        + F::from((b << (state.round + 1)) as u64)
                        + &data.identity_step
                })
            );
            zip_for_each!(
                (data.eq_xys.iter_mut(), self.eq_xys.iter()),
                (|(eval, idx)| *eval = state.eq_xys[*idx][b_1])
            );
            zip_for_each!(
                (data.polys.iter_mut(), self.polys.iter()),
                (|(eval, (poly, rotation))| if IS_FIRST_ROUND {
                    *eval = state.polys[*poly][self.num_vars][data.bs[*rotation].1]
                } else {
                    *eval = state.polys[*poly][self.rotations[*rotation].1][b_1]
                })
            );
            zip_for_each!(
                (data.eq_xy_steps.iter_mut(), self.eq_xys.iter()),
                (|(step, idx)| *step = state.eq_xys[*idx][b_1] - &state.eq_xys[*idx][b_0])
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
                    (data.lagranges.iter_mut(), &data.lagrange_steps),
                    (data.eq_xys.iter_mut(), &data.eq_xy_steps),
                    (data.polys.iter_mut(), &data.poly_steps)
                ],
                (|(eval, step)| *eval += step)
            );
            data.identitys
                .iter_mut()
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

        for (idx, calculation) in self.calculations.iter().enumerate() {
            calculation.calculate(&self.constants, &self.challenges, data, idx);
        }
        if cfg!(feature = "sanity-check") && IS_ZERO_CHECK {
            assert_eq!(data.calculations.last().unwrap(), &F::zero());
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
    Challenge(usize),
    Lagrange(usize),
    Identity(usize),
    EqXY(usize),
    Poly(usize),
    Calculation(usize),
}

impl ValueSource {
    fn load<'a, F: PrimeField>(
        &self,
        constants: &'a [F],
        challenges: &'a [F],
        data: &'a EvaluatorData<F>,
    ) -> &'a F {
        match self {
            ValueSource::Constant(idx) => &constants[*idx],
            ValueSource::Challenge(idx) => &challenges[*idx],
            ValueSource::Lagrange(idx) => &data.lagranges[*idx],
            ValueSource::Identity(idx) => &data.identitys[*idx],
            ValueSource::EqXY(idx) => &data.eq_xys[*idx],
            ValueSource::Poly(idx) => &data.polys[*idx],
            ValueSource::Calculation(idx) => &data.calculations[*idx],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Calculation {
    Negate(ValueSource),
    Add(ValueSource, ValueSource),
    Sub(ValueSource, ValueSource),
    Mul(ValueSource, ValueSource),
    Double(ValueSource),
    Square(ValueSource),
    Horner(Vec<ValueSource>, ValueSource),
}

impl Calculation {
    fn calculate<F: PrimeField>(
        &self,
        constants: &[F],
        challenges: &[F],
        data: &mut EvaluatorData<F>,
        idx: usize,
    ) {
        let load = |source: &ValueSource| source.load(constants, challenges, data);
        data.calculations[idx] = match self {
            Calculation::Negate(value) => -*load(value),
            Calculation::Add(lhs, rhs) => *load(lhs) + load(rhs),
            Calculation::Sub(lhs, rhs) => *load(lhs) - load(rhs),
            Calculation::Mul(lhs, rhs) => *load(lhs) * load(rhs),
            Calculation::Double(value) => load(value).double(),
            Calculation::Square(value) => load(value).square(),
            Calculation::Horner(values, base) => {
                let base = load(base);
                values
                    .iter()
                    .fold(F::zero(), |acc, item| acc * base + load(item))
            }
        };
    }
}

#[derive(Debug)]
struct EvaluatorData<F: PrimeField> {
    bs: Vec<(usize, usize)>,
    lagranges: Vec<F>,
    lagrange_steps: Vec<F>,
    identitys: Vec<F>,
    identity_step: F,
    eq_xys: Vec<F>,
    eq_xy_steps: Vec<F>,
    polys: Vec<F>,
    poly_steps: Vec<F>,
    calculations: Vec<F>,
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
        test::tests,
        vanilla::{EvaluationsProver, VanillaSumCheck},
    };

    tests!(VanillaSumCheck<EvaluationsProver<Fr, true>>);
}
