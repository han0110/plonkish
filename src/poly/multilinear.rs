use crate::{
    poly::impl_index,
    util::{
        arithmetic::{div_ceil, ilog2, usize_from_bits_be, BooleanHypercube, Field},
        expression::Rotation,
        num_threads, parallelize, parallelize_iter, BitIndex, Itertools,
    },
};
use num_integer::Integer;
use rand::RngCore;
use std::{
    borrow::Cow,
    iter::{self, Sum},
    ops::{Add, AddAssign, Range, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct MultilinearPolynomial<F> {
    evals: Vec<F>,
    num_vars: usize,
}

impl<F> Default for MultilinearPolynomial<F> {
    fn default() -> Self {
        MultilinearPolynomial::zero()
    }
}

impl<F> MultilinearPolynomial<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let num_vars = if evals.is_empty() {
            0
        } else {
            let num_vars = ilog2(evals.len());
            assert_eq!(evals.len(), 1 << num_vars);
            num_vars
        };

        Self { evals, num_vars }
    }

    pub const fn zero() -> Self {
        Self {
            evals: Vec::new(),
            num_vars: 0,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num_vars == 0
    }

    pub fn evals(&self) -> &[F] {
        self.evals.as_slice()
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.evals.iter()
    }
}

impl<F: Field> MultilinearPolynomial<F> {
    pub fn eq_xy(y: &[F]) -> Self {
        assert!(!y.is_empty());

        let expand_serial = |next_evals: &mut [F], evals: &[F], y_i: &F| {
            for (next_evals, eval) in next_evals.chunks_mut(2).zip(evals.iter()) {
                next_evals[1] = *eval * y_i;
                next_evals[0] = *eval - &next_evals[1];
            }
        };

        let mut evals = vec![F::one()];
        for y_i in y.iter().rev() {
            let mut next_evals = vec![F::zero(); 2 * evals.len()];
            if evals.len() < 32 {
                expand_serial(&mut next_evals, &evals, y_i);
            } else {
                let mut chunk_size = div_ceil(evals.len(), num_threads());
                if chunk_size.is_odd() {
                    chunk_size += 1;
                }
                parallelize_iter(
                    next_evals
                        .chunks_mut(chunk_size)
                        .zip(evals.chunks(chunk_size >> 1)),
                    |(next_evals, evals)| expand_serial(next_evals, evals, y_i),
                );
            }
            evals = next_evals;
        }

        Self {
            evals,
            num_vars: y.len(),
        }
    }

    pub fn rand(num_vars: usize, mut rng: impl RngCore) -> Self {
        Self::new(
            iter::repeat_with(|| F::random(&mut rng))
                .take(1 << num_vars)
                .collect(),
        )
    }

    pub fn evaluate(&self, x: &[F]) -> F {
        assert_eq!(x.len(), self.num_vars);

        self.fix_variables(x)[0]
    }

    pub fn fix_variables(&self, x: &[F]) -> Self {
        assert!(x.len() <= self.num_vars);

        let mut evals = Cow::Borrowed(self.evals());
        let mut bits = Vec::new();
        for x_i in x.iter() {
            if x_i == &F::zero() || x_i == &F::one() {
                bits.push(x_i == &F::one());
                continue;
            }

            let distance = bits.len() + 1;
            let skip = usize_from_bits_be(bits.drain(..).rev());
            evals = merge!(&evals, x_i, distance, skip).into()
        }

        if !bits.is_empty() {
            let distance = bits.len();
            let step = 1 << distance;
            let skip = usize_from_bits_be(bits.drain(..).rev());
            let mut next_evals = vec![F::zero(); evals.len() >> distance];
            parallelize(&mut next_evals, |(next_evals, start)| {
                for (next_eval, eval) in next_evals
                    .iter_mut()
                    .zip(evals[(start << distance) + skip..].iter().step_by(step))
                {
                    *next_eval = *eval;
                }
            });
            evals = next_evals.into()
        }

        Self::new(evals.into_owned())
    }

    pub fn evaluate_for_rotation(&self, x: &[F], rotation: Rotation) -> Vec<F> {
        assert_eq!(x.len(), self.num_vars);
        if rotation == Rotation::cur() {
            return vec![self.evaluate(x)];
        }

        let distance = rotation.distance();
        let num_x = self.num_vars - distance;
        if rotation < Rotation::cur() {
            let x = &x[distance..];
            let flipped_x = x.iter().map(flip).collect_vec();
            let pattern = rotation_eval_point_pattern::<false>(self.num_vars, distance);
            let offset_mask = (1 << self.num_vars) - (1 << num_x);
            let mut evals = vec![F::zero(); pattern.len()];
            parallelize(&mut evals, |(evals, start)| {
                for (eval, pat) in evals.iter_mut().zip(pattern[start..].iter()) {
                    let offset = pat & offset_mask;
                    let mut evals = Cow::Borrowed(&self[offset..offset + (1 << num_x)]);
                    for (idx, (x_i, flipped_x_i)) in x.iter().zip(flipped_x.iter()).enumerate() {
                        let x_i = if pat.nth_bit(idx) { flipped_x_i } else { x_i };
                        evals = merge!(&evals, x_i).into();
                    }
                    *eval = evals[0];
                }
            });
            evals
        } else {
            let x = &x[..num_x];
            let flipped_x = x.iter().map(flip).collect_vec();
            let pattern = rotation_eval_point_pattern::<true>(self.num_vars, distance);
            let skip_mask = (1 << distance) - 1;
            let mut evals = vec![F::zero(); pattern.len()];
            parallelize(&mut evals, |(evals, start)| {
                for (eval, pat) in evals.iter_mut().zip(pattern[start..].iter()) {
                    let mut evals = {
                        let skip = pat & skip_mask;
                        let x_0 = if pat.nth_bit(distance) {
                            &flipped_x[0]
                        } else {
                            &x[0]
                        };
                        merge!(self.evals(), x_0, distance + 1, skip)
                    };
                    for ((x_i, flipped_x_i), idx) in
                        x.iter().zip(flipped_x.iter()).zip(distance..).skip(1)
                    {
                        let x_i = if pat.nth_bit(idx) { flipped_x_i } else { x_i };
                        evals = merge!(&evals, x_i);
                    }
                    *eval = evals[0];
                }
            });
            evals
        }
    }
}

impl<'lhs, 'rhs, F: Field> Add<&'rhs MultilinearPolynomial<F>> for &'lhs MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn add(self, rhs: &'rhs MultilinearPolynomial<F>) -> MultilinearPolynomial<F> {
        let mut output = self.clone();
        output += rhs;
        output
    }
}

impl<'rhs, F: Field> AddAssign<&'rhs MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, rhs: &'rhs MultilinearPolynomial<F>) {
        assert_eq!(self.num_vars, rhs.num_vars);

        parallelize(&mut self.evals, |(lhs, start)| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                *lhs += rhs;
            }
        });
    }
}

impl<'lhs, 'rhs, F: Field> Sub<&'rhs MultilinearPolynomial<F>> for &'lhs MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn sub(self, rhs: &'rhs MultilinearPolynomial<F>) -> MultilinearPolynomial<F> {
        let mut output = self.clone();
        output -= rhs;
        output
    }
}

impl<'rhs, F: Field> SubAssign<&'rhs MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sub_assign(&mut self, rhs: &'rhs MultilinearPolynomial<F>) {
        assert_eq!(self.num_vars, rhs.num_vars);

        parallelize(&mut self.evals, |(lhs, start)| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs[start..].iter()) {
                *lhs -= rhs;
            }
        });
    }
}

impl<F: Field> AddAssign<F> for MultilinearPolynomial<F> {
    fn add_assign(&mut self, rhs: F) {
        self.evals[0] += &rhs;
    }
}

impl<F: Field> SubAssign<F> for MultilinearPolynomial<F> {
    fn sub_assign(&mut self, rhs: F) {
        self.evals[0] -= &rhs;
    }
}

impl<'lhs, F: Field> Mul<F> for &'lhs MultilinearPolynomial<F> {
    type Output = MultilinearPolynomial<F>;

    fn mul(self, rhs: F) -> MultilinearPolynomial<F> {
        let mut output = self.clone();
        output *= rhs;
        output
    }
}

impl<F: Field> MulAssign<F> for MultilinearPolynomial<F> {
    fn mul_assign(&mut self, rhs: F) {
        if rhs == F::zero() {
            self.evals = vec![F::zero(); 1 << self.num_vars]
        } else if rhs != F::one() {
            parallelize(&mut self.evals, |(lhs, _)| {
                for lhs in lhs.iter_mut() {
                    *lhs *= &rhs;
                }
            });
        }
    }
}

impl<F: Field> Sum<MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sum<I: Iterator<Item = MultilinearPolynomial<F>>>(mut iter: I) -> MultilinearPolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => &lhs + &rhs,
            (Some(lhs), None) => return lhs,
            _ => unreachable!(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += &poly;
            acc
        })
    }
}

impl<'a, F: Field> Sum<&'a MultilinearPolynomial<F>> for MultilinearPolynomial<F> {
    fn sum<I: Iterator<Item = &'a MultilinearPolynomial<F>>>(
        mut iter: I,
    ) -> MultilinearPolynomial<F> {
        let init = match (iter.next(), iter.next()) {
            (Some(lhs), Some(rhs)) => lhs + rhs,
            (Some(lhs), None) => return lhs.clone(),
            _ => unreachable!(),
        };
        iter.fold(init, |mut acc, poly| {
            acc += poly;
            acc
        })
    }
}

impl_index!(
    MultilinearPolynomial, evals,
    [
        usize => F,
        Range<usize> => [F],
        RangeFrom<usize> => [F],
        RangeFull => [F],
        RangeInclusive<usize> => [F],
        RangeTo<usize> => [F],
        RangeToInclusive<usize> => [F],
    ]
);

pub fn rotation_eval<F: Field>(x: &[F], rotation: Rotation, evals_for_rotation: &[F]) -> F {
    if rotation == Rotation::cur() {
        assert!(evals_for_rotation.len() == 1);
        return evals_for_rotation[0];
    }

    let num_vars = x.len();
    let distance = rotation.distance();
    assert!(evals_for_rotation.len() == 1 << distance);
    assert!(distance <= num_vars);

    let (pattern, nths, x) = if rotation < Rotation::cur() {
        (
            rotation_eval_coeff_pattern::<false>(num_vars, distance),
            (1..=distance).rev().collect_vec(),
            x[0..distance].iter().rev().collect_vec(),
        )
    } else {
        (
            rotation_eval_coeff_pattern::<true>(num_vars, distance),
            (num_vars - 1..).take(distance).collect(),
            x[num_vars - distance..].iter().collect_vec(),
        )
    };
    x.into_iter().zip(nths).enumerate().fold(
        Cow::Borrowed(evals_for_rotation),
        |evals, (idx, (x_i, nth))| {
            pattern
                .iter()
                .step_by(1 << idx)
                .map(|pat| pat.nth_bit(nth))
                .zip(zip_self!(evals.iter()))
                .map(merge_with_pattern_fn(x_i))
                .collect_vec()
                .into()
        },
    )[0]
}

pub fn rotation_eval_points<F: Field>(x: &[F], rotation: Rotation) -> Vec<Vec<F>> {
    if rotation == Rotation::cur() {
        return vec![x.to_vec()];
    }

    let distance = rotation.distance();
    let num_x = x.len() - distance;
    if rotation < Rotation::cur() {
        let pattern = rotation_eval_point_pattern::<false>(x.len(), distance);
        let x = &x[distance..];
        let flipped_x = x.iter().map(flip).collect_vec();
        pattern
            .iter()
            .map(|pat| {
                iter::empty()
                    .chain((0..num_x).map(|idx| {
                        if pat.nth_bit(idx) {
                            flipped_x[idx]
                        } else {
                            x[idx]
                        }
                    }))
                    .chain((0..distance).map(|idx| bit_to_field(pat.nth_bit(idx + num_x))))
                    .collect_vec()
            })
            .collect()
    } else {
        let pattern = rotation_eval_point_pattern::<true>(x.len(), distance);
        let x = &x[..num_x];
        let flipped_x = x.iter().map(flip).collect_vec();
        pattern
            .iter()
            .map(|pat| {
                iter::empty()
                    .chain((0..distance).map(|idx| bit_to_field(pat.nth_bit(idx))))
                    .chain((0..num_x).map(|idx| {
                        if pat.nth_bit(idx + distance) {
                            flipped_x[idx]
                        } else {
                            x[idx]
                        }
                    }))
                    .collect_vec()
            })
            .collect()
    }
}

fn rotation_eval_point_pattern<const NEXT: bool>(num_vars: usize, distance: usize) -> Vec<usize> {
    let bh = BooleanHypercube::new(num_vars);
    let remainder = if NEXT { bh.primitive() } else { bh.x_inv() };
    let mut pattern = vec![0; 1 << distance];
    for depth in 0..distance {
        for (e, o) in zip_self!(0..pattern.len(), 1 << (distance - depth)) {
            let rotated = if NEXT {
                pattern[e] << 1
            } else {
                pattern[e] >> 1
            };
            pattern[o] = rotated ^ remainder;
            pattern[e] = rotated;
        }
    }
    pattern
}

fn rotation_eval_coeff_pattern<const NEXT: bool>(num_vars: usize, distance: usize) -> Vec<usize> {
    let bh = BooleanHypercube::new(num_vars);
    let remainder = if NEXT {
        bh.primitive() - (1 << num_vars)
    } else {
        bh.x_inv() << distance
    };
    let mut pattern = vec![0; 1 << (distance - 1)];
    for depth in 0..distance - 1 {
        for (e, o) in zip_self!(0..pattern.len(), 1 << (distance - depth - 1)) {
            let rotated = if NEXT {
                pattern[e] << 1
            } else {
                pattern[e] >> 1
            };
            pattern[o] = rotated ^ remainder;
            pattern[e] = rotated;
        }
    }
    pattern
}

fn flip<F: Field>(x: &F) -> F {
    F::one() - x
}

fn bit_to_field<F: Field>(bit: bool) -> F {
    if bit {
        F::one()
    } else {
        F::zero()
    }
}

fn merge_fn<F: Field>(x_i: &F) -> impl Fn((&F, &F)) -> F + '_ {
    move |(eval_0, eval_1)| (*eval_1 - eval_0) * x_i + eval_0
}

fn merge_with_pattern_fn<F: Field>(x_i: &F) -> impl Fn((bool, (&F, &F))) -> F + '_ {
    move |(bit, (eval_0, eval_1))| {
        if bit {
            (*eval_0 - eval_1) * x_i + eval_1
        } else {
            (*eval_1 - eval_0) * x_i + eval_0
        }
    }
}

macro_rules! zip_self {
    (@ $iter:expr, $step:expr, $skip:expr) => {
        $iter.skip($skip).step_by($step).zip($iter.skip($skip + ($step >> 1)).step_by($step))
    };
    ($iter:expr) => {
        zip_self!(@ $iter, 2, 0)
    };
    ($iter:expr, $step:expr) => {
        zip_self!(@ $iter, $step, 0)
    };
    ($iter:expr, $step:expr, $skip:expr) => {
        zip_self!(@ $iter, $step, $skip)
    };
}

macro_rules! merge {
    (@ $evals:expr, $x_i:expr, $distance:expr, $skip:expr) => {{
        use $crate::{
            poly::multilinear::merge_fn,
            util::{arithmetic::div_ceil, num_threads, parallelize_iter},
        };

        let step = 1 << $distance;
        let merge_serial = |output: &mut [F], start: usize| {
            let start = (start << $distance) + $skip;
            for (output, merged) in output
                .iter_mut()
                .zip(zip_self!($evals.iter(), step, start).map(merge_fn($x_i)))
            {
                *output = merged;
            }
        };

        let mut output = vec![F::zero(); $evals.len() >> $distance];
        if output.len() < 32 {
            merge_serial(&mut output, 0);
        } else {
            let chunk_size = div_ceil(output.len(), num_threads());
            parallelize_iter(
                output.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)),
                |(output, start)| merge_serial(output, start),
            );
        }
        output
    }};
    ($evals:expr, $x_i:expr, $distance:expr, $skip:expr) => {
        merge!(@ $evals, $x_i, $distance, $skip)
    };
    ($evals:expr, $x_i:expr) => {
        merge!(@ $evals, $x_i, 1, 0)
    }
}

pub(crate) use {merge, zip_self};

#[cfg(test)]
mod test {
    use crate::{
        poly::multilinear::{merge, rotation_eval, MultilinearPolynomial},
        util::{
            arithmetic::{BooleanHypercube, Field},
            expression::Rotation,
            test::rand_vec,
            Itertools,
        },
    };
    use halo2_curves::bn256::Fr;
    use rand::{rngs::OsRng, RngCore};
    use std::{borrow::Cow, iter};

    fn fix_variables_simple<F: Field>(evals: &[F], x: &[F]) -> Vec<F> {
        x.iter()
            .fold(Cow::Borrowed(evals), |evals, x_i| {
                merge!(&evals, x_i).into()
            })
            .into_owned()
    }

    #[test]
    fn test_fix_variables() {
        let rand_x_i = || match OsRng.next_u32() % 3 {
            0 => Fr::zero(),
            1 => Fr::one(),
            2 => Fr::random(OsRng),
            _ => unreachable!(),
        };
        for num_vars in 0..16 {
            let poly = MultilinearPolynomial::rand(num_vars, OsRng);
            for x in (0..num_vars).map(|n| iter::repeat_with(rand_x_i).take(n).collect_vec()) {
                assert_eq!(
                    poly.fix_variables(&x).evals(),
                    fix_variables_simple(poly.evals(), &x)
                );
            }
        }
    }

    #[test]
    fn test_evaluate_for_rotation() {
        let mut rng = OsRng;
        for num_vars in 0..16 {
            let bh = BooleanHypercube::new(num_vars);
            let rotate = |f: &Vec<Fr>| {
                (0..1 << num_vars)
                    .map(|idx| f[bh.rotate(idx, Rotation::next())])
                    .collect_vec()
            };
            let f = rand_vec(1 << num_vars, &mut rng);
            let fs = iter::successors(Some(f), |f| Some(rotate(f)))
                .map(MultilinearPolynomial::new)
                .take(num_vars)
                .collect_vec();
            let x = rand_vec::<Fr>(num_vars, &mut rng);

            for rotation in -(num_vars as i32) + 1..num_vars as i32 {
                let rotation = Rotation(rotation);
                let (f, f_rotated) = if rotation < Rotation::cur() {
                    (fs.last().unwrap(), &fs[fs.len() - rotation.distance() - 1])
                } else {
                    (fs.first().unwrap(), &fs[rotation.distance()])
                };
                assert_eq!(
                    rotation_eval(&x, rotation, &f.evaluate_for_rotation(&x, rotation)),
                    f_rotated.evaluate(&x),
                );
            }
        }
    }
}
