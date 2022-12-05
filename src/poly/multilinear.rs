use crate::{
    poly::impl_index,
    util::{
        arithmetic::{int_from_fe_bits_le, BooleanHypercube, Field},
        num_threads, parallelize, parallelize_iter,
    },
};
use num_integer::Integer;
use rand::RngCore;
use std::{
    borrow::Cow,
    cmp::Ordering,
    iter::{self, Sum},
    ops::{Add, AddAssign, Range, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct MultilinearPolynomial<F> {
    evals: Vec<F>,
    num_vars: usize,
}

impl<F: Field> MultilinearPolynomial<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let num_vars = if evals.is_empty() {
            0
        } else {
            let num_vars = (usize::BITS - 1 - evals.len().leading_zeros()) as usize;
            assert_eq!(evals.len(), 1 << num_vars);
            num_vars
        };

        Self { evals, num_vars }
    }

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
                let mut chunk_size = Integer::div_ceil(&evals.len(), &num_threads());
                if chunk_size.is_odd() {
                    chunk_size += 1;
                }
                parallelize_iter(
                    next_evals
                        .chunks_mut(chunk_size)
                        .zip(evals.chunks(chunk_size / 2)),
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

    pub fn zero() -> Self {
        Self::new(Vec::new())
    }

    pub fn is_zero(&self) -> bool {
        self.num_vars == 0
    }

    pub fn coeffs(&self) -> &[F] {
        self.evals.as_slice()
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
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

    pub fn evaluate_for_rotation(&self, x: &[F], rotation: i32) -> Vec<F> {
        let rotation_eval_points = rotation_eval_points(x, rotation);
        rotation_eval_points
            .iter()
            .map(|x| self.evaluate(x))
            .collect()
    }

    pub fn fix_variables(&self, x: &[F]) -> Self {
        assert!(x.len() <= self.num_vars);

        let mut evals = Cow::Borrowed(&self.evals);
        let mut bits = Vec::new();
        for x_i in x.iter() {
            if x_i == &F::zero() || x_i == &F::one() {
                bits.push(x_i);
                continue;
            }

            let distance = bits.len() + 1;
            let step = 1 << distance;
            let skip = int_from_fe_bits_le(bits.drain(..).rev());
            let mut next_evals = vec![F::zero(); evals.len() >> distance];
            parallelize(&mut next_evals, |(next_evals, start)| {
                let start = (start << distance) + skip;
                for (next_eval, (eval_0, eval_1)) in next_evals.iter_mut().zip(
                    evals[start..]
                        .iter()
                        .step_by(step)
                        .zip(evals[start + (step >> 1)..].iter().step_by(step)),
                ) {
                    *next_eval = (*eval_1 - eval_0) * x_i + eval_0;
                }
            });
            evals = Cow::Owned(next_evals)
        }

        if !bits.is_empty() {
            let distance = bits.len();
            let step = 1 << distance;
            let skip = int_from_fe_bits_le(bits.drain(..).rev());
            let mut next_evals = vec![F::zero(); evals.len() >> distance];
            parallelize(&mut next_evals, |(next_evals, start)| {
                for (next_eval, eval) in next_evals
                    .iter_mut()
                    .zip(evals[(start << distance) + skip..].iter().step_by(step))
                {
                    *next_eval = *eval;
                }
            });
            evals = Cow::Owned(next_evals)
        }

        Self::new(evals.into_owned())
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
        self.evals[0] += rhs;
    }
}

impl<F: Field> SubAssign<F> for MultilinearPolynomial<F> {
    fn sub_assign(&mut self, rhs: F) {
        self.evals[0] -= rhs;
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

pub fn rotation_eval_points<F: Field>(x: &[F], rotation: i32) -> Vec<Vec<F>> {
    match rotation.cmp(&0) {
        Ordering::Less => unimplemented!("Negative roation is not supported yet"),
        Ordering::Equal => vec![x.to_vec()],
        Ordering::Greater => {
            let rotation = rotation as usize;
            let bh = BooleanHypercube::new(x.len());
            let mut points = vec![x.to_vec()];
            for _ in 0..rotation {
                points = points
                    .into_iter()
                    .flat_map(|point| {
                        let mut mask = bh.mask();
                        [
                            iter::once(F::zero())
                                .chain(point.iter().cloned())
                                .take(point.len())
                                .collect(),
                            iter::once(F::one())
                                .chain(point.iter().map(|x_i| {
                                    mask >>= 1;
                                    if mask.is_odd() {
                                        F::one() - x_i
                                    } else {
                                        *x_i
                                    }
                                }))
                                .take(point.len())
                                .collect(),
                        ]
                    })
                    .collect()
            }
            points
        }
    }
}

pub fn compute_rotation_eval<F: Field>(evals_for_roation: &[F], x: &[F]) -> F {
    assert!(evals_for_roation.len() < 1 << x.len());
    let evals = MultilinearPolynomial::new(evals_for_roation.to_vec());
    evals.evaluate(&x[x.len() - evals.num_vars()..])
}

#[cfg(test)]
mod test {
    use crate::{
        poly::multilinear::{compute_rotation_eval, MultilinearPolynomial},
        util::{arithmetic::BooleanHypercube, test::rand_vec, Itertools},
    };
    use halo2_curves::bn256::Fr;
    use rand::{rngs::StdRng, SeedableRng};
    use std::iter;

    // FIXME: Figure out why it only supports rotation up certain number.
    // The max rotation supported is the distance between second and first bit
    // of mask.
    #[test]
    fn test_evaluate_for_rotation() {
        let mut rng = StdRng::from_seed(Default::default());
        for num_vars in 0..16 {
            let next_map = BooleanHypercube::new(num_vars).next_map();
            let fs = iter::successors(Some(rand_vec(1 << num_vars, &mut rng)), |f| {
                Some((0..1 << num_vars).map(|idx| f[next_map[idx]]).collect_vec())
            })
            .map(MultilinearPolynomial::new)
            .take(num_vars)
            .collect_vec();
            let x = rand_vec::<Fr>(num_vars, &mut rng);

            for rotation in 0..num_vars {
                assert_eq!(
                    compute_rotation_eval(&fs[0].evaluate_for_rotation(&x, rotation as i32), &x),
                    fs[rotation].evaluate(&x)
                );
            }
        }
    }
}
