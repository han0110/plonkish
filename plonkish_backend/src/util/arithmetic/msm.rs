use crate::{
    pcs::Additive,
    util::{
        arithmetic::{div_ceil, field_size, CurveAffine, Field, Group, PrimeField},
        chain, izip_eq,
        parallel::{num_threads, parallelize, parallelize_iter},
        start_timer, Itertools,
    },
};
use std::{
    iter::Sum,
    mem::size_of,
    ops::{Add, Mul, Neg, Sub},
};

pub fn window_size(num_scalars: usize) -> usize {
    if num_scalars < 32 {
        3
    } else {
        (num_scalars as f64).ln().floor() as usize
    }
}

pub fn window_table<C: CurveAffine>(window_size: usize, generator: C) -> Vec<Vec<C>> {
    let scalar_size = field_size::<C::Scalar>();
    let num_windows = div_ceil(scalar_size, window_size);
    let mut table = vec![vec![C::identity(); (1 << window_size) - 1]; num_windows];
    parallelize(&mut table, |(table, start)| {
        for (table, idx) in table.iter_mut().zip(start..) {
            let offset = generator * C::Scalar::from(2).pow_vartime([(window_size * idx) as u64]);
            let mut acc = offset;
            for value in table.iter_mut() {
                *value = acc.into();
                acc += offset;
            }
        }
    });
    table
}

fn windowed_scalar(
    window_size: usize,
    window_mask: usize,
    idx: usize,
    repr: impl AsRef<[u8]>,
) -> usize {
    let skip_bits = idx * window_size;
    let skip_bytes = skip_bits / 8;

    let mut value = [0; size_of::<usize>()];
    for (dst, src) in value.iter_mut().zip(repr.as_ref()[skip_bytes..].iter()) {
        *dst = *src;
    }

    (usize::from_le_bytes(value) >> (skip_bits - (skip_bytes * 8))) & window_mask
}

fn windowed_scalar_mul<C: CurveAffine>(
    window_size: usize,
    window_mask: usize,
    window_table: &[Vec<C>],
    scalar: &C::Scalar,
) -> C::Curve {
    let repr = scalar.to_repr();
    let mut output = C::Curve::identity();
    for (idx, table) in window_table.iter().enumerate() {
        let scalar = windowed_scalar(window_size, window_mask, idx, repr);
        if scalar > 0 {
            output += table[scalar - 1];
        }
    }
    output
}

pub fn fixed_base_msm<'a, C: CurveAffine>(
    window_size: usize,
    window_table: &[Vec<C>],
    scalars: impl IntoIterator<Item = &'a C::Scalar>,
) -> Vec<C::Curve> {
    let window_mask = (1 << window_size) - 1;
    let scalars = scalars.into_iter().collect_vec();
    let mut outputs = vec![C::Curve::identity(); scalars.len()];
    parallelize(&mut outputs, |(outputs, start)| {
        for (output, scalar) in outputs.iter_mut().zip(scalars[start..].iter()) {
            *output = windowed_scalar_mul(window_size, window_mask, window_table, scalar);
        }
    });
    outputs
}

// Copy from https://github.com/zcash/halo2/blob/main/halo2_proofs/src/arithmetic.rs
pub fn variable_base_msm<'a, 'b, C: CurveAffine>(
    scalars: impl IntoIterator<Item = &'a C::Scalar>,
    bases: impl IntoIterator<Item = &'b C>,
) -> C::Curve {
    let scalars = scalars.into_iter().collect_vec();
    let bases = bases.into_iter().collect_vec();
    assert_eq!(scalars.len(), bases.len());

    let _timer = start_timer(|| format!("variable_base_msm-{}", scalars.len()));

    let num_threads = num_threads();
    if scalars.len() <= num_threads {
        let mut result = C::Curve::identity();
        variable_base_msm_serial(&scalars, &bases, &mut result);
        return result;
    }

    let chunk_size = div_ceil(scalars.len(), num_threads);
    let mut results = vec![C::Curve::identity(); num_threads];
    parallelize_iter(
        scalars
            .chunks(chunk_size)
            .zip(bases.chunks(chunk_size))
            .zip(results.iter_mut()),
        |((scalars, bases), result)| {
            variable_base_msm_serial(scalars, bases, result);
        },
    );
    results
        .iter()
        .fold(C::Curve::identity(), |acc, result| acc + result)
}

fn variable_base_msm_serial<C: CurveAffine>(
    scalars: &[&C::Scalar],
    bases: &[&C],
    result: &mut C::Curve,
) {
    #[derive(Clone, Copy)]
    enum CurveAcc<C: CurveAffine> {
        Empty,
        Affine(C),
        Projective(C::Curve),
    }

    impl<C: CurveAffine> CurveAcc<C> {
        fn add_assign(&mut self, rhs: &C) {
            *self = match *self {
                CurveAcc::Empty => CurveAcc::Affine(*rhs),
                CurveAcc::Affine(lhs) => CurveAcc::Projective(lhs + *rhs),
                CurveAcc::Projective(mut lhs) => {
                    lhs += *rhs;
                    CurveAcc::Projective(lhs)
                }
            }
        }

        fn add(self, mut rhs: C::Curve) -> C::Curve {
            match self {
                CurveAcc::Empty => rhs,
                CurveAcc::Affine(lhs) => {
                    rhs += lhs;
                    rhs
                }
                CurveAcc::Projective(lhs) => lhs + rhs,
            }
        }
    }

    let scalars = scalars.iter().map(|scalar| scalar.to_repr()).collect_vec();
    let num_bytes = scalars[0].as_ref().len();
    let num_bits = 8 * num_bytes;

    let window_size = window_size(scalars.len());
    let num_buckets = (1 << window_size) - 1;

    let num_windows = div_ceil(num_bits, window_size);
    for idx in (0..num_windows).rev() {
        for _ in 0..window_size {
            *result = result.double();
        }

        let mut buckets = vec![CurveAcc::Empty; num_buckets];

        for (scalar, base) in scalars.iter().zip(bases.iter().cloned()) {
            let scalar = windowed_scalar(window_size, num_buckets, idx, scalar);
            if scalar != 0 {
                buckets[scalar - 1].add_assign(base);
            }
        }

        let mut running_sum = C::Curve::identity();
        for bucket in buckets.into_iter().rev() {
            running_sum = bucket.add(running_sum);
            *result += &running_sum;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Msm<'a, F: Field, T: Additive<F>> {
    Scalar(F),
    Terms(F, Vec<(F, &'a T)>),
}

impl<'a, F: Field, T: Additive<F>> Msm<'a, F, T> {
    pub fn scalar(scalar: F) -> Self {
        Self::Scalar(scalar)
    }

    pub fn base(base: &'a T) -> Self {
        Self::term(F::ONE, base)
    }

    pub fn term(scalar: F, base: &'a T) -> Self {
        Self::Terms(F::ZERO, vec![(scalar, base)])
    }

    pub fn evaluate(self) -> (F, T) {
        match self {
            Msm::Scalar(constant) => (constant, T::default()),
            Msm::Terms(constant, terms) => {
                let (scalars, bases) = terms.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                (constant, T::msm(&scalars, bases))
            }
        }
    }
}

impl<'a, F: Field, T: Additive<F>> Default for Msm<'a, F, T> {
    fn default() -> Self {
        Msm::Terms(F::ZERO, Vec::new())
    }
}

impl<'a, F: Field, T: Additive<F>> Neg for Msm<'a, F, T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        match &mut self {
            Msm::Scalar(constant) => *constant = -*constant,
            Msm::Terms(constant, terms) => {
                *constant = -*constant;
                terms.iter_mut().for_each(|(scalar, _)| *scalar = -*scalar);
            }
        }
        self
    }
}

impl<'a, F: Field, T: Additive<F>> Add for Msm<'a, F, T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Msm::Scalar(lhs), Msm::Scalar(rhs)) => Msm::Scalar(lhs + rhs),
            (Msm::Scalar(scalar), Msm::Terms(constant, terms))
            | (Msm::Terms(constant, terms), Msm::Scalar(scalar)) => {
                Msm::Terms(constant + scalar, terms)
            }
            (Msm::Terms(lhs_constant, lhs_terms), Msm::Terms(rhs_constant, rhs_terms)) => {
                Msm::Terms(
                    lhs_constant + rhs_constant,
                    chain![lhs_terms, rhs_terms].collect(),
                )
            }
        }
    }
}

impl<'a, F: Field, T: Additive<F>> Sub for Msm<'a, F, T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a, F: Field, T: Additive<F>> Mul for Msm<'a, F, T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Msm::Scalar(lhs), Msm::Scalar(rhs)) => Msm::Scalar(lhs * rhs),
            (Msm::Scalar(rhs), Msm::Terms(constant, terms))
            | (Msm::Terms(constant, terms), Msm::Scalar(rhs)) => Msm::Terms(
                constant * rhs,
                chain![terms].map(|(lhs, base)| (lhs * rhs, base)).collect(),
            ),
            (Msm::Terms(_, _), Msm::Terms(_, _)) => unreachable!(),
        }
    }
}

impl<'a, F: Field, T: Additive<F>> Sum for Msm<'a, F, T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap()
    }
}

impl<'c, F: Field, T: Additive<F>> Additive<F> for Msm<'c, F, T> {
    fn msm<'a, 'b>(
        scalars: impl IntoIterator<Item = &'a F>,
        bases: impl IntoIterator<Item = &'b Self>,
    ) -> Self
    where
        Self: 'b,
    {
        izip_eq!(scalars, bases)
            .map(|(scalar, base)| Msm::scalar(*scalar) * base.clone())
            .sum()
    }
}
