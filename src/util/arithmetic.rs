use crate::util::Itertools;
use halo2_curves::pairing::{self, MillerLoopResult};
use num_bigint::BigUint;
use std::{borrow::Borrow, fmt::Debug, iter};

mod bh;
mod msm;

pub use bh::BooleanHypercube;
pub use halo2_curves::{
    group::{
        ff::{BatchInvert, Field, PrimeField, PrimeFieldBits},
        prime::PrimeCurveAffine,
        Curve, Group,
    },
    Coordinates, CurveAffine,
};
pub use msm::{fixed_base_msm, variable_base_msm, window_size, window_table};

pub trait MultiMillerLoop: pairing::MultiMillerLoop + Debug {
    fn pairings_product_is_identity(terms: &[(&Self::G1Affine, &Self::G2Prepared)]) -> bool {
        Self::multi_miller_loop(terms)
            .final_exponentiation()
            .is_identity()
            .into()
    }
}

impl<M> MultiMillerLoop for M where M: pairing::MultiMillerLoop + Debug {}

pub fn field_size<F: PrimeField>() -> usize {
    let neg_one = (-F::one()).to_repr();
    let bytes = neg_one.as_ref();
    debug_assert!(bytes[0] != 0xff);
    8 * bytes.len() - bytes.last().unwrap().leading_zeros() as usize
}

pub fn ilog2(n: usize) -> usize {
    assert!(n > 0);
    (usize::BITS - n.leading_zeros() - 1) as usize
}

pub fn horner<F: Field>(coeffs: &[F], x: F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::zero(), |acc, coeff| acc * x + coeff)
}

pub fn powers<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::one()), move |power| Some(scalar * power))
}

pub fn product<F: Field>(values: impl IntoIterator<Item = impl Borrow<F>>) -> F {
    values
        .into_iter()
        .fold(F::one(), |acc, value| acc * value.borrow())
}

pub fn inner_product<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    lhs.iter()
        .zip_eq(rhs.iter())
        .map(|(lhs, rhs)| *lhs * rhs)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

pub fn modulus<F: PrimeField>() -> BigUint {
    BigUint::from_bytes_le((-F::one()).to_repr().as_ref()) + 1u64
}

pub fn fe_from_bytes_le<F: PrimeField>(bytes: impl AsRef<[u8]>) -> F {
    let bytes = (BigUint::from_bytes_le(bytes.as_ref()) % modulus::<F>()).to_bytes_le();
    let mut repr = F::Repr::default();
    assert!(bytes.len() <= repr.as_ref().len());
    repr.as_mut()[..bytes.len()].copy_from_slice(&bytes);
    F::from_repr(repr).unwrap()
}

pub fn neg_plus_1<F: Field>(value: impl Borrow<F>) -> F {
    F::one() - value.borrow()
}

#[cfg(test)]
mod test {
    use crate::util::arithmetic::field_size;
    use halo2_curves::bn256;

    #[test]
    fn test_field_size() {
        assert_eq!(field_size::<bn256::Fr>(), 254);
    }
}
