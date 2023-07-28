use crate::util::arithmetic::PrimeField;
use sha3::digest::{Digest, HashMarker};
use std::fmt::Debug;

pub use sha3::{
    digest::{FixedOutputReset, Output, Update},
    Keccak256,
};

pub use poseidon::{self, Poseidon};

pub trait Hash:
    'static + Sized + Clone + Debug + FixedOutputReset + Default + Update + HashMarker
{
    fn new() -> Self {
        Self::default()
    }

    fn update_field_element(&mut self, field: &impl PrimeField) {
        Digest::update(self, field.to_repr());
    }

    fn digest(data: impl AsRef<[u8]>) -> Output<Self> {
        let mut hasher = Self::default();
        hasher.update(data.as_ref());
        hasher.finalize()
    }
}

impl<T: 'static + Sized + Clone + Debug + FixedOutputReset + Default + Update + HashMarker> Hash
    for T
{
}
