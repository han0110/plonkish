use crate::util::arithmetic::PrimeField;
use sha3::digest::{Digest, HashMarker};

pub use sha3::{
    digest::{FixedOutputReset, Output, Update},
    Keccak256,
};

pub trait Hash: FixedOutputReset + Default + Update + HashMarker {
    fn new() -> Self {
        Self::default()
    }

    fn update_field_element(&mut self, field: &impl PrimeField) {
        Digest::update(self, field.to_repr());
    }
}

impl<T: FixedOutputReset + Default + Update + HashMarker> Hash for T {}
