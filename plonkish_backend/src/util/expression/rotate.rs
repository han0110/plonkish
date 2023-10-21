use crate::util::{Deserialize, Serialize};
use std::{fmt::Debug, ops::Neg};

mod binary_field;
mod lexical;

pub use binary_field::BinaryField;
pub use lexical::Lexical;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Rotation(pub i32);

impl Rotation {
    pub const fn cur() -> Self {
        Rotation(0)
    }

    pub const fn prev() -> Self {
        Rotation(-1)
    }

    pub const fn next() -> Self {
        Rotation(1)
    }

    pub const fn distance(&self) -> usize {
        self.0.unsigned_abs() as usize
    }

    pub fn positive(&self, n: usize) -> Rotation {
        Rotation(self.0.rem_euclid(n as i32))
    }
}

impl From<i32> for Rotation {
    fn from(rotation: i32) -> Self {
        Self(rotation)
    }
}

impl Neg for Rotation {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

pub trait Rotatable: 'static + Debug + Send + Sync {
    /// Return `self.n().ilog2()`
    fn k(&self) -> usize;

    /// Return `self.usable_indices().next_power_of_two()`
    fn n(&self) -> usize;

    /// Return usable indices that are cyclic
    fn usable_indices(&self) -> Vec<usize>;

    /// Return maximum rotation the implementation supports
    fn max_rotation(&self) -> usize;

    /// Rotate `idx` by `rotation``
    fn rotate(&self, idx: usize, rotation: Rotation) -> usize;

    /// Return a map from `idx` to `self.rotate(idx, rotation)`
    fn rotation_map(&self, rotation: Rotation) -> Vec<usize>;

    /// Return `self.usable_indices()[nth]`
    fn nth(&self, nth: i32) -> usize;
}

impl Rotatable for usize {
    fn k(&self) -> usize {
        *self
    }

    fn n(&self) -> usize {
        1 << self
    }

    fn usable_indices(&self) -> Vec<usize> {
        (0..1 << self).collect()
    }

    fn max_rotation(&self) -> usize {
        0
    }

    fn rotate(&self, idx: usize, rotation: Rotation) -> usize {
        if rotation.0 == 0 {
            return idx;
        }
        unreachable!()
    }

    fn rotation_map(&self, rotation: Rotation) -> Vec<usize> {
        if rotation.0 == 0 {
            return self.usable_indices();
        }
        unreachable!()
    }

    fn nth(&self, nth: i32) -> usize {
        nth.rem_euclid(1 << self) as usize
    }
}
