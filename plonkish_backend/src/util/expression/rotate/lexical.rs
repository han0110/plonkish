use crate::util::expression::{rotate::Rotatable, Rotation};

#[derive(Clone, Copy, Debug)]
pub struct Lexical {
    k: usize,
    n: usize,
}

impl Lexical {
    pub const fn new(k: usize) -> Self {
        assert!(k > 0);
        Self { k, n: 1 << k }
    }
}

impl From<usize> for Lexical {
    fn from(k: usize) -> Self {
        Self::new(k)
    }
}

impl Rotatable for Lexical {
    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.n
    }

    fn usable_indices(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    fn max_rotation(&self) -> usize {
        self.n
    }

    fn rotate(&self, idx: usize, rotation: Rotation) -> usize {
        (idx as i32 + rotation.0).rem_euclid(self.n as i32) as usize
    }

    fn rotation_map(&self, rotation: Rotation) -> Vec<usize> {
        (0..self.n)
            .cycle()
            .skip(self.rotate(0, rotation))
            .take(self.n)
            .collect()
    }

    fn nth(&self, nth: i32) -> usize {
        self.rotate(0, Rotation(nth))
    }
}
