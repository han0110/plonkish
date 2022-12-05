use std::iter;

/// Integer representation of irreducible fucntion in GF(2) as masks.
/// Mostly taken from https://www.hpl.hp.com/techreports/98/HPL-98-135.pdf,
/// some of them don't work initially, but with some tweak now they all work.
const MASK: [usize; 32] = [
    0,          // [0]
    3,          // [0, 1]
    7,          // [0, 1, 2]
    11,         // [0, 1, 3]
    19,         // [0, 1, 4]
    37,         // [0, 2, 5]
    67,         // [0, 1, 6]
    131,        // [0, 1, 7]
    285,        // [0, 2, 3, 4, 8]
    529,        // [0, 4, 9]
    1033,       // [0, 3, 10]
    2053,       // [0, 2, 11]
    4179,       // [0, 1, 4, 6, 12]
    8219,       // [0, 1, 3, 4, 13]
    16427,      // [0, 1, 3, 5, 14]
    32771,      // [0, 1, 15]
    65581,      // [0, 2, 3, 5, 16]
    131081,     // [0, 3, 17]
    262183,     // [0, 1, 2, 5, 18]
    524327,     // [0, 1, 2, 5, 19]
    1048585,    // [0, 3, 20]
    2097157,    // [0, 2, 21]
    4194307,    // [0, 1, 22]
    8388641,    // [0, 5, 23]
    16777243,   // [0, 1, 3, 4, 24]
    33554441,   // [0, 3, 25]
    67108935,   // [0, 1, 2, 6, 26]
    134217767,  // [0, 1, 2, 5, 27]
    268435465,  // [0, 3, 28]
    536870917,  // [0, 2, 29]
    1073741907, // [0, 1, 4, 6, 30]
    2147483657, // [0, 3, 31]
];

#[derive(Debug, Clone, Copy)]
pub struct BooleanHypercube {
    num_vars: usize,
    mask: usize,
}

impl BooleanHypercube {
    pub fn new(num_vars: usize) -> Self {
        assert!(num_vars <= 31);
        Self {
            num_vars,
            mask: MASK[num_vars],
        }
    }

    pub fn mask(&self) -> usize {
        self.mask
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut b = 1;
        iter::once(0)
            .chain(iter::repeat_with(move || {
                let item = b;
                b = next(b, self.num_vars, self.mask);
                item
            }))
            .take(1 << self.num_vars)
    }

    pub fn next_map(&self) -> Vec<usize> {
        (0..1 << self.num_vars)
            .map(|b| next(b, self.num_vars, self.mask))
            .collect()
    }

    pub fn idx_map(&self) -> Vec<usize> {
        let mut idx_map = vec![0; 1 << self.num_vars];
        let mut b = 1;
        for idx in 1..1 << self.num_vars {
            idx_map[b] = idx;
            b = next(b, self.num_vars, self.mask);
        }
        idx_map
    }
}

#[inline(always)]
fn next(mut b: usize, num_vars: usize, mask: usize) -> usize {
    b <<= 1;
    b ^= (b >> num_vars) * mask;
    b
}

#[cfg(test)]
mod test {
    use crate::util::arithmetic::BooleanHypercube;

    #[test]
    #[ignore = "Cause it takes some minutes to run with release profile"]
    fn test_boolean_hypercube() {
        for num_vars in 0..32 {
            let bh = BooleanHypercube::new(num_vars);
            let mut set = vec![false; 1 << num_vars];
            for i in bh.iter() {
                if set[i] {
                    panic!(
                        "Found repeated item while iterating the boolean hypercube with {num_vars}"
                    );
                }
                set[i] = true;
            }
        }
    }
}
