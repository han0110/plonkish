use num_integer::Integer;

pub mod arithmetic;
pub mod expression;
mod timer;
pub mod transcript;

pub use itertools::Itertools;
pub use rayon::current_num_threads as num_threads;
pub use timer::{end_timer, start_timer};

pub fn parallelize_iter<I, T, F>(iter: I, f: F)
where
    I: Send + Iterator<Item = T>,
    T: Send,
    F: Fn(T) + Send + Sync + Clone,
{
    rayon::scope(|scope| {
        for item in iter {
            let f = f.clone();
            scope.spawn(move |_| f(item));
        }
    });
}

pub fn parallelize<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn((&mut [T], usize)) + Send + Sync + Clone,
{
    {
        let num_threads = num_threads();
        let chunk_size = Integer::div_ceil(&v.len(), &num_threads);
        if chunk_size < num_threads {
            f((v, 0));
        } else {
            parallelize_iter(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
        }
    }
}

pub trait BitIndex {
    fn nth_bit(&self, nth: usize) -> bool;
}

impl BitIndex for usize {
    fn nth_bit(&self, nth: usize) -> bool {
        (self >> nth) & 1 == 1
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::util::arithmetic::Field;
    use rand::RngCore;
    use std::{array, iter, ops::Range};

    pub fn rand_idx(range: Range<usize>, mut rng: impl RngCore) -> usize {
        range.start + (rng.next_u64() as usize % (range.end - range.start))
    }

    pub fn rand_array<F: Field, const N: usize>(mut rng: impl RngCore) -> [F; N] {
        array::from_fn(|_| F::random(&mut rng))
    }

    pub fn rand_vec<F: Field>(n: usize, mut rng: impl RngCore) -> Vec<F> {
        iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
    }
}
