pub mod multilinear;
pub mod univariate;

macro_rules! impl_index {
    ($name:ident, $field:tt, [$($range:ty => $output:ty,)*]) => {
        $(
            #[allow(unused_imports)]
            use std::ops::*;

            impl<F> Index<$range> for $name<F> {
                type Output = $output;

                fn index(&self, index: $range) -> &$output {
                    self.$field.index(index)
                }
            }

            impl<F> IndexMut<$range> for $name<F> {
                fn index_mut(&mut self, index: $range) -> &mut $output {
                    self.$field.index_mut(index)
                }
            }
        )*
    };
}

pub(crate) use impl_index;
