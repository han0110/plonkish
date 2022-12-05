use crate::{
    poly::multilinear::MultilinearPolynomial,
    sum_check::VirtualPolynomial,
    util::{
        arithmetic::{BooleanHypercube, PrimeField},
        expression::{CommonPolynomial, Rotation},
        num_threads, parallelize_iter, Itertools,
    },
};
use num_integer::Integer;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    ops::Range,
};

#[derive(Debug)]
pub struct ProvingState<'a, F: PrimeField> {
    virtual_poly: &'a VirtualPolynomial<'a, F>,
    lagranges: HashMap<(i32, usize), (bool, F)>,
    eq_xys: Vec<MultilinearPolynomial<F>>,
    polys: Vec<Cow<'a, MultilinearPolynomial<F>>>,
    round: usize,
    next: Vec<usize>,
}

impl<'a, F: PrimeField> ProvingState<'a, F> {
    pub fn new(virtual_poly: &'a VirtualPolynomial<F>) -> Self {
        let bh = BooleanHypercube::new(virtual_poly.info.num_vars());
        let idx_map = bh.idx_map();
        let lagranges = virtual_poly
            .info
            .expression()
            .used_langrange()
            .into_iter()
            .map(|i| {
                let b = idx_map[i.rem_euclid((1 << virtual_poly.info.num_vars()) as i32) as usize];
                ((i, b >> 1), (b.is_even(), F::one()))
            })
            .collect();
        let eq_xys = virtual_poly
            .ys
            .iter()
            .map(|y| MultilinearPolynomial::eq_xy(y))
            .collect_vec();
        let polys = {
            let query_idx_to_poly = virtual_poly
                .info
                .expression()
                .used_query()
                .into_iter()
                .map(|query| (query.index(), query.poly()))
                .collect::<BTreeMap<_, _>>();
            (0..=query_idx_to_poly
                .iter()
                .rev()
                .next()
                .map(|(idx, _)| *idx)
                .unwrap_or_default())
                .map(|idx| {
                    query_idx_to_poly
                        .get(&idx)
                        .map(|poly| Cow::Borrowed(virtual_poly.polys[*poly]))
                        .unwrap_or_else(|| Cow::Owned(MultilinearPolynomial::zero()))
                })
                .collect_vec()
        };
        Self {
            virtual_poly,
            lagranges,
            eq_xys,
            polys,
            round: 0,
            next: bh.next_map(),
        }
    }

    pub fn sample_evals(&self) -> Vec<F> {
        let expression = self.virtual_poly.info.expression();
        let points = self.virtual_poly.info.sample_points();
        let mut sums = vec![F::zero(); points.len()];

        let evaluate_serial = |range: Range<usize>, point: &F| {
            let mut sum = F::zero();
            for b in range {
                sum += expression.evaluate(
                    &|scalar| scalar,
                    &|poly| match poly {
                        CommonPolynomial::Lagrange(i) => self.lagrange(i, b, point),
                        CommonPolynomial::EqXY(idx) => {
                            let poly = &self.eq_xys[idx];
                            poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * point
                        }
                    },
                    &|query| {
                        let (b_0, b_1) = if self.round == 0 {
                            (
                                self.rotate(b << 1, query.rotation()),
                                self.rotate((b << 1) + 1, query.rotation()),
                            )
                        } else {
                            (b << 1, (b << 1) + 1)
                        };
                        let poly = &self.polys[query.index()];
                        poly[b_0] + (poly[b_1] - poly[b_0]) * point
                    },
                    &|idx| self.virtual_poly.challenges[idx],
                    &|scalar| -scalar,
                    &|lhs, rhs| lhs + &rhs,
                    &|lhs, rhs| lhs * &rhs,
                    &|value, scalar| scalar * &value,
                );
            }
            sum
        };

        let size = 1 << (self.virtual_poly.info.num_vars() - self.round - 1);
        for (point, sum) in points.iter().zip(sums.iter_mut()) {
            let point = &F::from(*point as u64);

            if size < 32 {
                *sum = evaluate_serial(0..size, point);
            } else {
                let num_threads = num_threads();
                let chunk_size = Integer::div_ceil(&size, &num_threads);
                let mut partials = vec![F::zero(); num_threads];
                parallelize_iter(
                    partials.iter_mut().zip((0..).step_by(chunk_size)),
                    |(partial, start)| {
                        *partial = evaluate_serial(start..start + chunk_size, point);
                    },
                );
                *sum = partials
                    .into_iter()
                    .reduce(|acc, partial| acc + &partial)
                    .unwrap();
            }
        }

        sums
    }

    pub fn next_round(&mut self, challenge: F) {
        self.lagranges = self
            .lagranges
            .drain()
            .into_iter()
            .map(|((i, b), (is_even, value))| {
                let mut output = value * &challenge;
                if is_even {
                    output = value - &output;
                }
                ((i, b >> 1), (b.is_even(), output))
            })
            .collect();
        self.eq_xys
            .iter_mut()
            .for_each(|eq_xy| *eq_xy = eq_xy.fix_variables(&[challenge]));
        if self.round == 0 {
            let query_idx_to_rotation = self
                .virtual_poly
                .info
                .expression()
                .used_query()
                .into_iter()
                .map(|query| (query.index(), query.rotation()))
                .collect::<BTreeMap<_, _>>();
            self.polys = self
                .polys
                .iter()
                .enumerate()
                .map(|(idx, poly)| {
                    match (poly.is_zero(), query_idx_to_rotation.get(&idx).copied()) {
                        (true, _) => Cow::Owned(MultilinearPolynomial::zero()),
                        (false, Some(Rotation(0))) => Cow::Owned(poly.fix_variables(&[challenge])),
                        (false, Some(rotation)) => {
                            let poly = MultilinearPolynomial::new(
                                (0..1 << self.virtual_poly.info.num_vars())
                                    .map(|b| poly[self.rotate(b, rotation)])
                                    .collect_vec(),
                            );
                            Cow::Owned(poly.fix_variables(&[challenge]))
                        }
                        _ => unreachable!(),
                    }
                })
                .collect_vec();
        } else {
            self.polys.iter_mut().for_each(|poly| {
                if !poly.is_zero() {
                    *poly = Cow::Owned(poly.fix_variables(&[challenge]));
                }
            });
        }
        self.next =
            BooleanHypercube::new(self.virtual_poly.info.num_vars() - self.round - 1).next_map();
        self.round += 1;
    }

    fn lagrange(&self, i: i32, b: usize, point: &F) -> F {
        self.lagranges
            .get(&(i, b))
            .map(|(is_even, value)| {
                let output = *value * point;
                if *is_even {
                    *value - &output
                } else {
                    output
                }
            })
            .unwrap_or_else(F::zero)
    }

    fn rotate(&self, mut b: usize, rotation: Rotation) -> usize {
        match rotation.0.cmp(&0) {
            Ordering::Less => unimplemented!("Negative roation is not supported yet"),
            Ordering::Equal => b,
            Ordering::Greater => {
                for _ in 0..rotation.0 as usize {
                    b = self.next[b];
                }
                b
            }
        }
    }
}
