use crate::{
    piop::sum_check::VirtualPolynomialInfo,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, BooleanHypercube, PrimeField},
        expression::{CommonPolynomial, Query},
        num_threads, parallelize_iter, Itertools,
    },
};
use num_integer::Integer;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    mem,
    ops::Range,
};

#[derive(Debug)]
pub struct VirtualPolynomial<'a, F> {
    pub(crate) info: &'a VirtualPolynomialInfo<F>,
    pub(crate) polys: Vec<&'a MultilinearPolynomial<F>>,
    pub(crate) challenges: Vec<F>,
    pub(crate) ys: Vec<Vec<F>>,
}

impl<'a, F: PrimeField> VirtualPolynomial<'a, F> {
    pub fn new(
        info: &'a VirtualPolynomialInfo<F>,
        polys: impl IntoIterator<Item = &'a MultilinearPolynomial<F>>,
        challenges: Vec<F>,
        ys: Vec<Vec<F>>,
    ) -> Self {
        Self {
            info,
            polys: polys.into_iter().collect(),
            challenges,
            ys,
        }
    }
}

#[derive(Debug)]
pub struct ProvingState<'a, F: PrimeField> {
    num_vars: usize,
    virtual_poly: VirtualPolynomial<'a, F>,
    lagranges: HashMap<(i32, usize), (bool, F)>,
    eq_xys: Vec<MultilinearPolynomial<F>>,
    identities: Vec<F>,
    polys: Vec<Vec<Cow<'a, MultilinearPolynomial<F>>>>,
    round: usize,
    bh: BooleanHypercube,
}

impl<'a, F: PrimeField> ProvingState<'a, F> {
    pub fn new(num_vars: usize, virtual_poly: VirtualPolynomial<'a, F>) -> Self {
        let expression = virtual_poly.info.expression();
        assert!(num_vars > 0 && expression.max_used_rotation_distance() <= num_vars);
        let lagranges = {
            let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
            expression
                .used_langrange()
                .into_iter()
                .map(|i| {
                    let b = bh[i.rem_euclid(1 << num_vars) as usize];
                    ((i, b >> 1), (b.is_even(), F::one()))
                })
                .collect()
        };
        let eq_xys = virtual_poly
            .ys
            .iter()
            .map(|y| MultilinearPolynomial::eq_xy(y))
            .collect_vec();
        let identities = (0..)
            .map(|idx| F::from((idx as u64) << num_vars))
            .take(
                expression
                    .used_identity()
                    .into_iter()
                    .max()
                    .unwrap_or_default()
                    + 1,
            )
            .collect_vec();
        let polys = virtual_poly
            .polys
            .iter()
            .map(|poly| {
                let mut polys = vec![Cow::Owned(MultilinearPolynomial::zero()); 2 * num_vars];
                polys[num_vars] = Cow::Borrowed(*poly);
                polys
            })
            .collect_vec();
        Self {
            num_vars,
            virtual_poly,
            lagranges,
            eq_xys,
            identities,
            polys,
            round: 0,
            bh: BooleanHypercube::new(num_vars),
        }
    }

    pub fn sample_evals(&self) -> Vec<F> {
        let size = 1 << (self.num_vars - self.round - 1);
        let points = self
            .virtual_poly
            .info
            .sample_points()
            .into_iter()
            .map(F::from)
            .collect_vec();

        let evaluate = |range: Range<usize>, point: &F| {
            if self.round == 0 {
                self.evaluate::<true>(range, point)
            } else {
                self.evaluate::<false>(range, point)
            }
        };

        if size < 32 {
            points
                .iter()
                .map(|point| evaluate(0..size, point))
                .collect()
        } else {
            let num_threads = num_threads();
            let chunk_size = div_ceil(size, num_threads);
            points
                .iter()
                .map(|point| {
                    let mut partials = vec![F::zero(); num_threads];
                    parallelize_iter(
                        partials.iter_mut().zip((0..).step_by(chunk_size)),
                        |(partial, start)| {
                            *partial = evaluate(start..(start + chunk_size).min(size), point);
                        },
                    );
                    partials
                        .into_iter()
                        .reduce(|acc, partial| acc + &partial)
                        .unwrap()
                })
                .collect()
        }
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
        self.identities
            .iter_mut()
            .for_each(|constant| *constant += challenge * F::from(1 << self.round));
        if self.round == 0 {
            let poly_rotations = self
                .virtual_poly
                .info
                .expression()
                .used_query()
                .into_iter()
                .fold(
                    vec![HashSet::<_>::new(); self.polys.len()],
                    |mut poly_rotations, query| {
                        if query.rotation().0 != 0 {
                            poly_rotations[query.poly()].insert(query.rotation());
                        }
                        poly_rotations
                    },
                );
            self.polys = mem::take(&mut self.polys)
                .into_iter()
                .zip(poly_rotations)
                .map(|(mut polys, rotations)| {
                    for rotation in rotations {
                        let idx = (rotation.0 + self.num_vars as i32) as usize;
                        polys[idx] = {
                            let poly = &polys[self.num_vars];
                            let rotated = MultilinearPolynomial::new(
                                (0..1 << self.num_vars)
                                    .map(|b| poly[self.bh.rotate(b, rotation)])
                                    .collect_vec(),
                            );
                            Cow::Owned(rotated.fix_variables(&[challenge]))
                        };
                    }
                    polys[self.num_vars] =
                        Cow::Owned(polys[self.num_vars].fix_variables(&[challenge]));
                    polys
                })
                .collect();
        } else {
            self.polys.iter_mut().for_each(|polys| {
                polys.iter_mut().for_each(|poly| {
                    if !poly.is_zero() {
                        *poly = Cow::Owned(poly.fix_variables(&[challenge]));
                    }
                });
            });
        }
        self.round += 1;
        self.bh = BooleanHypercube::new(self.num_vars - self.round);
    }

    fn evaluate<const IS_FIRST_ROUND: bool>(&self, range: Range<usize>, point: &F) -> F {
        let partial_identity_eval = F::from((1 << self.round) as u64) * point;

        let mut sum = F::zero();
        for b in range {
            sum += self.virtual_poly.info.expression().evaluate(
                &|scalar| scalar,
                &|poly| match poly {
                    CommonPolynomial::Lagrange(i) => self.lagrange(i, b, point),
                    CommonPolynomial::EqXY(idx) => {
                        let poly = &self.eq_xys[idx];
                        poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * point
                    }
                    CommonPolynomial::Identity(idx) => {
                        self.identities[idx]
                            + F::from((b << (self.round + 1)) as u64)
                            + &partial_identity_eval
                    }
                },
                &|query| {
                    let (b_0, b_1) = if IS_FIRST_ROUND {
                        (
                            self.bh.rotate(b << 1, query.rotation()),
                            self.bh.rotate((b << 1) + 1, query.rotation()),
                        )
                    } else {
                        (b << 1, (b << 1) + 1)
                    };
                    let poly = self.poly::<IS_FIRST_ROUND>(query);
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
    }

    fn poly<const IS_FIRST_ROUND: bool>(&self, query: Query) -> &MultilinearPolynomial<F> {
        if IS_FIRST_ROUND {
            &self.polys[query.poly()][self.num_vars]
        } else {
            &self.polys[query.poly()][(query.rotation().0 + self.num_vars as i32) as usize]
        }
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
}
