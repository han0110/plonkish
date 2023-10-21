use crate::util::{
    arithmetic::{
        radix2_fft, root_of_unity, root_of_unity_inv, BatchInvert, WithSmallOrderMulGroup,
    },
    chain,
    expression::{
        evaluator::ExpressionRegistry,
        rotate::{Lexical, Rotatable},
        CommonPolynomial, Expression, Query, Rotation,
    },
    izip,
    parallel::parallelize,
    BitIndex, Itertools,
};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    iter,
};

#[derive(Clone, Debug)]
pub struct Radix2Domain<F> {
    k: usize,
    n: usize,
    extended_k: usize,
    extended_n: usize,
    magnification: usize,
    omega: F,
    omega_inv: F,
    extended_omega: F,
    extended_omega_inv: F,
    zeta: F,
    zeta_inv: F,
    n_inv: F,
    n_inv_zeta: F,
    n_inv_zeta_inv: F,
    extended_n_inv: F,
    extended_n_inv_zeta: F,
    extended_n_inv_zeta_inv: F,
}

impl<F: WithSmallOrderMulGroup<3>> Radix2Domain<F> {
    pub fn new(k: usize, degree: usize) -> Self {
        let quotient_degree = degree.checked_sub(1).unwrap_or_default();
        let extended_k = k + quotient_degree.next_power_of_two().ilog2() as usize;
        let extended_n = 1 << extended_k;
        assert!(extended_k <= F::S as usize);

        let omega = root_of_unity(k);
        let omega_inv = root_of_unity_inv(k);
        let extended_omega = root_of_unity::<F>(extended_k);
        let extended_omega_inv = root_of_unity_inv(extended_k);

        let zeta = F::ZETA;
        let zeta_inv = F::ZETA.square();
        assert_eq!(zeta * zeta_inv, F::ONE);

        let n_inv = F::TWO_INV.pow([k as u64]);
        let n_inv_zeta = n_inv * zeta;
        let n_inv_zeta_inv = n_inv * zeta_inv;
        let extended_n_inv = F::TWO_INV.pow([extended_k as u64]);
        let extended_n_inv_zeta = extended_n_inv * zeta;
        let extended_n_inv_zeta_inv = extended_n_inv * zeta_inv;

        Self {
            k,
            n: 1 << k,
            extended_k,
            extended_n,
            magnification: 1 << (extended_k - k),
            omega,
            omega_inv,
            extended_omega,
            extended_omega_inv,
            zeta,
            zeta_inv,
            n_inv,
            n_inv_zeta,
            n_inv_zeta_inv,
            extended_n_inv,
            extended_n_inv_zeta,
            extended_n_inv_zeta_inv,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn n_inv(&self) -> F {
        self.n_inv
    }

    pub fn extended_k(&self) -> usize {
        self.extended_k
    }

    pub fn extended_n(&self) -> usize {
        self.extended_n
    }

    pub fn extended_omega(&self) -> F {
        self.extended_omega
    }

    pub fn zeta(&self) -> F {
        self.zeta
    }

    pub fn zeta_inv(&self) -> F {
        self.zeta_inv
    }

    pub fn rotate_point(&self, x: F, rotation: Rotation) -> F {
        let rotation = Lexical::new(self.k).rotate(0, rotation);
        let rotation = if rotation > self.n >> 1 {
            rotation as i32 - self.n as i32
        } else {
            rotation as i32
        };
        let omega = match rotation.cmp(&0) {
            Ordering::Less => self.omega_inv,
            Ordering::Greater => self.omega,
            Ordering::Equal => return x,
        };
        let exponent = rotation.unsigned_abs() as usize;
        let mut scalar = F::ONE;
        for nth in (1..=(usize::BITS - exponent.leading_zeros()) as usize).rev() {
            if exponent.nth_bit(nth) {
                scalar *= omega;
            }
            scalar = scalar.square();
        }
        if exponent.nth_bit(0) {
            scalar *= omega;
        }
        scalar * x
    }

    pub fn evaluate(
        &self,
        expression: &Expression<F>,
        evals: &HashMap<Query, F>,
        challenges: &[F],
        x: F,
    ) -> F {
        let lagrange = {
            let common = (x.pow_vartime([self.n as u64]) - F::ONE) * self.n_inv;
            let used_lagrange = expression.used_langrange();
            let mut denoms = chain![&used_lagrange]
                .map(|i| x - self.rotate_point(F::ONE, Rotation(*i)))
                .collect_vec();
            denoms.batch_invert();
            izip!(used_lagrange, denoms)
                .map(|(i, denom)| (i, self.rotate_point(common * denom, Rotation(i))))
                .collect::<BTreeMap<_, _>>()
        };

        expression.evaluate(
            &|scalar| scalar,
            &|poly| match poly {
                CommonPolynomial::Identity => x,
                CommonPolynomial::Lagrange(i) => lagrange[&i],
                CommonPolynomial::EqXY(_) => unreachable!(),
            },
            &|query| evals[&query],
            &|idx| challenges[idx],
            &|scalar| -scalar,
            &|lhs, rhs| lhs + &rhs,
            &|lhs, rhs| lhs * &rhs,
            &|value, scalar| scalar * value,
        )
    }

    pub fn lagrange_to_monomial(&self, buf: Cow<[F]>) -> Vec<F> {
        assert_eq!(buf.len(), self.n);
        let mut buf = buf.into_owned();

        radix2_fft(&mut buf, self.omega_inv, self.k);

        parallelize(&mut buf, |(buf, _)| {
            buf.iter_mut().for_each(|buf| *buf *= self.n_inv)
        });

        buf
    }

    pub fn lagrange_to_extended_lagrange(&self, buf: Cow<[F]>) -> Vec<F> {
        assert_eq!(buf.len(), self.n);
        let mut buf = buf.into_owned();

        radix2_fft(&mut buf, self.omega_inv, self.k);

        let scalars = [self.n_inv, self.n_inv_zeta, self.n_inv_zeta_inv];
        parallelize(&mut buf, |(buf, start)| {
            izip!(buf, scalars.iter().cycle().skip(start % scalars.len()))
                .for_each(|(buf, scalar)| *buf *= scalar);
        });

        buf.resize(self.extended_n, F::ZERO);
        radix2_fft(&mut buf, self.extended_omega, self.extended_k);

        buf
    }

    pub fn monomial_to_extended_lagrange(&self, buf: Cow<[F]>) -> Vec<F> {
        assert!(buf.len() <= self.n);
        let mut buf = buf.into_owned();

        let scalars = [None, Some(self.zeta), Some(self.zeta_inv)];
        parallelize(&mut buf, |(buf, start)| {
            izip!(buf, scalars.iter().cycle().skip(start % scalars.len())).for_each(
                |(buf, scalar)| {
                    scalar.map(|scalar| *buf *= scalar);
                },
            );
        });

        buf.resize(self.extended_n, F::ZERO);
        radix2_fft(&mut buf, self.extended_omega, self.extended_k);

        buf
    }

    pub fn extended_lagrange_to_monomial(&self, buf: Cow<[F]>) -> Vec<F> {
        assert_eq!(buf.len(), self.extended_n);
        let mut buf = buf.into_owned();

        radix2_fft(&mut buf, self.extended_omega_inv, self.extended_k);

        let scalars = [
            self.extended_n_inv,
            self.extended_n_inv_zeta_inv,
            self.extended_n_inv_zeta,
        ];
        parallelize(&mut buf, |(buf, start)| {
            izip!(buf, scalars.iter().cycle().skip(start % scalars.len()))
                .for_each(|(buf, scalar)| *buf *= scalar);
        });

        buf
    }
}

#[derive(Clone, Debug)]
pub struct QuotientEvaluator<'a, F: WithSmallOrderMulGroup<3>> {
    magnification: i32,
    extended_n: i32,
    reg: ExpressionRegistry<F>,
    eval_idx: usize,
    identity: Vec<F>,
    lagranges: Vec<&'a [F]>,
    polys: Vec<&'a [F]>,
    vanishing_invs: Vec<F>,
}

impl<'a, F: WithSmallOrderMulGroup<3>> QuotientEvaluator<'a, F> {
    pub fn new(
        domain: &'a Radix2Domain<F>,
        expression: &Expression<F>,
        lagranges: BTreeMap<i32, &'a [F]>,
        polys: impl IntoIterator<Item = &'a [F]>,
    ) -> Self {
        let mut reg = ExpressionRegistry::new();
        reg.register(expression);
        assert!(reg.eq_xys().is_empty());

        let eval_idx = reg.indexed_outputs()[0];

        let identity = reg
            .has_identity()
            .then(|| {
                iter::successors(Some(F::ZETA), move |state| {
                    Some(domain.extended_omega * state)
                })
                .take(domain.extended_n)
                .collect_vec()
            })
            .unwrap_or_default();
        let lagranges = reg.lagranges().iter().map(|i| lagranges[i]).collect_vec();
        let polys = polys.into_iter().collect_vec();
        let vanishing_invs = {
            let step = domain.extended_omega.pow([domain.n() as u64]);
            let mut vanishing_invs = iter::successors(
                Some(match domain.n() % 3 {
                    1 => domain.zeta,
                    2 => domain.zeta_inv,
                    _ => unreachable!(),
                }),
                |value| Some(step * value),
            )
            .map(|value| value - F::ONE)
            .take(domain.magnification)
            .collect_vec();
            vanishing_invs.batch_invert();
            vanishing_invs
        };

        Self {
            magnification: domain.magnification as i32,
            extended_n: domain.extended_n as i32,
            reg,
            eval_idx,
            identity,
            lagranges,
            polys,
            vanishing_invs,
        }
    }

    pub fn cache(&self) -> Vec<F> {
        self.reg.cache()
    }

    pub fn evaluate(&self, eval: &mut F, cache: &mut [F], row: usize) {
        if self.reg.has_identity() {
            cache[self.reg.offsets().identity()] = self.identity[row];
        }
        cache[self.reg.offsets().lagranges()..]
            .iter_mut()
            .zip(&self.lagranges)
            .for_each(|(value, lagrange)| *value = lagrange[row]);
        cache[self.reg.offsets().polys()..]
            .iter_mut()
            .zip(self.reg.polys())
            .for_each(|(value, (query, _))| {
                *value = self.polys[query.poly()][self.rotated_row(row, query.rotation())]
            });
        self.reg
            .indexed_calculations()
            .iter()
            .zip(self.reg.offsets().calculations()..)
            .for_each(|(calculation, idx)| calculation.calculate(cache, idx));
        *eval = cache[self.eval_idx] * self.vanishing_inv(row);
    }

    fn vanishing_inv(&self, row: usize) -> &F {
        &self.vanishing_invs[row % self.vanishing_invs.len()]
    }

    fn rotated_row(&self, row: usize, rotation: Rotation) -> usize {
        ((row as i32 + self.magnification * rotation.0).rem_euclid(self.extended_n)) as usize
    }
}

#[cfg(test)]
mod test {
    use crate::util::{
        arithmetic::Field,
        expression::evaluator::quotient::Radix2Domain,
        test::{rand_vec, seeded_std_rng},
        Itertools,
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn basis_conversion() {
        let mut rng = seeded_std_rng();

        let lagrange = rand_vec::<Fr>(1 << 16, &mut rng);
        for (k, degree) in (1..16).cartesian_product(1..9) {
            let domain = Radix2Domain::new(k, degree);
            let extended = domain.lagrange_to_extended_lagrange((&lagrange[..domain.n()]).into());
            let monomial = domain.extended_lagrange_to_monomial(extended.into());
            assert_eq!(
                domain.lagrange_to_monomial((&lagrange[..domain.n()]).into()),
                monomial[..domain.n()]
            );
            assert!(!monomial[domain.n()..].iter().any(|v| *v != Fr::ZERO));
        }
    }
}
