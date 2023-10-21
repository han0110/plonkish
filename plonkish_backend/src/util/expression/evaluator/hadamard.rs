use crate::util::{
    arithmetic::PrimeField,
    expression::{evaluator::ExpressionRegistry, rotate::Rotatable, Expression},
    izip_eq, Itertools,
};
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub(crate) struct HadamardEvaluator<'a, F: PrimeField, R: Rotatable + From<usize>> {
    pub(crate) num_vars: usize,
    pub(crate) reg: ExpressionRegistry<F>,
    lagranges: Vec<usize>,
    polys: Vec<Cow<'a, [F]>>,
    rotatable: R,
}

impl<'a, F: PrimeField, R: Rotatable + From<usize>> HadamardEvaluator<'a, F, R> {
    pub(crate) fn new(
        num_vars: usize,
        expressions: &[Expression<F>],
        polys: impl IntoIterator<Item = Cow<'a, [F]>>,
    ) -> Self {
        let mut reg = ExpressionRegistry::new();
        for expression in expressions.iter() {
            reg.register(expression);
        }
        assert!(reg.eq_xys().is_empty());

        let rotatable = R::from(num_vars);
        let lagranges = reg
            .lagranges()
            .iter()
            .map(|i| rotatable.nth(*i))
            .collect_vec();

        Self {
            num_vars,
            reg,
            lagranges,
            polys: polys.into_iter().collect(),
            rotatable,
        }
    }

    pub(crate) fn cache(&self) -> Vec<F> {
        self.reg.cache()
    }

    pub(crate) fn evaluate(&self, evals: &mut [F], cache: &mut [F], b: usize) {
        self.evaluate_calculations(cache, b);
        izip_eq!(evals, self.reg.indexed_outputs()).for_each(|(eval, idx)| *eval = cache[*idx])
    }

    pub(crate) fn evaluate_and_sum(&self, sums: &mut [F], cache: &mut [F], b: usize) {
        self.evaluate_calculations(cache, b);
        izip_eq!(sums, self.reg.indexed_outputs()).for_each(|(sum, idx)| *sum += cache[*idx])
    }

    fn evaluate_calculations(&self, cache: &mut [F], b: usize) {
        if self.reg.has_identity() {
            cache[self.reg.offsets().identity()] = F::from(b as u64);
        }
        cache[self.reg.offsets().lagranges()..]
            .iter_mut()
            .zip(&self.lagranges)
            .for_each(|(value, i)| *value = if &b == i { F::ONE } else { F::ZERO });
        cache[self.reg.offsets().polys()..]
            .iter_mut()
            .zip(self.reg.polys())
            .for_each(|(value, (query, _))| {
                *value = self.polys[query.poly()][self.rotatable.rotate(b, query.rotation())]
            });
        self.reg
            .indexed_calculations()
            .iter()
            .zip(self.reg.offsets().calculations()..)
            .for_each(|(calculation, idx)| calculation.calculate(cache, idx));
    }
}
