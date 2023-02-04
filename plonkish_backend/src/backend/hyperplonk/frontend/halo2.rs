use crate::{
    backend::hyperplonk::PlonkishCircuitInfo,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{BatchInvert, BooleanHypercube, Field},
        expression::{Expression, Query, Rotation},
        Itertools,
    },
};
use halo2_proofs::{
    circuit::Value,
    plonk::{
        self, Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, ConstraintSystem,
        Error, Fixed, FloorPlanner, Instance, Selector,
    },
};
use std::{
    collections::{HashMap, HashSet},
    iter, mem,
};

#[cfg(any(test, feature = "benchmark"))]
pub mod circuit;
#[cfg(test)]
mod test;

pub fn circuit_info<F, C>(
    k: usize,
    circuit: &C,
    num_instances: Vec<usize>,
) -> Result<PlonkishCircuitInfo<F>, crate::Error>
where
    F: Field,
    C: Circuit<F>,
{
    let (cs, config) = {
        let mut cs = ConstraintSystem::default();
        let config = C::configure(&mut cs);
        (cs, config)
    };

    let advice_idx = advice_idx(&cs);
    let challenge_idx = idx_order_by_phase(&cs.challenge_phase(), 0);
    let constraints = cs
        .gates()
        .iter()
        .flat_map(|gate| {
            gate.polynomials()
                .iter()
                .map(|expression| convert_expression(&cs, &advice_idx, &challenge_idx, expression))
        })
        .collect();
    let lookups = cs
        .lookups()
        .iter()
        .map(|lookup| {
            lookup
                .input_expressions()
                .iter()
                .zip(lookup.table_expressions())
                .map(|(input, table)| {
                    let [input, table] = [input, table].map(|expression| {
                        convert_expression(&cs, &advice_idx, &challenge_idx, expression)
                    });
                    (input, table)
                })
                .collect_vec()
        })
        .collect();

    let column_idx = column_idx(&cs);
    let permutation_column_idx = cs
        .permutation()
        .get_columns()
        .iter()
        .map(|column| {
            let key = (*column.column_type(), column.index());
            (key, column_idx[&key])
        })
        .collect();
    let mut preprocess_collector = PreprocessCollector {
        k: k as u32,
        num_instances: num_instances.clone(),
        fixeds: vec![vec![F::zero().into(); 1 << k]; cs.num_fixed_columns()],
        permutation: Permutation::new(permutation_column_idx),
        selectors: vec![vec![false; 1 << k]; cs.num_selectors()],
        row_map: row_map(k),
    };

    C::FloorPlanner::synthesize(
        &mut preprocess_collector,
        circuit,
        config,
        cs.constants().clone(),
    )
    .map_err(|_| crate::Error::InvalidSnark("Synthesize failure".to_string()))?;

    let preprocess_polys = iter::empty()
        .chain(batch_invert_assigned(preprocess_collector.fixeds))
        .chain(preprocess_collector.selectors.into_iter().map(|selectors| {
            selectors
                .into_iter()
                .map(|selector| selector.then(F::one).unwrap_or_else(F::zero))
                .collect()
        }))
        .map(MultilinearPolynomial::new)
        .collect();
    let permutations = preprocess_collector.permutation.into_cycles();

    Ok(PlonkishCircuitInfo {
        k,
        num_instances,
        preprocess_polys,
        num_witness_polys: num_by_phase(&cs.advice_column_phase()),
        num_challenges: num_by_phase(&cs.challenge_phase()),
        constraints,
        lookups,
        permutations,
        max_degree: Some(cs.degree::<false>()),
    })
}

pub fn witness_collector<'a, F, C>(
    k: usize,
    circuit: &'a C,
    instances: &'a [&[F]],
) -> impl Fn(&[F]) -> Result<Vec<Vec<F>>, crate::Error> + 'a
where
    F: Field,
    C: Circuit<F>,
{
    let (cs, config) = {
        let mut cs = ConstraintSystem::default();
        let config = C::configure(&mut cs);
        (cs, config)
    };

    let phase_map =
        HashMap::<_, _>::from_iter(phase_offsets(&cs.challenge_phase()).into_iter().zip(0..));
    let num_witness_polys = num_by_phase(&cs.advice_column_phase());
    let advice_idx_in_phase = idx_in_phase(&cs.advice_column_phase());
    let challenge_idx = idx_order_by_phase(&cs.challenge_phase(), 0);
    let row_map = row_map(k);

    move |challenges| {
        let phase = phase_map[&challenges.len()];
        let mut witness_collector = WitnessCollector {
            k: k as u32,
            phase,
            advice_idx_in_phase: &advice_idx_in_phase,
            challenge_idx: &challenge_idx,
            instances,
            advices: vec![vec![F::zero().into(); 1 << k]; num_witness_polys[phase as usize]],
            challenges,
            row_map: &row_map,
        };

        C::FloorPlanner::synthesize(
            &mut witness_collector,
            circuit,
            config.clone(),
            cs.constants().clone(),
        )
        .map_err(|_| crate::Error::InvalidSnark("Synthesize failure".to_string()))?;

        Ok(batch_invert_assigned(witness_collector.advices))
    }
}

#[derive(Debug)]
struct PreprocessCollector<F: Field> {
    k: u32,
    num_instances: Vec<usize>,
    fixeds: Vec<Vec<Assigned<F>>>,
    permutation: Permutation,
    selectors: Vec<Vec<bool>>,
    row_map: Vec<usize>,
}

impl<F: Field> Assignment<F> for PreprocessCollector<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let Some(row) = self.row_map.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        self.selectors[selector.index()][row] = true;

        Ok(())
    }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        self.num_instances
            .get(column.index())
            .and_then(|num_instances| (row < *num_instances).then(Value::unknown))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let Some(row) = self.row_map.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        *self
            .fixeds
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn copy(
        &mut self,
        lhs_column: Column<Any>,
        lhs_row: usize,
        rhs_column: Column<Any>,
        rhs_row: usize,
    ) -> Result<(), Error> {
        let Some(lhs_row) = self.row_map.get(lhs_row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };
        let Some(rhs_row) = self.row_map.get(rhs_row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };
        self.permutation
            .copy(lhs_column, lhs_row, rhs_column, rhs_row)
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        let Some(_) = self.row_map.get(from_row) else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        let col = self
            .fixeds
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;

        let filler = to.assign()?;
        for row in self.row_map.iter().skip(from_row).copied() {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}

#[derive(Debug)]
struct Permutation {
    column_idx: HashMap<(Any, usize), usize>,
    cycles: Vec<HashSet<(usize, usize)>>,
    cycle_idx: HashMap<(usize, usize), usize>,
}

impl Permutation {
    fn new(column_idx: HashMap<(Any, usize), usize>) -> Self {
        Self {
            column_idx,
            cycles: Default::default(),
            cycle_idx: Default::default(),
        }
    }

    fn copy(
        &mut self,
        lhs_column: Column<Any>,
        lhs_row: usize,
        rhs_column: Column<Any>,
        rhs_row: usize,
    ) -> Result<(), Error> {
        let lhs_idx = *self
            .column_idx
            .get(&(*lhs_column.column_type(), lhs_column.index()))
            .ok_or(Error::ColumnNotInPermutation(lhs_column))?;
        let rhs_idx = *self
            .column_idx
            .get(&(*rhs_column.column_type(), rhs_column.index()))
            .ok_or(Error::ColumnNotInPermutation(rhs_column))?;

        match (
            self.cycle_idx.get(&(lhs_idx, lhs_row)).copied(),
            self.cycle_idx.get(&(rhs_idx, rhs_row)).copied(),
        ) {
            (Some(lhs_cycle_idx), Some(rhs_cycle_idx)) => {
                for cell in self.cycles[rhs_cycle_idx].iter().copied() {
                    self.cycle_idx.insert(cell, lhs_cycle_idx);
                }
                let rhs_cycle = mem::take(&mut self.cycles[rhs_cycle_idx]);
                self.cycles[lhs_cycle_idx].extend(rhs_cycle);
            }
            cycle_idx => {
                let cycle_idx = if let (Some(cycle_idx), None) | (None, Some(cycle_idx)) = cycle_idx
                {
                    cycle_idx
                } else {
                    let cycle_idx = self.cycles.len();
                    self.cycles.push(Default::default());
                    cycle_idx
                };
                for cell in [(lhs_idx, lhs_row), (rhs_idx, rhs_row)] {
                    self.cycles[cycle_idx].insert(cell);
                    self.cycle_idx.insert(cell, cycle_idx);
                }
            }
        };

        Ok(())
    }

    fn into_cycles(self) -> Vec<Vec<(usize, usize)>> {
        self.cycles
            .into_iter()
            .filter_map(|cycle| {
                (!cycle.is_empty()).then(|| cycle.into_iter().sorted().collect_vec())
            })
            .collect()
    }
}

#[derive(Debug)]
struct WitnessCollector<'a, F: Field> {
    k: u32,
    phase: u8,
    advice_idx_in_phase: &'a [usize],
    challenge_idx: &'a [usize],
    instances: &'a [&'a [F]],
    advices: Vec<Vec<Assigned<F>>>,
    challenges: &'a [F],
    row_map: &'a [usize],
}

impl<'a, F: Field> Assignment<F> for WitnessCollector<'a, F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        self.instances
            .get(column.index())
            .and_then(|column| column.get(row))
            .map(|v| Value::known(*v))
            .ok_or(Error::BoundsFailure)
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Advice>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if self.phase != column.column_type().phase() {
            return Ok(());
        }

        let Some(row) = self.row_map.get(row).copied() else {
            return Err(Error::NotEnoughRowsAvailable { current_k: self.k });
        };

        *self
            .advices
            .get_mut(self.advice_idx_in_phase[column.index()])
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Fixed>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        Ok(())
    }

    fn copy(&mut self, _: Column<Any>, _: usize, _: Column<Any>, _: usize) -> Result<(), Error> {
        Ok(())
    }

    fn fill_from_row(
        &mut self,
        _: Column<Fixed>,
        _: usize,
        _: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.challenges
            .get(self.challenge_idx[challenge.index()])
            .copied()
            .map(Value::known)
            .unwrap_or_else(Value::unknown)
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}

fn advice_idx<F: Field>(cs: &ConstraintSystem<F>) -> Vec<usize> {
    let advice_offset = cs.num_instance_columns() + cs.num_fixed_columns() + cs.num_selectors();
    idx_order_by_phase(&cs.advice_column_phase(), advice_offset)
}

fn column_idx<F: Field>(cs: &ConstraintSystem<F>) -> HashMap<(Any, usize), usize> {
    let advice_idx = advice_idx(cs);
    iter::empty()
        .chain((0..cs.num_instance_columns()).map(|idx| (Any::Instance, idx)))
        .chain((0..cs.num_fixed_columns() + cs.num_selectors()).map(|idx| (Any::Fixed, idx)))
        .enumerate()
        .map(|(idx, column)| (column, idx))
        .chain((0..advice_idx.len()).map(|idx| ((Any::advice(), idx), advice_idx[idx])))
        .collect()
}

fn num_phases(phases: &[u8]) -> usize {
    phases.iter().max().copied().unwrap_or_default() as usize + 1
}

fn num_by_phase(phases: &[u8]) -> Vec<usize> {
    phases.iter().copied().fold(
        vec![0usize; num_phases(phases)],
        |mut num_by_phase, phase| {
            num_by_phase[phase as usize] += 1;
            num_by_phase
        },
    )
}

fn idx_in_phase(phases: &[u8]) -> Vec<usize> {
    phases
        .iter()
        .copied()
        .scan(vec![0; num_phases(phases)], |state, phase| {
            let index = state[phase as usize];
            state[phase as usize] += 1;
            Some(index)
        })
        .collect_vec()
}

fn idx_order_by_phase(phases: &[u8], offset: usize) -> Vec<usize> {
    phases
        .iter()
        .copied()
        .scan(phase_offsets(phases), |state, phase| {
            let index = state[phase as usize];
            state[phase as usize] += 1;
            Some(offset + index)
        })
        .collect()
}

fn phase_offsets(phases: &[u8]) -> Vec<usize> {
    num_by_phase(phases)
        .into_iter()
        .scan(0, |state, num| {
            let offset = *state;
            *state += num;
            Some(offset)
        })
        .collect()
}

fn convert_expression<F: Field>(
    cs: &ConstraintSystem<F>,
    advice_idx: &[usize],
    challenge_idx: &[usize],
    expression: &plonk::Expression<F>,
) -> Expression<F> {
    expression.evaluate(
        &|constant| Expression::Constant(constant),
        &|selector| {
            Expression::Polynomial(Query::new(
                cs.num_instance_columns() + cs.num_fixed_columns() + selector.index(),
                Rotation::cur(),
            ))
        },
        &|query| {
            Expression::Polynomial(Query::new(
                cs.num_instance_columns() + query.column_index(),
                Rotation(query.rotation().0),
            ))
        },
        &|query| {
            Expression::Polynomial(Query::new(
                advice_idx[query.column_index()],
                Rotation(query.rotation().0),
            ))
        },
        &|query| {
            Expression::Polynomial(Query::new(
                query.column_index(),
                Rotation(query.rotation().0),
            ))
        },
        &|challenge| Expression::Challenge(challenge_idx[challenge.index()]),
        &|value| -value,
        &|lhs, rhs| lhs + rhs,
        &|lhs, rhs| lhs * rhs,
        &|value, scalar| value * scalar,
    )
}

fn row_map(k: usize) -> Vec<usize> {
    BooleanHypercube::new(k).iter().skip(1).collect()
}

fn batch_invert_assigned<F: Field>(assigneds: Vec<Vec<Assigned<F>>>) -> Vec<Vec<F>> {
    let mut denoms: Vec<_> = assigneds
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    denoms
        .iter_mut()
        .flat_map(|f| f.iter_mut().filter_map(|d| d.as_mut()))
        .batch_invert();

    assigneds
        .iter()
        .zip(denoms.into_iter())
        .map(|(assigneds, denoms)| {
            assigneds
                .iter()
                .zip(denoms)
                .map(|(assigned, denom)| {
                    denom
                        .map(|denom| assigned.numerator() * denom)
                        .unwrap_or_else(|| assigned.numerator())
                })
                .collect()
        })
        .collect()
}
