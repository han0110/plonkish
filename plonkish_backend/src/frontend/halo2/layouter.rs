use crate::util::arithmetic::Field;
use halo2_proofs::{
    circuit::{
        layouter::{RegionLayouter, TableLayouter},
        Cell, Layouter, Region, Table, Value,
    },
    plonk::{
        Advice, Any, Assigned, Challenge, Circuit, Column, Error, Fixed, Instance, Selector,
        TableColumn,
    },
};
use std::{collections::HashMap, fmt::Debug};

#[derive(Debug, Default)]
pub struct InstanceExtractor<F> {
    advices: HashMap<usize, Vec<Value<F>>>,
    instances: Vec<Vec<F>>,
}

impl<F: Field> InstanceExtractor<F> {
    pub fn extract<C: Circuit<F>>(circuit: &C) -> Result<Vec<Vec<F>>, Error> {
        let mut extractor = Self::default();
        let config = C::configure(&mut Default::default());
        circuit.synthesize(config, &mut extractor)?;
        Ok(extractor.instances)
    }
}

impl<F: Field> RegionLayouter<F> for InstanceExtractor<F> {
    fn enable_selector<'v>(
        &'v mut self,
        _: &'v (dyn Fn() -> String + 'v),
        _: &Selector,
        _: usize,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn name_column<'v>(&'v mut self, _: &'v (dyn Fn() -> String + 'v), _: Column<Any>) {}

    fn assign_advice<'v>(
        &'v mut self,
        _: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        _: usize,
        to: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        let value = to().to_field().evaluate();
        let mut offset = 0;
        self.advices
            .entry(column.index())
            .and_modify(|advices| {
                offset = advices.len();
                advices.push(value);
            })
            .or_insert_with(|| vec![value]);
        Ok((0.into(), offset, column.into()).into())
    }

    fn assign_advice_from_constant<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        _: usize,
        constant: Assigned<F>,
    ) -> Result<Cell, Error> {
        self.assign_advice(annotation, column, 0, &mut || Value::known(constant))
    }

    fn assign_advice_from_instance<'v>(
        &mut self,
        _: &'v (dyn Fn() -> String + 'v),
        _: Column<Instance>,
        _: usize,
        _: Column<Advice>,
        _: usize,
    ) -> Result<(Cell, Value<F>), Error> {
        unreachable!()
    }

    fn assign_fixed<'v>(
        &'v mut self,
        _: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        _: usize,
        _: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        Ok((0.into(), 0, column.into()).into())
    }

    fn constrain_constant(&mut self, _: Cell, _: Assigned<F>) -> Result<(), Error> {
        Ok(())
    }

    fn constrain_equal(&mut self, _: Cell, _: Cell) -> Result<(), Error> {
        Ok(())
    }
}

impl<F: Field> TableLayouter<F> for InstanceExtractor<F> {
    fn assign_cell<'v>(
        &'v mut self,
        _: &'v (dyn Fn() -> String + 'v),
        _: TableColumn,
        _: usize,
        _: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<(), Error> {
        Ok(())
    }
}

impl<F: Field> Layouter<F> for &mut InstanceExtractor<F> {
    type Root = Self;

    fn assign_region<T, AR, N, NR>(&mut self, _: N, mut f: T) -> Result<AR, Error>
    where
        T: FnMut(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let region: &mut dyn RegionLayouter<F> = *self;
        f(region.into())
    }

    fn assign_table<T, N, NR>(&mut self, _: N, mut f: T) -> Result<(), Error>
    where
        T: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        let table: &mut dyn TableLayouter<F> = *self;
        f(table.into())
    }

    fn constrain_instance(
        &mut self,
        cell: Cell,
        column: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        if self.instances.len() < column.index() + 1 {
            self.instances.resize_with(column.index() + 1, Vec::new);
        }
        if self.instances[column.index()].len() < row + 1 {
            self.instances[column.index()].resize(row + 1, F::ZERO);
        }
        assert_eq!(*cell.column().column_type(), Any::advice());
        let value = self.advices[&cell.column().index()][cell.row_offset()];
        value.map(|value| {
            self.instances[column.index()][row] = value;
        });
        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        unreachable!()
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}
