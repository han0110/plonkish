use crate::util::{arithmetic::Field, expression::Expression};

pub mod hyperplonk;

pub struct CircuitInfo<F: Field> {
    /// Number of preprocessed polynoimal.
    pub num_preprocessed_poly: usize,
    /// Number of witness polynoimal in each phase.
    /// Witness polynomial index starts with offset `num_preprocessed_poly`.
    pub num_witness_poly: Vec<usize>,
    /// Number of instnace value in each instance polynomial.
    /// Instance polynomial index starts with offset `num_preprocessed_poly`
    /// plus sum of `num_witness_poly`.
    pub num_instance: Vec<usize>,
    /// Number of challenge in each phase.
    pub num_challenge: Vec<usize>,
    /// Gates.
    pub gates: Vec<Expression<F>>,
    /// Each item inside outer vector repesents an independent vector lookup,
    /// which contains vector of tuples representing the input and table
    /// respectively.
    pub lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    /// Each item inside outer vector repesents an closed permutation cycle,
    /// which contains vetor of tuples representing the polynomial index and
    /// row respectively.
    pub permutations: Vec<Vec<(usize, usize)>>,
}
