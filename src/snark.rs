use crate::{
    pcs::PolynomialCommitmentScheme,
    util::{
        arithmetic::Field,
        expression::{Expression, Query},
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{collections::BTreeSet, fmt::Debug, iter};

pub mod hyperplonk;

#[derive(Clone, Debug)]
pub struct CircuitInfo<F> {
    /// 2^k is the size of the circuit
    pub k: usize,
    /// Number of instnace value in each instance polynomial.
    pub num_instance: Vec<usize>,
    /// Number of preprocessed polynoimal.
    /// Preprocessed polynomial index starts with offset `num_instance.len()`.
    pub num_preprocessed_poly: usize,
    /// Number of witness polynoimal in each phase.
    /// Witness polynomial index starts with offset `num_instance.len()` +
    /// `num_preprocessed_poly`.
    pub num_witness_poly: Vec<usize>,
    /// Number of challenge in each phase.
    pub num_challenge: Vec<usize>,
    /// Constraints.
    pub constraints: Vec<Expression<F>>,
    /// Each item inside outer vector repesents an independent vector lookup,
    /// which contains vector of tuples representing the input and table
    /// respectively.
    pub lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    /// Each item inside outer vector repesents an closed permutation cycle,
    /// which contains vetor of tuples representing the polynomial index and
    /// row respectively.
    pub permutations: Vec<Vec<(usize, usize)>>,
    /// Maximum degree of constraints
    pub max_degree: Option<usize>,
}

impl<F: Clone> CircuitInfo<F> {
    pub fn is_well_formed(&self) -> bool {
        let poly_range = 0..self.num_poly();
        let polys = iter::empty()
            .chain(self.used_query().iter().map(Query::poly))
            .chain(self.permutation_polys())
            .collect::<BTreeSet<_>>();
        // Same amount of phases
        self.num_witness_poly.len() == self.num_challenge.len()
            // Polynomial indices are in range
            && (polys.is_empty()
                || (poly_range.contains(polys.first().unwrap())
                    && poly_range.contains(polys.last().unwrap())))
            // Every constraint has degree less equal than `max_degree`
            && self
                .max_degree
                .map(|max_degree| {
                    !self
                        .constraints
                        .iter()
                        .any(|constraint| constraint.degree() > max_degree)
                })
                .unwrap_or(true)
    }

    pub fn num_poly(&self) -> usize {
        self.num_instance.len()
            + self.num_preprocessed_poly
            + self.num_witness_poly.iter().sum::<usize>()
    }

    pub fn permutation_polys(&self) -> Vec<usize> {
        self.permutations
            .iter()
            .flat_map(|cycle| cycle.iter().map(|(poly, _)| *poly))
            .unique()
            .sorted()
            .collect()
    }

    pub fn used_query(&self) -> BTreeSet<Query> {
        iter::empty()
            .chain(
                self.constraints
                    .iter()
                    .flat_map(|gate| gate.used_query().into_iter()),
            )
            .chain(self.lookups.iter().flat_map(|lookup| {
                lookup.iter().flat_map(|(input, table)| {
                    input.used_query().into_iter().chain(table.used_query())
                })
            }))
            .collect()
    }
}

trait UniversalSnark<F, Pcs>
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type ProverParam: Debug;
    type VerifierParam: Debug;

    fn setup(size: usize, rng: impl RngCore) -> Result<Pcs::Param, Error>;

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &CircuitInfo<F>,
        preprocessed: Vec<Vec<F>>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn prove(
        pp: &Self::ProverParam,
        instance: &[&[F]],
        witness_collector: impl Fn(&[F]) -> Vec<Vec<F>>,
        transcript: &mut impl TranscriptWrite<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        instance: &[&[F]],
        transcript: &mut impl TranscriptRead<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;
}
