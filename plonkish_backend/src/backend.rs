use crate::{
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::Field,
        expression::Expression,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use rand::RngCore;
use std::{collections::BTreeSet, fmt::Debug, iter};

pub mod hyperplonk;

pub trait PlonkishBackend<F, Pcs>: Clone + Debug
where
    F: Field,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type ProverParam: Debug;
    type VerifierParam: Debug;

    fn setup(size: usize, rng: impl RngCore) -> Result<Pcs::Param, Error>;

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn prove(
        pp: &Self::ProverParam,
        instances: &[&[F]],
        witness_collector: &impl Fn(&[F]) -> Result<Vec<Vec<F>>, Error>,
        transcript: &mut impl TranscriptWrite<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[&[F]],
        transcript: &mut impl TranscriptRead<F, Commitment = Pcs::Commitment>,
        rng: impl RngCore,
    ) -> Result<(), Error>;
}

#[derive(Clone, Debug)]
pub struct PlonkishCircuitInfo<F> {
    /// 2^k is the size of the circuit
    pub k: usize,
    /// Number of instnace value in each instance polynomial.
    pub num_instances: Vec<usize>,
    /// Preprocessed polynomials, which has index starts with offset
    /// `num_instances.len()`.
    pub preprocess_polys: Vec<MultilinearPolynomial<F>>,
    /// Number of witness polynoimal in each phase.
    /// Witness polynomial index starts with offset `num_instances.len()` +
    /// `preprocess_polys.len()`.
    pub num_witness_polys: Vec<usize>,
    /// Number of challenge in each phase.
    pub num_challenges: Vec<usize>,
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

impl<F: Clone> PlonkishCircuitInfo<F> {
    pub fn is_well_formed(&self) -> bool {
        let num_poly = self.num_poly();
        let num_challenges = self.num_challenges.iter().sum::<usize>();
        let polys = iter::empty()
            .chain(self.expressions().flat_map(Expression::used_poly))
            .chain(self.permutation_polys())
            .collect::<BTreeSet<_>>();
        let challenges = iter::empty()
            .chain(self.expressions().flat_map(Expression::used_challenge))
            .collect::<BTreeSet<_>>();
        // Same amount of phases
        self.num_witness_polys.len() == self.num_challenges.len()
            // Polynomial indices are in range
            && (polys.is_empty() || *polys.last().unwrap() < num_poly)
            // Challenge indices are in range
            && (challenges.is_empty() || *challenges.last().unwrap() < num_challenges)
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
        self.num_instances.len()
            + self.preprocess_polys.len()
            + self.num_witness_polys.iter().sum::<usize>()
    }

    pub fn permutation_polys(&self) -> Vec<usize> {
        self.permutations
            .iter()
            .flat_map(|cycle| cycle.iter().map(|(poly, _)| *poly))
            .unique()
            .sorted()
            .collect()
    }

    pub fn expressions(&self) -> impl Iterator<Item = &Expression<F>> {
        iter::empty().chain(self.constraints.iter()).chain(
            self.lookups
                .iter()
                .flat_map(|lookup| lookup.iter().flat_map(|(input, table)| [input, table])),
        )
    }
}
