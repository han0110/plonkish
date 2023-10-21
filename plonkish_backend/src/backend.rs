use crate::{
    pcs::{CommitmentChunk, PolynomialCommitmentScheme},
    util::{
        arithmetic::Field,
        chain,
        expression::Expression,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{collections::BTreeSet, fmt::Debug};

pub mod hyperplonk;
pub mod unihyperplonk;

pub trait PlonkishBackend<F: Field>: Clone + Debug {
    type Pcs: PolynomialCommitmentScheme<F>;
    type ProverParam: Clone + Debug + Serialize + DeserializeOwned;
    type VerifierParam: Clone + Debug + Serialize + DeserializeOwned;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<<Self::Pcs as PolynomialCommitmentScheme<F>>::Param, Error>;

    fn preprocess(
        param: &<Self::Pcs as PolynomialCommitmentScheme<F>>::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error>;

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlonkishCircuitInfo<F> {
    /// 2^k is the size of the circuit
    pub k: usize,
    /// Number of instnace value in each instance polynomial.
    pub num_instances: Vec<usize>,
    /// Preprocessed polynomials, which has index starts with offset
    /// `num_instances.len()`.
    pub preprocess_polys: Vec<Vec<F>>,
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
        let polys = chain![
            self.expressions().flat_map(Expression::used_poly),
            self.permutation_polys(),
        ]
        .collect::<BTreeSet<_>>();
        let challenges = chain![self.expressions().flat_map(Expression::used_challenge)]
            .collect::<BTreeSet<_>>();
        // Same amount of phases
        self.num_witness_polys.len() == self.num_challenges.len()
            // Every phase has some witness polys
            && !self.num_witness_polys.iter().any(|n| *n == 0)
            // Every phase except the last one has some challenges after the witness polys are committed
            && !self.num_challenges[..self.num_challenges.len() - 1].iter().any(|n| *n == 0)
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
        chain![
            &self.constraints,
            chain![&self.lookups]
                .flat_map(|lookup| lookup.iter().flat_map(|(input, table)| [input, table])),
        ]
    }
}

pub trait PlonkishCircuit<F> {
    fn circuit_info_without_preprocess(&self) -> Result<PlonkishCircuitInfo<F>, Error>;

    fn circuit_info(&self) -> Result<PlonkishCircuitInfo<F>, Error>;

    fn instances(&self) -> &[Vec<F>];

    fn synthesize(&self, round: usize, challenges: &[F]) -> Result<Vec<Vec<F>>, Error>;
}

pub trait WitnessEncoding {
    fn row_mapping(k: usize) -> Vec<usize>;
}

#[cfg(any(test, feature = "benchmark"))]
mod mock {
    use crate::{
        backend::{PlonkishCircuit, PlonkishCircuitInfo},
        Error,
    };

    pub(crate) struct MockCircuit<F> {
        instances: Vec<Vec<F>>,
        witnesses: Vec<Vec<F>>,
    }

    impl<F> MockCircuit<F> {
        pub(crate) fn new(instances: Vec<Vec<F>>, witnesses: Vec<Vec<F>>) -> Self {
            Self {
                instances,
                witnesses,
            }
        }
    }

    impl<F: Clone> PlonkishCircuit<F> for MockCircuit<F> {
        fn circuit_info_without_preprocess(&self) -> Result<PlonkishCircuitInfo<F>, Error> {
            unreachable!()
        }

        fn circuit_info(&self) -> Result<PlonkishCircuitInfo<F>, Error> {
            unreachable!()
        }

        fn instances(&self) -> &[Vec<F>] {
            &self.instances
        }

        fn synthesize(&self, round: usize, challenges: &[F]) -> Result<Vec<Vec<F>>, Error> {
            assert!(round == 0 && challenges.is_empty());
            Ok(self.witnesses.clone())
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::{PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo},
        pcs::PolynomialCommitmentScheme,
        util::{
            arithmetic::PrimeField,
            end_timer, start_timer,
            test::seeded_std_rng,
            transcript::{InMemoryTranscript, TranscriptRead, TranscriptWrite},
            DeserializeOwned, Serialize,
        },
    };
    use std::{hash::Hash, ops::Range};

    pub fn run_plonkish_backend<F, Pb, T, C>(
        num_vars_range: Range<usize>,
        circuit_fn: impl Fn(usize) -> (PlonkishCircuitInfo<F>, C),
    ) where
        F: PrimeField + Hash + Serialize + DeserializeOwned,
        Pb: PlonkishBackend<F>,
        T: TranscriptRead<<Pb::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
            + TranscriptWrite<<Pb::Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
        C: PlonkishCircuit<F>,
    {
        for num_vars in num_vars_range {
            let (circuit_info, circuit) = circuit_fn(num_vars);
            let instances = circuit.instances();

            let timer = start_timer(|| format!("setup-{num_vars}"));
            let param = Pb::setup(&circuit_info, seeded_std_rng()).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("preprocess-{num_vars}"));
            let (pp, vp) = Pb::preprocess(&param, &circuit_info).unwrap();
            end_timer(timer);

            let timer = start_timer(|| format!("prove-{num_vars}"));
            let proof = {
                let mut transcript = T::new(());
                Pb::prove(&pp, &circuit, &mut transcript, seeded_std_rng()).unwrap();
                transcript.into_proof()
            };
            end_timer(timer);

            let timer = start_timer(|| format!("verify-{num_vars}"));
            let result = {
                let mut transcript = T::from_proof((), proof.as_slice());
                Pb::verify(&vp, instances, &mut transcript, seeded_std_rng())
            };
            assert_eq!(result, Ok(()));
            end_timer(timer);
        }
    }
}
