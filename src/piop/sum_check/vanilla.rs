use crate::{
    piop::sum_check::{SumCheck, VirtualPolynomial},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{BooleanHypercube, Field, PrimeField},
        expression::{Expression, Query, Rotation},
        parallel::par_map_collect,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use num_integer::Integer;
use std::{borrow::Cow, collections::HashMap, fmt::Debug, marker::PhantomData};

mod coeff;
mod eval;

pub use coeff::CoefficientsProver;
pub use eval::EvaluationsProver;

#[derive(Debug)]
pub struct ProverState<'a, F: Field> {
    num_vars: usize,
    expression: &'a Expression<F>,
    challenges: &'a [F],
    polys: Vec<Vec<Cow<'a, MultilinearPolynomial<F>>>>,
    eq_xys: Vec<MultilinearPolynomial<F>>,
    lagranges: HashMap<i32, (usize, F)>,
    identities: Vec<F>,
    round: usize,
    bh: BooleanHypercube,
}

impl<'a, F: PrimeField> ProverState<'a, F> {
    fn new(num_vars: usize, virtual_poly: VirtualPolynomial<'a, F>) -> Self {
        assert!(num_vars > 0 && virtual_poly.expression.max_used_rotation_distance() <= num_vars);
        let bh = BooleanHypercube::new(num_vars);
        let lagranges = {
            let bh = bh.iter().collect_vec();
            virtual_poly
                .expression
                .used_langrange()
                .into_iter()
                .map(|i| {
                    let b = bh[i.rem_euclid(1 << num_vars) as usize];
                    (i, (b, F::one()))
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
                virtual_poly
                    .expression
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
            expression: virtual_poly.expression,
            challenges: virtual_poly.challenges,
            lagranges,
            eq_xys,
            identities,
            polys,
            round: 0,
            bh,
        }
    }

    fn size(&self) -> usize {
        1 << (self.num_vars - self.round - 1)
    }

    fn next_round(&mut self, challenge: &F) {
        self.lagranges.values_mut().for_each(|(b, value)| {
            if b.is_even() {
                *value *= &(F::one() - challenge);
            } else {
                *value *= challenge;
            }
            *b >>= 1;
        });
        self.eq_xys
            .iter_mut()
            .for_each(|eq_xy| *eq_xy = eq_xy.fix_variables(&[*challenge]));
        self.identities
            .iter_mut()
            .for_each(|constant| *constant += F::from(1 << self.round) * challenge);
        if self.round == 0 {
            let rotation_maps = self
                .expression
                .used_rotation()
                .into_iter()
                .filter_map(|rotation| {
                    (rotation != Rotation::cur()).then(|| {
                        (
                            rotation,
                            BooleanHypercube::new(self.num_vars).rotation_map(rotation),
                        )
                    })
                })
                .collect::<HashMap<_, _>>();
            for query in self.expression.used_query() {
                if query.rotation() != Rotation::cur() {
                    let poly = &self.polys[query.poly()][self.num_vars];
                    let rotated = MultilinearPolynomial::new(par_map_collect(
                        &rotation_maps[&query.rotation()],
                        |b| poly[*b],
                    ));
                    self.polys[query.poly()]
                        [(query.rotation().0 + self.num_vars as i32) as usize] =
                        Cow::Owned(rotated.fix_variables(&[*challenge]));
                }
            }
            self.polys.iter_mut().for_each(|polys| {
                polys[self.num_vars] =
                    Cow::Owned(polys[self.num_vars].fix_variables(&[*challenge]));
            });
        } else {
            self.polys.iter_mut().for_each(|polys| {
                polys.iter_mut().for_each(|poly| {
                    if !poly.is_zero() {
                        *poly = Cow::Owned(poly.fix_variables(&[*challenge]));
                    }
                });
            });
        }
        self.round += 1;
        self.bh = BooleanHypercube::new(self.num_vars - self.round);
    }

    fn poly<const IS_FIRST_ROUND: bool>(&self, query: &Query) -> &MultilinearPolynomial<F> {
        if IS_FIRST_ROUND {
            &self.polys[query.poly()][self.num_vars]
        } else {
            &self.polys[query.poly()][(query.rotation().0 + self.num_vars as i32) as usize]
        }
    }
}

pub trait VanillaSumCheckProver<F: Field>: Clone + Debug {
    type RoundMessage: VanillaSumCheckRoundMessage<F>;

    fn new(state: &ProverState<F>) -> Self;

    fn prove_round<'a>(
        &self,
        state: &mut ProverState<'a, F>,
        challenge: Option<&F>,
    ) -> Self::RoundMessage;
}

pub trait VanillaSumCheckRoundMessage<F: Field>: Sized + Debug {
    type Auxiliary: Default;

    fn write(&self, transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error>;

    fn read(degree: usize, transcript: &mut impl TranscriptRead<F>) -> Result<Self, Error>;

    fn sum(&self) -> F;

    fn auxiliary(_degree: usize) -> Self::Auxiliary {
        Default::default()
    }

    fn evaluate(&self, aux: &Self::Auxiliary, challenge: &F) -> F;

    fn verify_consistency(
        degree: usize,
        mut sum: F,
        msgs: &[Self],
        challenges: &[F],
    ) -> Result<F, Error> {
        let aux = Self::auxiliary(degree);
        for (round, (msg, challenge)) in msgs.iter().zip(challenges.iter()).enumerate() {
            if sum != msg.sum() {
                let msg = if round == 0 {
                    format!("Expect sum {sum:?} but get {:?}", msg.sum())
                } else {
                    format!("Consistency failure at round {round}")
                };
                return Err(Error::InvalidSumcheck(msg));
            }
            sum = msg.evaluate(&aux, challenge);
        }
        Ok(sum)
    }
}

#[derive(Clone, Debug)]
pub struct VanillaSumCheck<P>(PhantomData<P>);

impl<F, P> SumCheck<F> for VanillaSumCheck<P>
where
    F: PrimeField,
    P: VanillaSumCheckProver<F>,
{
    type ProverParam = ();
    type VerifierParam = ();

    fn prove(
        _: &Self::ProverParam,
        num_vars: usize,
        virtual_poly: VirtualPolynomial<F>,
        transcript: &mut impl TranscriptWrite<F>,
    ) -> Result<Vec<F>, Error> {
        let _timer = start_timer(|| {
            let degree = virtual_poly.expression.degree();
            format!("sum_check_prove-{num_vars}-{degree}")
        });

        let mut state = ProverState::new(num_vars, virtual_poly);
        let mut challenges = Vec::with_capacity(num_vars);
        let prover = P::new(&state);

        for _ in 0..num_vars {
            let msg = prover.prove_round(&mut state, challenges.last());
            msg.write(transcript)?;
            challenges.push(transcript.squeeze_challenge());
        }

        Ok(challenges)
    }

    fn verify(
        _: &Self::VerifierParam,
        num_vars: usize,
        degree: usize,
        sum: F,
        transcript: &mut impl TranscriptRead<F>,
    ) -> Result<(F, Vec<F>), Error> {
        let (msgs, challenges) = {
            let mut msgs = Vec::with_capacity(num_vars);
            let mut challenges = Vec::with_capacity(num_vars);
            for _ in 0..num_vars {
                msgs.push(P::RoundMessage::read(degree, transcript)?);
                challenges.push(transcript.squeeze_challenge());
            }
            (msgs, challenges)
        };

        Ok((
            P::RoundMessage::verify_consistency(degree, sum, &msgs, &challenges)?,
            challenges,
        ))
    }
}
