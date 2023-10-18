use crate::{
    piop::sum_check::{SumCheck, VirtualPolynomial},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{Field, PrimeField},
        end_timer,
        expression::{rotate::Rotatable, Expression, Query, Rotation},
        izip, izip_eq,
        parallel::par_map_collect,
        start_timer,
        transcript::{FieldTranscriptRead, FieldTranscriptWrite},
        Itertools,
    },
    Error,
};
use num_integer::Integer;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    marker::PhantomData,
};

mod coeff;
mod eval;

pub use coeff::CoefficientsProver;
pub use eval::EvaluationsProver;

#[derive(Debug)]
pub struct ProverState<'a, F: Field> {
    num_vars: usize,
    expression: &'a Expression<F>,
    degree: usize,
    sum: F,
    lagranges: HashMap<i32, (usize, F)>,
    identity: F,
    eq_xys: Vec<MultilinearPolynomial<F>>,
    polys: Vec<Vec<Cow<'a, MultilinearPolynomial<F>>>>,
    challenges: &'a [F],
    buf: MultilinearPolynomial<F>,
    round: usize,
    rotatable: Box<dyn Rotatable>,
}

impl<'a, F: PrimeField> ProverState<'a, F> {
    fn new<R: Rotatable + From<usize>>(
        num_vars: usize,
        sum: F,
        virtual_poly: VirtualPolynomial<'a, F>,
    ) -> Self {
        let rotatable = Box::new(R::from(num_vars));
        assert!(virtual_poly.expression.max_used_rotation_distance() <= rotatable.max_rotation());

        let lagranges = {
            virtual_poly
                .expression
                .used_langrange()
                .into_iter()
                .map(|i| (i, (rotatable.nth(i), F::ONE)))
                .collect()
        };
        let eq_xys = virtual_poly
            .ys
            .iter()
            .map(|y| MultilinearPolynomial::eq_xy(y))
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
            degree: virtual_poly.expression.degree(),
            sum,
            lagranges,
            identity: F::ZERO,
            eq_xys,
            polys,
            challenges: virtual_poly.challenges,
            buf: MultilinearPolynomial::new(vec![F::ZERO; 1 << (num_vars - 1)]),
            round: 0,
            rotatable,
        }
    }

    fn size(&self) -> usize {
        1 << (self.num_vars - self.round - 1)
    }

    fn next_round<R: Rotatable + From<usize>>(&mut self, sum: F, challenge: &F) {
        self.sum = sum;
        self.identity += F::from(1 << self.round) * challenge;
        self.lagranges.values_mut().for_each(|(b, value)| {
            if b.is_even() {
                *value *= &(F::ONE - challenge);
            } else {
                *value *= challenge;
            }
            *b >>= 1;
        });
        self.eq_xys
            .iter_mut()
            .for_each(|eq_xy| eq_xy.fix_var_in_place(challenge, &mut self.buf));
        if self.round == 0 {
            let rotation_maps = self
                .expression
                .used_rotation()
                .into_iter()
                .filter(|rotation| rotation != &Rotation::cur())
                .map(|rotation| (rotation, self.rotatable.rotation_map(rotation)))
                .collect::<HashMap<_, _>>();
            for query in self.expression.used_query() {
                if query.rotation() != Rotation::cur() {
                    let poly = &self.polys[query.poly()][self.num_vars];
                    let mut rotated = MultilinearPolynomial::new(par_map_collect(
                        &rotation_maps[&query.rotation()],
                        |b| poly[*b],
                    ));
                    rotated.fix_var_in_place(challenge, &mut self.buf);
                    self.polys[query.poly()]
                        [(query.rotation().0 + self.num_vars as i32) as usize] =
                        Cow::Owned(rotated);
                }
            }
            self.polys.iter_mut().for_each(|polys| {
                polys[self.num_vars] = Cow::Owned(polys[self.num_vars].fix_var(challenge));
            });
        } else {
            self.polys.iter_mut().for_each(|polys| {
                polys.iter_mut().for_each(|poly| {
                    if !poly.is_empty() {
                        poly.to_mut().fix_var_in_place(challenge, &mut self.buf);
                    }
                });
            });
        }
        self.round += 1;
        if self.round != self.num_vars {
            self.rotatable = Box::new(R::from(self.num_vars - self.round));
        }
    }

    // TODO: Track polys by BTreeMap already
    fn into_evals(self) -> BTreeMap<Query, F> {
        assert_eq!(self.round, self.num_vars);
        izip!(0.., self.polys)
            .flat_map(|(idx, polys)| {
                izip_eq!(-(self.num_vars as i32)..self.num_vars as i32, polys).flat_map(
                    move |(rotation, poly)| {
                        (!poly.is_empty()).then(|| (Query::new(idx, Rotation(rotation)), poly[0]))
                    },
                )
            })
            .collect()
    }
}

pub trait ClassicSumCheckProver<F: Field>: Clone + Debug {
    type RoundMessage: ClassicSumCheckRoundMessage<F>;

    fn new(state: &ProverState<F>) -> Self;

    fn prove_round(&self, state: &ProverState<F>) -> Self::RoundMessage;
}

pub trait ClassicSumCheckRoundMessage<F: Field>: Sized + Debug {
    type Auxiliary: Default;

    fn write(&self, transcript: &mut impl FieldTranscriptWrite<F>) -> Result<(), Error>;

    fn read(degree: usize, transcript: &mut impl FieldTranscriptRead<F>) -> Result<Self, Error>;

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

#[derive(Debug)]
pub struct ClassicSumCheck<P, R = usize>(PhantomData<(P, R)>);

impl<P, R> Clone for ClassicSumCheck<P, R> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<F, P, R> SumCheck<F> for ClassicSumCheck<P, R>
where
    F: PrimeField,
    P: ClassicSumCheckProver<F>,
    R: Rotatable + From<usize>,
{
    type ProverParam = ();
    type VerifierParam = ();

    fn prove(
        _: &Self::ProverParam,
        num_vars: usize,
        virtual_poly: VirtualPolynomial<F>,
        sum: F,
        transcript: &mut impl FieldTranscriptWrite<F>,
    ) -> Result<(F, Vec<F>, BTreeMap<Query, F>), Error> {
        let _timer = start_timer(|| {
            let degree = virtual_poly.expression.degree();
            format!("sum_check_prove-{num_vars}-{degree}")
        });

        let mut state = ProverState::new::<R>(num_vars, sum, virtual_poly);
        let mut challenges = Vec::with_capacity(num_vars);
        let prover = P::new(&state);
        let aux = P::RoundMessage::auxiliary(state.degree);

        for round in 0..num_vars {
            let timer = start_timer(|| format!("sum_check_prove_round-{round}"));
            let msg = prover.prove_round(&state);
            end_timer(timer);
            msg.write(transcript)?;

            let challenge = transcript.squeeze_challenge();
            challenges.push(challenge);

            let timer = start_timer(|| format!("sum_check_next_round-{round}"));
            state.next_round::<R>(msg.evaluate(&aux, &challenge), &challenge);
            end_timer(timer);
        }

        Ok((state.sum, challenges, state.into_evals()))
    }

    fn verify(
        _: &Self::VerifierParam,
        num_vars: usize,
        degree: usize,
        sum: F,
        transcript: &mut impl FieldTranscriptRead<F>,
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
