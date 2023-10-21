use crate::{
    pcs::{Additive, Evaluation, PolynomialCommitmentScheme},
    piop::multilinear_eval::ph23::{additive::QueryGroup::*, s_evals, vanishing_eval},
    poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
    util::{
        arithmetic::{
            inner_product, powers, product, BatchInvert, Msm, PrimeField, WithSmallOrderMulGroup,
        },
        chain, end_timer,
        expression::{
            evaluator::quotient::{QuotientEvaluator, Radix2Domain},
            rotate::{Lexical, Rotatable},
            Expression, Query, Rotation,
        },
        izip, izip_eq,
        parallel::parallelize,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use std::{borrow::Cow, collections::BTreeMap, mem};

#[allow(clippy::too_many_arguments)]
pub fn prove_multilinear_eval<'a, F, Pcs>(
    pp: &Pcs::ProverParam,
    num_vars: usize,
    s_polys: &[Vec<F>],
    polys: impl IntoIterator<Item = &'a UnivariatePolynomial<F>>,
    comms: impl IntoIterator<Item = &'a Pcs::Commitment>,
    point: &[F],
    evals: &[(Query, F)],
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> Result<(), Error>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
    Pcs::Commitment: 'a + Additive<F>,
{
    let domain = &Radix2Domain::<F>::new(num_vars, 2);
    let polys = polys.into_iter().collect_vec();
    let comms = comms.into_iter().collect_vec();
    let num_polys = polys.len();
    assert_eq!(comms.len(), num_polys);

    let (queries, evals) = evals.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();

    let gamma = transcript.squeeze_challenge();
    let powers_of_gamma = powers(gamma).take(queries.len()).collect_vec();

    let query_groups = query_groups(num_vars, &queries, &powers_of_gamma);

    let u_step = -domain.n_inv() * inner_product(&powers_of_gamma, &evals);
    let (eq_0, eq_u_fs) = {
        let fs = chain![&query_groups]
            .map(|group| group.poly(&polys))
            .collect_vec();
        let (eq, u) = eq_u(point, &query_groups, &fs, u_step);

        let eq_0 = eq[0];
        let eq_u_fs = chain![[eq, u].map(Cow::Owned), fs]
            .map(|buf| domain.lagrange_to_monomial(buf))
            .map(UnivariatePolynomial::monomial)
            .collect_vec();
        (eq_0, eq_u_fs)
    };

    let eq_u_comm = Pcs::batch_commit_and_write(pp, &eq_u_fs[..2], transcript)?;

    let alpha = transcript.squeeze_challenge();

    let expression = expression(&query_groups, point, u_step, eq_0, alpha);

    let q = {
        let eq_u_fs = chain![&eq_u_fs]
            .map(|poly| domain.monomial_to_extended_lagrange(poly.coeffs().into()))
            .collect_vec();
        let polys = chain![&eq_u_fs, s_polys].map(Vec::as_slice);

        let timer = start_timer(|| "quotient");
        let ev = QuotientEvaluator::new(domain, &expression, Default::default(), polys);
        let mut q = vec![F::ZERO; domain.extended_n()];
        parallelize(&mut q, |(q, start)| {
            let mut cache = ev.cache();
            izip!(q, start..).for_each(|(q, row)| ev.evaluate(q, &mut cache, row));
        });
        end_timer(timer);

        UnivariatePolynomial::monomial(domain.extended_lagrange_to_monomial(q.into()))
    };

    let q_comm = Pcs::commit_and_write(pp, &q, transcript)?;

    let x = transcript.squeeze_challenge();

    let evals = eq_u_queries(&expression)
        .map(|query| {
            let point = domain.rotate_point(x, query.rotation());
            (query, eq_u_fs[query.poly()].evaluate(&point))
        })
        .collect_vec();

    transcript.write_field_elements(evals.iter().map(|(_, eval)| eval))?;

    let (lin, lin_comm, lin_eval) = {
        let evals = chain![evals.iter().cloned(), s_evals(domain, eq_u_fs.len(), x)].collect();
        let vanishing_eval = vanishing_eval(domain, x);
        let (constant, poly) = {
            let q = Msm::term(vanishing_eval, &q);
            linearization(&expression, &eq_u_fs, &evals, q)
        };
        let comm = if cfg!(feature = "sanity-check") {
            let comms = {
                let f_comms = query_groups.iter().map(|group| group.comm(&comms));
                chain![eq_u_comm.iter().map(Msm::base), f_comms].collect_vec()
            };
            let q = Msm::term(vanishing_eval, &q_comm);
            let (_, comm) = linearization(&expression, &comms, &evals, Msm::base(&q));
            let (_, comm) = comm.evaluate();
            comm
        } else {
            Default::default()
        };
        (poly, comm, -constant)
    };

    if cfg!(feature = "sanity-check") {
        assert_eq!(lin.evaluate(&x), lin_eval);
        assert_eq!(&Pcs::commit(pp, &lin).unwrap(), &lin_comm);
    }

    let polys = chain![&eq_u_fs[..2], [&lin]];
    let comms = chain![&eq_u_comm, [&lin_comm]];
    let (points, evals) = points_evals(domain, x, &evals, lin_eval);
    let _timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
    Pcs::batch_open(pp, polys, comms, &points, &evals, transcript)
}

pub fn verify_multilinear_eval<'a, F, Pcs>(
    vp: &Pcs::VerifierParam,
    num_vars: usize,
    comms: impl IntoIterator<Item = &'a Pcs::Commitment>,
    point: &[F],
    evals: &[(Query, F)],
    transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
) -> Result<(), Error>
where
    F: WithSmallOrderMulGroup<3>,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
    Pcs::Commitment: 'a + Additive<F>,
{
    let domain = &Radix2Domain::<F>::new(num_vars, 2);
    let comms = comms.into_iter().collect_vec();

    let (queries, evals) = evals.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();

    let gamma = transcript.squeeze_challenge();
    let powers_of_gamma = powers(gamma).take(evals.len()).collect_vec();

    let query_groups = query_groups(num_vars, &queries, &powers_of_gamma);

    let u_step = -domain.n_inv() * inner_product(&powers_of_gamma, &evals);
    let eq_0 = product(point.iter().map(|point_i| F::ONE - point_i));

    let eq_u_comm = Pcs::read_commitments(vp, 2, transcript)?;

    let alpha = transcript.squeeze_challenge();

    let expression = expression(&query_groups, point, u_step, eq_0, alpha);

    let q_comm = Pcs::read_commitment(vp, transcript)?;

    let x = transcript.squeeze_challenge();

    let evals = {
        let queries = eq_u_queries(&expression).collect_vec();
        let evals = transcript.read_field_elements(queries.len())?;
        izip_eq!(queries, evals).collect_vec()
    };

    let (lin_comm, lin_eval) = {
        let comms = {
            let f_comms = query_groups.iter().map(|group| group.comm(&comms));
            chain![eq_u_comm.iter().map(Msm::base), f_comms].collect_vec()
        };
        let evals = chain![evals.iter().cloned(), s_evals(domain, comms.len(), x)].collect();
        let vanishing_eval = vanishing_eval(domain, x);
        let q = Msm::term(vanishing_eval, &q_comm);
        let (constant, comm) = linearization(&expression, &comms, &evals, Msm::base(&q));
        let (_, comm) = comm.evaluate();
        (comm, -constant)
    };

    let comms = chain![&eq_u_comm, [&lin_comm]];
    let (points, evals) = points_evals(domain, x, &evals, lin_eval);
    Pcs::batch_verify(vp, comms, &points, &evals, transcript)
}

#[derive(Clone, Debug)]
enum QueryGroup<F> {
    ByPoly {
        poly: usize,
        rotations: Vec<Rotation>,
        scalars: Vec<F>,
    },
    ByRotation {
        rotation: Rotation,
        polys: Vec<usize>,
        scalars: Vec<F>,
    },
}

impl<F: PrimeField> QueryGroup<F> {
    fn eq(&self) -> Expression<F> {
        match self {
            ByPoly {
                rotations, scalars, ..
            } => {
                let eq_rots =
                    chain![rotations].map(|rotation| Expression::Polynomial((0, *rotation).into()));
                izip!(eq_rots, scalars)
                    .map(|(eq_rot, scalar)| eq_rot * *scalar)
                    .sum::<Expression<_>>()
            }
            ByRotation { rotation, .. } => Expression::<F>::Polynomial((0, *rotation).into()),
        }
    }

    fn poly<'a>(&self, polys: &[&'a UnivariatePolynomial<F>]) -> Cow<'a, [F]> {
        match self {
            ByPoly { poly, .. } => polys[*poly].coeffs().into(),
            ByRotation {
                polys: ps, scalars, ..
            } => izip!(scalars, ps)
                .map(|(scalar, poly)| (scalar, polys[*poly]))
                .sum::<UnivariatePolynomial<_>>()
                .into_coeffs()
                .into(),
        }
    }

    fn comm<'a, T: Additive<F>>(&self, comms: &[&'a T]) -> Msm<'a, F, T> {
        match self {
            ByPoly { poly, .. } => Msm::base(comms[*poly]),
            ByRotation { polys, scalars, .. } => izip!(polys, scalars)
                .map(|(poly, scalar)| Msm::term(*scalar, comms[*poly]))
                .sum(),
        }
    }
}

fn query_groups<F: PrimeField>(
    num_vars: usize,
    queries: &[Query],
    powers_of_gamma: &[F],
) -> Vec<QueryGroup<F>> {
    let n = 1 << num_vars;
    let repeated_rotations = chain![queries.iter().map(|query| (-query.rotation()).positive(n))]
        .counts()
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .sorted_by(|a, b| b.1.cmp(&a.1))
        .map(|(rotation, _)| rotation);
    let mut by_polys = izip!(queries, powers_of_gamma)
        .fold(BTreeMap::new(), |mut polys, (query, scalar)| {
            polys
                .entry(query.poly())
                .and_modify(|poly| match poly {
                    ByPoly {
                        rotations, scalars, ..
                    } => {
                        rotations.push((-query.rotation()).positive(n));
                        scalars.push(*scalar);
                    }
                    _ => unreachable!(),
                })
                .or_insert_with(|| ByPoly {
                    poly: query.poly(),
                    rotations: vec![(-query.rotation()).positive(n)],
                    scalars: vec![*scalar],
                });
            polys
        })
        .into_values()
        .collect_vec();
    let mut by_rotations = Vec::new();
    let mut output = by_polys.clone();
    for rotation in repeated_rotations {
        let mut by_rotation = (Vec::new(), Vec::new());
        by_polys.retain_mut(|poly| match poly {
            ByPoly {
                poly,
                rotations,
                scalars,
            } => {
                if let Some(idx) = rotations.iter().position(|value| *value == rotation) {
                    rotations.remove(idx);
                    by_rotation.0.push(*poly);
                    by_rotation.1.push(scalars.remove(idx));
                    !rotations.is_empty()
                } else {
                    true
                }
            }
            _ => unreachable!(),
        });
        by_rotations.push(ByRotation {
            rotation,
            polys: by_rotation.0,
            scalars: by_rotation.1,
        });
        if by_polys.len() + by_rotations.len() <= output.len() {
            output = chain![&by_polys, &by_rotations].cloned().collect_vec();
        }
    }
    output
}

fn eq_u<F: PrimeField>(
    point: &[F],
    query_groups: &[QueryGroup<F>],
    polys: &[Cow<[F]>],
    u_step: F,
) -> (Vec<F>, Vec<F>) {
    let _timer = start_timer(|| "u");

    let lexical = Lexical::new(point.len());
    let eq = MultilinearPolynomial::eq_xy(point).into_evals();
    let sums = {
        let mut coeffs = vec![F::ZERO; lexical.n()];
        izip!(query_groups, polys).for_each(|(group, poly)| match group {
            ByPoly {
                rotations, scalars, ..
            } => {
                parallelize(&mut coeffs, |(coeffs, start)| {
                    izip!(start.., coeffs, &poly[start..]).for_each(|(idx, coeffs, poly)| {
                        let eq_rot =
                            chain![rotations].map(|rotation| &eq[lexical.rotate(idx, *rotation)]);
                        *coeffs += inner_product(eq_rot, scalars) * poly;
                    });
                });
            }
            ByRotation { rotation, .. } => {
                parallelize(&mut coeffs, |(coeffs, start)| {
                    let skip = lexical.rotate(start, *rotation);
                    izip!(coeffs, eq.iter().cycle().skip(skip), &poly[start..])
                        .for_each(|(coeffs, eq, poly)| *coeffs += *eq * poly);
                });
            }
        });
        coeffs
    };
    let u = chain![&sums]
        .scan(F::ZERO, |u, sum| mem::replace(u, *u + sum + u_step).into())
        .collect_vec();

    if cfg!(feature = "sanity-check") {
        assert_eq!(F::ZERO, u[lexical.nth(-1)] + sums[lexical.nth(-1)] + u_step);
    }

    (eq, u)
}

fn expression<F: PrimeField>(
    query_groups: &[QueryGroup<F>],
    point: &[F],
    u_step: F,
    eq_0: F,
    alpha: F,
) -> Expression<F> {
    let num_vars = point.len();
    let [u_step, eq_0, alpha] = &[u_step, eq_0, alpha].map(Expression::Constant);
    let eq_ratios = {
        let mut denoms = point.iter().map(|point_i| F::ONE - point_i).collect_vec();
        denoms.batch_invert();
        izip!(point, denoms)
            .map(|(numer, denom)| denom * numer)
            .rev()
            .collect_vec()
    };
    let eq = &Expression::Polynomial(Query::new(0, Rotation::cur()));
    let eq_rots = (0..num_vars)
        .rev()
        .map(|rotation| Expression::Polynomial(Query::new(0, Rotation(1 << rotation))))
        .collect_vec();
    let [u, u_next] = &[Rotation::cur(), Rotation::next()]
        .map(|rotation| Expression::Polynomial(Query::new(1, rotation)));
    let f = izip!(2.., query_groups)
        .map(|(idx, set)| set.eq() * Expression::Polynomial((idx, 0).into()))
        .sum::<Expression<_>>();
    let s = (2 + query_groups.len()..)
        .take(num_vars)
        .map(|poly| Expression::<F>::Polynomial(Query::new(poly, Rotation::cur())))
        .collect_vec();
    let constraints = chain![
        [u_next - u - f - u_step],
        [&s[0] * (eq - eq_0)],
        izip!(&s, &eq_rots, &eq_ratios).map(|(s, eq_rot, eq_ratio)| s * (eq * eq_ratio - eq_rot))
    ]
    .collect_vec();
    Expression::distribute_powers(&constraints, alpha)
        .simplified(None)
        .unwrap()
}

fn eq_u_queries<F: PrimeField>(expression: &Expression<F>) -> impl Iterator<Item = Query> {
    chain![
        chain![expression.used_query()].filter(|query| query.poly() == 0),
        [(1, Rotation::next()).into()]
    ]
}

fn linearization<'a, F: PrimeField, T: Additive<F> + 'a>(
    expression: &Expression<F>,
    bases: impl IntoIterator<Item = &'a T>,
    evals: &BTreeMap<Query, F>,
    vanishing_q: Msm<F, T>,
) -> (F, T) {
    let bases = bases.into_iter().collect_vec();
    (expression.evaluate(
        &|scalar| Msm::scalar(scalar),
        &|_| unreachable!(),
        &|query| {
            if let Some(eval) = evals.get(&query) {
                Msm::scalar(*eval)
            } else {
                assert_eq!(query.rotation(), Rotation::cur());
                Msm::base(bases[query.poly()])
            }
        },
        &|_| unreachable!(),
        &|scalar| -scalar,
        &|lhs, rhs| lhs + rhs,
        &|lhs, rhs| lhs * rhs,
        &|value, scalar| value * Msm::scalar(scalar),
    ) - vanishing_q)
        .evaluate()
}

fn points_evals<F: WithSmallOrderMulGroup<3>>(
    domain: &Radix2Domain<F>,
    x: F,
    evals: &[(Query, F)],
    lin_eval: F,
) -> (Vec<F>, Vec<Evaluation<F>>) {
    let point_index = evals
        .iter()
        .fold(BTreeMap::new(), |mut point_index, (query, _)| {
            let rotation = query.rotation().positive(domain.n());
            let idx = point_index.len();
            point_index.entry(rotation).or_insert(idx);
            point_index
        });
    let points = point_index
        .iter()
        .sorted_by(|a, b| a.1.cmp(b.1))
        .map(|(rotation, _)| domain.rotate_point(x, *rotation))
        .collect_vec();
    let evals = chain![
        evals.iter().map(|(query, eval)| {
            let point = point_index[&query.rotation().positive(domain.n())];
            Evaluation::new(query.poly(), point, *eval)
        }),
        [Evaluation::new(2, point_index[&Rotation::cur()], lin_eval)]
    ]
    .collect_vec();
    (points, evals)
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{univariate::UnivariateKzg, Additive, PolynomialCommitmentScheme},
        piop::multilinear_eval::ph23::{
            additive::{prove_multilinear_eval, verify_multilinear_eval},
            s_polys,
        },
        poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
        util::{
            arithmetic::WithSmallOrderMulGroup,
            expression::{Query, Rotation},
            izip,
            test::{rand_vec, seeded_std_rng},
            transcript::{
                InMemoryTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite,
            },
            Itertools,
        },
    };
    use halo2_curves::bn256::{Bn256, Fr};
    use rand::Rng;
    use std::{io::Cursor, iter};

    fn run_prove_verify<F, Pcs>(num_vars: usize)
    where
        F: WithSmallOrderMulGroup<3>,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
        Pcs::Commitment: Additive<F>,
        Keccak256Transcript<Cursor<Vec<u8>>>: TranscriptRead<Pcs::CommitmentChunk, F>
            + TranscriptWrite<Pcs::CommitmentChunk, F>
            + InMemoryTranscript<Param = ()>,
    {
        let mut rng = seeded_std_rng();

        let n = 1 << num_vars;
        let param = Pcs::setup(n, 0, &mut rng).unwrap();
        let (pp, vp) = Pcs::trim(&param, n, 0).unwrap();

        let s_polys = s_polys(num_vars);
        let polys = iter::repeat_with(|| UnivariatePolynomial::lagrange(rand_vec(n, &mut rng)))
            .take(10)
            .collect_vec();
        let comms = polys
            .iter()
            .map(|poly| Pcs::commit(&pp, poly).unwrap())
            .collect_vec();
        let point = rand_vec(num_vars, &mut rng);
        let evals = izip!(0.., &polys)
            .flat_map(|(idx, poly)| {
                let point = &point;
                let max_rotation = 1 << (num_vars - 1);
                let num_rotations = rng.gen_range(1..3.min(max_rotation));
                let rotation_range = -(5.min(max_rotation) as i32)..=5.min(max_rotation) as i32;
                iter::repeat_with(|| rng.gen_range(rotation_range.clone()))
                    .unique()
                    .take(num_rotations)
                    .map(move |rotation| {
                        let mut poly = poly.coeffs().to_vec();
                        if rotation < 0 {
                            poly.rotate_right(rotation.unsigned_abs() as usize)
                        } else {
                            poly.rotate_left(rotation.unsigned_abs() as usize)
                        }
                        let eval = MultilinearPolynomial::new(poly).evaluate(point);
                        (Query::new(idx, Rotation(rotation)), eval)
                    })
                    .collect_vec()
            })
            .collect_vec();

        let proof = {
            let mut transcript = Keccak256Transcript::default();
            prove_multilinear_eval::<F, Pcs>(
                &pp,
                num_vars,
                &s_polys,
                &polys,
                &comms,
                &point,
                &evals,
                &mut transcript,
            )
            .unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
            verify_multilinear_eval::<F, Pcs>(
                &vp,
                num_vars,
                &comms,
                &point,
                &evals,
                &mut transcript,
            )
        };
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn prove_verify() {
        type Pcs = UnivariateKzg<Bn256>;

        for num_vars in 2..16 {
            run_prove_verify::<Fr, Pcs>(num_vars);
        }
    }
}
