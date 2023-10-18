//! Implementation of section 5.1 of [PH23].
//!
//! [PH23]: https://eprint.iacr.org/2023/1284.pdf

use crate::{
    pcs::{Evaluation, PolynomialCommitmentScheme},
    poly::{multilinear::MultilinearPolynomial, univariate::UnivariatePolynomial},
    util::{
        arithmetic::{
            inner_product, powers, product, BatchInvert, PrimeField, WithSmallOrderMulGroup,
        },
        chain, end_timer,
        expression::{
            evaluator::quotient::{QuotientEvaluator, Radix2Domain},
            rotate::{Lexical, Rotatable},
            Expression, Query, Rotation,
        },
        izip,
        parallel::parallelize,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Itertools,
    },
    Error,
};
use std::{collections::BTreeMap, mem};

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
    Pcs::Commitment: 'a,
{
    let domain = &Radix2Domain::<F>::new(num_vars, 2);
    let polys = polys.into_iter().collect_vec();
    let comms = comms.into_iter().collect_vec();
    let num_polys = polys.len();
    assert_eq!(comms.len(), num_polys);

    let (queries, evals) = evals.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();

    let gamma = transcript.squeeze_challenge();

    let powers_of_gamma = powers(gamma).take(queries.len()).collect_vec();
    let eval = inner_product(&powers_of_gamma, &evals);

    let u_step = -domain.n_inv() * eval;
    let (polys, eq_xy_0, eq_xy, u) = {
        let timer = start_timer(|| "u");
        let eq_xy = MultilinearPolynomial::eq_xy(point).into_evals();
        let step = {
            let lexical = Lexical::new(domain.k());
            let polys = chain![&queries].map(|query| (query.rotation(), polys[query.poly()]));
            let mut coeffs = vec![F::ZERO; domain.n()];
            izip!(&powers_of_gamma, polys).for_each(|(scalar, (rotation, poly))| {
                parallelize(&mut coeffs, |(coeffs, start)| {
                    let skip = lexical.rotate(start, rotation);
                    let scalar = *scalar;
                    izip!(coeffs, poly.coeffs().iter().cycle().skip(skip))
                        .for_each(|(coeffs, poly)| *coeffs += scalar * poly);
                });
            });
            parallelize(&mut coeffs, |(coeffs, start)| {
                izip!(coeffs, &eq_xy[start..]).for_each(|(coeffs, eq_xy)| {
                    *coeffs *= eq_xy;
                    *coeffs += u_step
                });
            });
            coeffs
        };
        let u = chain![&step]
            .scan(F::ZERO, |u, step| mem::replace(u, *u + step).into())
            .collect_vec();
        end_timer(timer);

        if cfg!(feature = "sanity-check") {
            assert_eq!(F::ZERO, u[domain.n() - 1] + step[domain.n() - 1]);
        }

        // TODO: Merge polys by rotation first to reduce FFT.
        let polys = chain![&polys]
            .map(|poly| domain.lagrange_to_monomial(poly.coeffs().into()))
            .map(UnivariatePolynomial::monomial)
            .collect_vec();
        let eq_xy_0 = eq_xy[0];
        let [eq_xy, u] = [eq_xy, u]
            .map(|buf| domain.lagrange_to_monomial(buf.into()))
            .map(UnivariatePolynomial::monomial);
        (polys, eq_xy_0, eq_xy, u)
    };

    let eq_xy_u_comm = Pcs::batch_commit_and_write(pp, [&eq_xy, &u], transcript)?;

    let alpha = transcript.squeeze_challenge();

    let expression = expression(num_polys, point, &queries, gamma, u_step, eq_xy_0, alpha);
    let q = {
        let polys = chain![&polys, [&eq_xy, &u]]
            .map(|poly| domain.monomial_to_extended_lagrange(poly.coeffs().into()))
            .collect_vec();

        let timer = start_timer(|| "quotient");
        let ev = {
            let polys = chain![&polys, s_polys].map(Vec::as_slice);
            QuotientEvaluator::new(domain, &expression, Default::default(), polys)
        };

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

    let evals = chain![
        chain![&queries].map(|query| (&polys[query.poly()], query.rotation())),
        [(&eq_xy, Rotation::cur())],
        (0..domain.k()).map(|idx| (&eq_xy, Rotation(1 << idx))),
        [Rotation::cur(), Rotation::next()].map(|rotation| (&u, rotation)),
    ]
    .map(|(poly, rotation)| poly.evaluate(&domain.rotate_point(x, rotation)))
    .collect_vec();

    transcript.write_field_elements(&evals)?;

    let polys = chain![&polys, [&eq_xy, &u, &q]].collect_vec();
    let comms = chain![comms, &eq_xy_u_comm, [&q_comm]];
    let (points, evals) = points_evals(domain, num_polys, &expression, &queries, &evals, x);

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
    Pcs::Commitment: 'a,
{
    let domain = &Radix2Domain::<F>::new(num_vars, 2);
    let comms = comms.into_iter().collect_vec();
    let num_polys = comms.len();

    let (queries, evals) = evals.iter().cloned().unzip::<_, _, Vec<_>, Vec<_>>();

    let gamma = transcript.squeeze_challenge();

    let powers_of_gamma = powers(gamma).take(evals.len()).collect_vec();
    let eval = inner_product(&powers_of_gamma, &evals);

    let u_step = -domain.n_inv() * eval;
    let eq_xy_0 = product(point.iter().map(|point_i| F::ONE - point_i));

    let eq_xy_u_comm = Pcs::read_commitments(vp, 2, transcript)?;

    let alpha = transcript.squeeze_challenge();

    let expression = expression(num_polys, point, &queries, gamma, u_step, eq_xy_0, alpha);

    let q_comm = Pcs::read_commitment(vp, transcript)?;

    let x = transcript.squeeze_challenge();

    let evals = transcript.read_field_elements(queries.len() + domain.k() + 3)?;

    let comms = chain![comms, &eq_xy_u_comm, [&q_comm]];
    let (points, evals) = points_evals(domain, num_polys, &expression, &queries, &evals, x);

    Pcs::batch_verify(vp, comms, &points, &evals, transcript)
}

pub fn s_polys<F: WithSmallOrderMulGroup<3>>(num_vars: usize) -> Vec<Vec<F>> {
    let domain = Radix2Domain::<F>::new(num_vars, 2);
    let vanishing = {
        let coset_scalar = match domain.n() % 3 {
            1 => domain.zeta(),
            2 => domain.zeta_inv(),
            _ => unreachable!(),
        };
        powers(domain.extended_omega().pow([domain.n() as u64]))
            .map(|value| coset_scalar * value - F::ONE)
            .take(1 << (domain.extended_k() - domain.k()))
            .collect_vec()
    };
    let omegas = powers(domain.extended_omega())
        .take(domain.extended_n())
        .collect_vec();
    let mut s_polys = vec![vec![F::ZERO; domain.extended_n()]; domain.k()];
    parallelize(&mut s_polys, |(s_polys, start)| {
        izip!(s_polys, start..).for_each(|(s_polys, idx)| {
            let exponent = 1 << idx;
            let offset = match exponent % 3 {
                1 => domain.zeta(),
                2 => domain.zeta_inv(),
                _ => unreachable!(),
            };
            izip!((0..).step_by(exponent), s_polys.iter_mut()).for_each(|(idx, value)| {
                *value = offset * omegas[idx % domain.extended_n()] - F::ONE
            });
            s_polys.batch_invert();
            izip!(s_polys.iter_mut(), vanishing.iter().cycle())
                .for_each(|(denom, numer)| *denom *= numer);
        })
    });
    s_polys
}

fn expression<F: PrimeField>(
    num_polys: usize,
    point: &[F],
    queries: &[Query],
    gamma: F,
    u_step: F,
    eq_xy_0: F,
    alpha: F,
) -> Expression<F> {
    let num_vars = point.len();
    let [gamma, u_step, eq_xy_0, alpha] =
        &[gamma, u_step, eq_xy_0, alpha].map(Expression::Constant);
    let eq_xy_ratios = {
        let mut denoms = point.iter().map(|point_i| F::ONE - point_i).collect_vec();
        denoms.batch_invert();
        izip!(point, denoms)
            .map(|(numer, denom)| denom * numer)
            .rev()
            .collect_vec()
    };
    let f =
        &Expression::distribute_powers(queries.iter().copied().map(Expression::Polynomial), gamma);
    let eq_xy = &Expression::Polynomial(Query::new(num_polys, Rotation::cur()));
    let rotated_eq_xys = (0..num_vars)
        .rev()
        .map(|rotation| Expression::Polynomial(Query::new(num_polys, Rotation(1 << rotation))))
        .collect_vec();
    let [u, u_next] = &[Rotation::cur(), Rotation::next()]
        .map(|rotation| Expression::Polynomial(Query::new(num_polys + 1, rotation)));
    let s = (num_polys + 2..)
        .take(num_vars)
        .map(|poly| Expression::<F>::Polynomial(Query::new(poly, Rotation::cur())))
        .collect_vec();
    let constraints = chain![
        [u_next - u - f * eq_xy - u_step],
        [&s[0] * (eq_xy - eq_xy_0)],
        izip!(&s, &rotated_eq_xys, &eq_xy_ratios)
            .map(|(s, rotated_eq_xy, eq_xy_ratio)| s * (eq_xy * eq_xy_ratio - rotated_eq_xy))
    ]
    .collect_vec();
    Expression::distribute_powers(&constraints, alpha)
        .simplified(None)
        .unwrap()
}

fn points_evals<F: WithSmallOrderMulGroup<3>>(
    domain: &Radix2Domain<F>,
    num_polys: usize,
    expression: &Expression<F>,
    queries: &[Query],
    evals: &[F],
    x: F,
) -> (Vec<F>, Vec<Evaluation<F>>) {
    let point_index = chain![
        queries.iter().map(Query::rotation),
        [Rotation::cur()],
        (0..domain.k()).map(|idx| Rotation(1 << idx)),
    ]
    .fold(BTreeMap::new(), |mut point_index, rotation| {
        let idx = point_index.len();
        point_index.entry(rotation).or_insert(idx);
        point_index
    });
    let points = point_index
        .iter()
        .sorted_by(|a, b| a.1.cmp(b.1))
        .map(|(rotation, _)| domain.rotate_point(x, *rotation))
        .collect_vec();
    let evals = {
        let queries = chain![
            queries.iter().cloned(),
            [Query::new(num_polys, Rotation::cur())],
            (0..domain.k()).map(|idx| Query::new(num_polys, Rotation(1 << idx))),
            [Rotation::cur(), Rotation::next()].map(|rotation| Query::new(num_polys + 1, rotation)),
        ]
        .collect_vec();
        let evals = chain![
            izip!(queries.iter().cloned(), evals.iter().copied()),
            izip!(
                chain![(num_polys + 2..).take(domain.k())]
                    .map(|idx| Query::new(idx, Rotation::cur())),
                s_evals(domain, x)
            )
        ]
        .collect();
        let q_eval = domain.evaluate(expression, &evals, &[], x) * vanishing_eval_inv(domain, x);
        chain![
            queries.into_iter().map(|query| Evaluation::new(
                query.poly(),
                point_index[&query.rotation()],
                evals[&query]
            )),
            [Evaluation::new(
                num_polys + 2,
                point_index[&Rotation::cur()],
                q_eval
            )],
        ]
        .collect_vec()
    };
    (points, evals)
}

fn s_evals<F: WithSmallOrderMulGroup<3>>(domain: &Radix2Domain<F>, x: F) -> Vec<F> {
    let vanishing_eval = x.pow([domain.n() as u64]) - F::ONE;
    let mut s_denom_evals = (0..domain.k())
        .map(|idx| x.pow([1 << idx]) - F::ONE)
        .collect_vec();
    s_denom_evals.batch_invert();
    s_denom_evals
        .iter()
        .map(|denom| vanishing_eval * denom)
        .collect()
}

fn vanishing_eval_inv<F: WithSmallOrderMulGroup<3>>(domain: &Radix2Domain<F>, x: F) -> F {
    (x.pow([domain.n() as u64]) - F::ONE).invert().unwrap()
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{univariate::UnivariateKzg, PolynomialCommitmentScheme},
        piop::multilinear_eval::ph23::{prove_multilinear_eval, s_polys, verify_multilinear_eval},
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
    use std::{io::Cursor, iter};

    fn run_prove_verify<F, Pcs>(num_vars: usize)
    where
        F: WithSmallOrderMulGroup<3>,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = UnivariatePolynomial<F>>,
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
                (-(idx.min(num_vars) as i32)..idx.min(num_vars) as i32).map(move |rotation| {
                    let mut poly = poly.coeffs().to_vec();
                    if rotation < 0 {
                        poly.rotate_right(rotation.unsigned_abs() as usize)
                    } else {
                        poly.rotate_left(rotation.unsigned_abs() as usize)
                    }
                    let eval = MultilinearPolynomial::new(poly).evaluate(point);
                    (Query::new(idx, Rotation(rotation)), eval)
                })
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
