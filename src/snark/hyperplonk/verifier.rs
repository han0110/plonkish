use crate::{
    pcs::Evaluation,
    piop::sum_check::{
        evaluate, lagrange_eval,
        vanilla::{EvaluationsProver, VanillaSumCheck},
        SumCheck,
    },
    poly::multilinear::{rotation_eval, rotation_eval_points},
    util::{
        arithmetic::{inner_product, BooleanHypercube, PrimeField},
        expression::{Expression, Query, Rotation},
        transcript::TranscriptRead,
        Itertools,
    },
    Error,
};
use std::collections::{BTreeSet, HashMap};

#[allow(clippy::type_complexity)]
pub(super) fn verify_zero_check<F: PrimeField>(
    num_vars: usize,
    expression: &Expression<F>,
    instances: &[&[F]],
    challenges: &[F],
    y: &[F],
    transcript: &mut impl TranscriptRead<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let (x_eval, x) = VanillaSumCheck::<EvaluationsProver<_, true>>::verify(
        &(),
        num_vars,
        expression.degree(),
        F::zero(),
        transcript,
    )?;

    let pcs_query = pcs_query(expression, instances.len());
    let (evals_for_rotation, evals) = pcs_query
        .iter()
        .map(|query| {
            let evals_for_rotation = transcript.read_n_scalars(1 << query.rotation().distance())?;
            let eval = rotation_eval(&x, query.rotation(), &evals_for_rotation);
            Ok((evals_for_rotation, (*query, eval)))
        })
        .try_collect::<_, Vec<_>, _>()?
        .into_iter()
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let evals = instance_evals(num_vars, expression, instances, &x)
        .into_iter()
        .chain(evals)
        .collect();
    if evaluate(expression, num_vars, &evals, challenges, &[y], &x) != x_eval {
        return Err(Error::InvalidSnark(
            "Unmatched between sum_check output and query evaluation".to_string(),
        ));
    }

    let point_offset = point_offset(&pcs_query);
    let evals = pcs_query
        .iter()
        .zip(evals_for_rotation)
        .flat_map(|(query, evals_for_rotation)| {
            (point_offset[&query.rotation()]..)
                .zip(evals_for_rotation)
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect();
    Ok((points(&pcs_query, &x), evals))
}

fn instance_evals<F: PrimeField>(
    num_vars: usize,
    expression: &Expression<F>,
    instances: &[&[F]],
    x: &[F],
) -> Vec<(Query, F)> {
    let mut instance_query = expression.used_query();
    instance_query.retain(|query| query.poly() < instances.len());

    let lagranges = {
        let mut lagranges = instance_query.iter().fold(0..0, |range, query| {
            let i = -query.rotation().0;
            range.start.min(i)..range.end.max(i + instances[query.poly()].len() as i32)
        });
        if lagranges.start < 0 {
            lagranges.start -= 1;
        }
        if lagranges.end > 0 {
            lagranges.end += 1;
        }
        lagranges
    };

    let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
    let lagrange_evals = lagranges
        .filter_map(|i| {
            (i != 0).then(|| {
                let b = bh[i.rem_euclid(1 << num_vars as i32) as usize];
                (i, lagrange_eval(x, b))
            })
        })
        .collect::<HashMap<_, _>>();

    instance_query
        .into_iter()
        .map(|query| {
            let is = if query.rotation() > Rotation::cur() {
                (-query.rotation().0..0)
                    .chain(1..)
                    .take(instances[query.poly()].len())
                    .collect_vec()
            } else {
                (1 - query.rotation().0..)
                    .take(instances[query.poly()].len())
                    .collect_vec()
            };
            let eval = inner_product(
                instances[query.poly()],
                is.iter().map(|i| lagrange_evals.get(i).unwrap()),
            );
            (query, eval)
        })
        .collect()
}

pub(super) fn pcs_query<F: PrimeField>(
    expression: &Expression<F>,
    num_instance_poly: usize,
) -> BTreeSet<Query> {
    let mut used_query = expression.used_query();
    used_query.retain(|query| query.poly() >= num_instance_poly);
    used_query
}

pub(super) fn points<F: PrimeField>(pcs_query: &BTreeSet<Query>, x: &[F]) -> Vec<Vec<F>> {
    pcs_query
        .iter()
        .map(Query::rotation)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .flat_map(|rotation| rotation_eval_points(x, rotation))
        .collect_vec()
}

pub(super) fn point_offset(pcs_query: &BTreeSet<Query>) -> HashMap<Rotation, usize> {
    let rotations = pcs_query
        .iter()
        .map(Query::rotation)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect_vec();
    rotations.windows(2).fold(
        HashMap::from_iter([(rotations[0], 0)]),
        |mut point_offset, rotations| {
            let last_rotation = rotations[0];
            let offset = point_offset[&last_rotation] + (1 << last_rotation.distance());
            point_offset.insert(rotations[1], offset);
            point_offset
        },
    )
}
