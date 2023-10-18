use crate::{
    backend::hyperplonk::verifier::{instance_evals, pcs_query},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        evaluate, SumCheck,
    },
    util::{
        arithmetic::PrimeField,
        chain,
        expression::{rotate::Lexical, Expression, Query},
        izip,
        transcript::FieldTranscriptRead,
        Itertools,
    },
    Error,
};

#[allow(clippy::type_complexity)]
pub(super) fn verify_zero_check<F: PrimeField>(
    num_vars: usize,
    expression: &Expression<F>,
    instances: &[Vec<F>],
    challenges: &[F],
    y: &[F],
    transcript: &mut impl FieldTranscriptRead<F>,
) -> Result<(Vec<F>, Vec<(Query, F)>), Error> {
    verify_sum_check(
        num_vars,
        expression,
        F::ZERO,
        instances,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn verify_sum_check<F: PrimeField>(
    num_vars: usize,
    expression: &Expression<F>,
    sum: F,
    instances: &[Vec<F>],
    challenges: &[F],
    y: &[F],
    transcript: &mut impl FieldTranscriptRead<F>,
) -> Result<(Vec<F>, Vec<(Query, F)>), Error> {
    let (x_eval, x) = ClassicSumCheck::<EvaluationsProver<_>, Lexical>::verify(
        &(),
        num_vars,
        expression.degree(),
        sum,
        transcript,
    )?;

    let evals = {
        let pcs_query = pcs_query(expression, instances.len());
        let evals = transcript.read_field_elements(pcs_query.len())?;
        izip!(pcs_query, evals).collect_vec()
    };

    let query_eval = {
        let instance_evals = instance_evals::<_, Lexical>(num_vars, expression, instances, &x);
        let evals = chain![evals.iter().copied(), instance_evals].collect();
        evaluate::<_, Lexical>(expression, num_vars, &evals, challenges, &[y], &x)
    };
    if query_eval != x_eval {
        return Err(Error::InvalidSnark(
            "Unmatched between sum_check output and query evaluation".to_string(),
        ));
    }

    Ok((x, evals.into_iter().collect()))
}
