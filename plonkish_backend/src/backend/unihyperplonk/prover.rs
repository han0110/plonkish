use crate::{
    backend::hyperplonk::verifier::pcs_query,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::PrimeField,
        expression::{rotate::Lexical, Expression, Query},
        transcript::FieldTranscriptWrite,
        Itertools,
    },
    Error,
};
use std::borrow::Borrow;

pub(super) use crate::backend::hyperplonk::prover::{
    instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys, permutation_z_polys,
};

#[allow(clippy::type_complexity)]
pub(super) fn prove_zero_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<F>, Vec<(Query, F)>), Error> {
    prove_sum_check(
        num_instance_poly,
        expression,
        F::ZERO,
        polys,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_sum_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<F>, Vec<(Query, F)>), Error> {
    let polys = polys.iter().map(Borrow::borrow).collect_vec();
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys, &challenges, &ys);
    let (_, x, mut evals) = ClassicSumCheck::<EvaluationsProver<_>, Lexical>::prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    evals.retain(|query, _| pcs_query.contains(query));
    transcript.write_field_elements(evals.values())?;

    Ok((x, evals.into_iter().collect()))
}
