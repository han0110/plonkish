use crate::{
    backend::hyperplonk::verifier::{pcs_query, point_offset, points},
    pcs::Evaluation,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps_by, sum, BatchInvert, PrimeField},
        chain, end_timer,
        expression::{
            rotate::{BinaryField, Rotatable},
            CommonPolynomial, Expression, Rotation,
        },
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        start_timer,
        transcript::FieldTranscriptWrite,
        Itertools,
    },
    Error,
};
use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    hash::Hash,
};

pub(crate) fn instance_polys<'a, F: PrimeField, R: Rotatable + From<usize>>(
    num_vars: usize,
    instances: impl IntoIterator<Item = impl IntoIterator<Item = &'a F>>,
) -> Vec<MultilinearPolynomial<F>> {
    let usable_indices = R::from(num_vars).usable_indices();
    instances
        .into_iter()
        .map(|instances| {
            let mut poly = vec![F::ZERO; 1 << num_vars];
            for (b, instance) in usable_indices.iter().zip(instances.into_iter()) {
                poly[*b] = *instance;
            }
            poly
        })
        .map(MultilinearPolynomial::new)
        .collect()
}

pub(crate) fn lookup_compressed_polys<F: PrimeField, R: Rotatable + From<usize>>(
    lookups: &[Vec<(Expression<F>, Expression<F>)>],
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    challenges: &[F],
    betas: &[F],
) -> Vec<[MultilinearPolynomial<F>; 2]> {
    if lookups.is_empty() {
        return Default::default();
    }

    let polys = polys.iter().map(Borrow::borrow).collect_vec();
    let num_vars = polys[0].num_vars();
    let expression = lookups
        .iter()
        .flat_map(|lookup| lookup.iter().map(|(input, table)| (input + table)))
        .sum::<Expression<_>>();
    let lagranges = {
        let rotatable = R::from(num_vars);
        expression
            .used_langrange()
            .into_iter()
            .map(|i| (i, rotatable.nth(i)))
            .collect::<HashSet<_>>()
    };
    lookups
        .iter()
        .map(|lookup| lookup_compressed_poly::<_, R>(lookup, &lagranges, &polys, challenges, betas))
        .collect()
}

pub(super) fn lookup_compressed_poly<F: PrimeField, R: Rotatable + From<usize>>(
    lookup: &[(Expression<F>, Expression<F>)],
    lagranges: &HashSet<(i32, usize)>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    betas: &[F],
) -> [MultilinearPolynomial<F>; 2] {
    let num_vars = polys[0].num_vars();
    let rotatable = R::from(num_vars);
    let compress = |expressions: &[&Expression<F>]| {
        betas
            .iter()
            .copied()
            .zip(expressions.iter().map(|expression| {
                let mut compressed = vec![F::ZERO; 1 << num_vars];
                parallelize(&mut compressed, |(compressed, start)| {
                    for (b, compressed) in (start..).zip(compressed) {
                        *compressed = expression.evaluate(
                            &|constant| constant,
                            &|common_poly| match common_poly {
                                CommonPolynomial::Identity => F::from(b as u64),
                                CommonPolynomial::Lagrange(i) => {
                                    if lagranges.contains(&(i, b)) {
                                        F::ONE
                                    } else {
                                        F::ZERO
                                    }
                                }
                                CommonPolynomial::EqXY(_) => unreachable!(),
                            },
                            &|query| polys[query.poly()][rotatable.rotate(b, query.rotation())],
                            &|challenge| challenges[challenge],
                            &|value| -value,
                            &|lhs, rhs| lhs + &rhs,
                            &|lhs, rhs| lhs * &rhs,
                            &|value, scalar| value * &scalar,
                        );
                    }
                });
                MultilinearPolynomial::new(compressed)
            }))
            .sum::<MultilinearPolynomial<_>>()
    };

    let (inputs, tables) = lookup
        .iter()
        .map(|(input, table)| (input, table))
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let timer = start_timer(|| "compressed_input_poly");
    let compressed_input_poly = compress(&inputs);
    end_timer(timer);

    let timer = start_timer(|| "compressed_table_poly");
    let compressed_table_poly = compress(&tables);
    end_timer(timer);

    [compressed_input_poly, compressed_table_poly]
}

pub(crate) fn lookup_m_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
) -> Result<Vec<MultilinearPolynomial<F>>, Error> {
    compressed_polys.iter().map(lookup_m_poly).try_collect()
}

pub(super) fn lookup_m_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
) -> Result<MultilinearPolynomial<F>, Error> {
    let [input, table] = compressed_polys;

    let counts = {
        let indice_map = table.iter().zip(0..).collect::<HashMap<_, usize>>();

        let chunk_size = div_ceil(input.evals().len(), num_threads());
        let num_chunks = div_ceil(input.evals().len(), chunk_size);
        let mut counts = vec![HashMap::new(); num_chunks];
        let mut valids = vec![true; num_chunks];
        parallelize_iter(
            counts
                .iter_mut()
                .zip(valids.iter_mut())
                .zip((0..).step_by(chunk_size)),
            |((count, valid), start)| {
                for input in input[start..].iter().take(chunk_size) {
                    if let Some(idx) = indice_map.get(input) {
                        count
                            .entry(*idx)
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    } else {
                        *valid = false;
                        break;
                    }
                }
            },
        );
        if valids.iter().any(|valid| !valid) {
            return Err(Error::InvalidSnark("Invalid lookup input".to_string()));
        }
        counts
    };

    let mut m = vec![0; 1 << input.num_vars()];
    for (idx, count) in counts.into_iter().flatten() {
        m[idx] += count;
    }
    let m = par_map_collect(m, |count| match count {
        0 => F::ZERO,
        1 => F::ONE,
        count => F::from(count),
    });
    Ok(MultilinearPolynomial::new(m))
}

pub(crate) fn lookup_h_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    m_polys: &[MultilinearPolynomial<F>],
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    compressed_polys
        .iter()
        .zip(m_polys.iter())
        .map(|(compressed_polys, m_poly)| lookup_h_poly(compressed_polys, m_poly, gamma))
        .collect()
}

pub(super) fn lookup_h_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
    m_poly: &MultilinearPolynomial<F>,
    gamma: &F,
) -> MultilinearPolynomial<F> {
    let [input, table] = compressed_polys;
    let mut h_input = vec![F::ZERO; 1 << input.num_vars()];
    let mut h_table = vec![F::ZERO; 1 << input.num_vars()];

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, input) in h_input.iter_mut().zip(input[start..].iter()) {
            *h_input = *gamma + input;
        }
    });
    parallelize(&mut h_table, |(h_table, start)| {
        for (h_table, table) in h_table.iter_mut().zip(table[start..].iter()) {
            *h_table = *gamma + table;
        }
    });

    let chunk_size = div_ceil(2 * h_input.len(), num_threads());
    parallelize_iter(
        chain![
            h_input.chunks_mut(chunk_size),
            h_table.chunks_mut(chunk_size)
        ],
        |h| {
            h.batch_invert();
        },
    );

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, (h_table, m)) in h_input
            .iter_mut()
            .zip(h_table[start..].iter().zip(m_poly[start..].iter()))
        {
            *h_input -= *h_table * m;
        }
    });

    if cfg!(feature = "sanity-check") {
        assert_eq!(sum::<F>(&h_input), F::ZERO);
    }

    MultilinearPolynomial::new(h_input)
}

pub(crate) fn permutation_z_polys<F: PrimeField, R: Rotatable + From<usize>>(
    num_chunks: usize,
    permutation_polys: &[(usize, MultilinearPolynomial<F>)],
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    if permutation_polys.is_empty() {
        return Vec::new();
    }

    let chunk_size = div_ceil(permutation_polys.len(), num_chunks);
    let polys = polys.iter().map(Borrow::borrow).collect_vec();
    let num_vars = polys[0].num_vars();

    let timer = start_timer(|| "products");
    let products = permutation_polys
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, permutation_polys)| {
            let mut product = vec![F::ONE; 1 << num_vars];

            for (poly, permutation_poly) in permutation_polys.iter() {
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), permutation) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(permutation_poly[start..].iter())
                    {
                        *product *= (*beta * permutation) + gamma + value;
                    }
                });
            }

            parallelize(&mut product, |(product, _)| {
                product.batch_invert();
            });

            for ((poly, _), idx) in permutation_polys.iter().zip(chunk_idx * chunk_size..) {
                let id_offset = idx << num_vars;
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), beta_id) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(steps_by(F::from((id_offset + start) as u64) * beta, *beta))
                    {
                        *product *= beta_id + gamma + value;
                    }
                });
            }

            product
        })
        .collect_vec();
    end_timer(timer);

    let _timer = start_timer(|| "z_polys");
    let mut z = vec![vec![F::ZERO; 1 << num_vars]; num_chunks];

    let usable_indices = R::from(num_vars).usable_indices();
    let first_idx = usable_indices[0];
    z[0][first_idx] = F::ONE;
    for chunk_idx in 1..num_chunks {
        z[chunk_idx][first_idx] = z[chunk_idx - 1][first_idx] * products[chunk_idx - 1][first_idx];
    }
    for (last_idx, idx) in usable_indices.iter().copied().tuple_windows() {
        z[0][idx] = z[num_chunks - 1][last_idx] * products[num_chunks - 1][last_idx];
        for chunk_idx in 1..num_chunks {
            z[chunk_idx][idx] = z[chunk_idx - 1][idx] * products[chunk_idx - 1][idx];
        }
    }

    if cfg!(feature = "sanity-check") {
        let last_idx = *usable_indices.last().unwrap();
        assert_eq!(
            z.last().unwrap()[last_idx] * products.last().unwrap()[last_idx],
            F::ONE
        );
    }

    z.into_iter().map(MultilinearPolynomial::new).collect()
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_zero_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
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
pub(crate) fn prove_sum_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let (_, x, evals) = ClassicSumCheck::<EvaluationsProver<_>, BinaryField>::prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(if query.rotation() == Rotation::cur() {
                    vec![evals[query]]
                } else {
                    polys[query.poly()].evaluate_for_rotation(&x, query.rotation())
                })
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

    Ok((points(&pcs_query, &x), evals))
}
