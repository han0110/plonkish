use crate::{
    pcs::Evaluation,
    piop::sum_check::{
        vanilla::{EvaluationsProver, VanillaSumCheck},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    snark::hyperplonk::verifier::{pcs_query, point_offset, points},
    util::{
        arithmetic::{div_ceil, BatchInvert, BooleanHypercube, PrimeField},
        end_timer,
        expression::{CommonPolynomial, Expression},
        parallel::{par_map_collect, par_sort_unstable, parallelize},
        start_timer,
        transcript::TranscriptWrite,
        Itertools,
    },
    Error,
};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    iter,
};

pub(super) fn instances_polys<'a, F: PrimeField>(
    num_vars: usize,
    instances: impl IntoIterator<Item = impl IntoIterator<Item = &'a F>>,
) -> Vec<MultilinearPolynomial<F>> {
    let bh = BooleanHypercube::new(num_vars);
    instances
        .into_iter()
        .map(|instances| {
            let mut poly = vec![F::zero(); 1 << num_vars];
            for (b, instance) in bh.iter().skip(1).zip(instances.into_iter()) {
                poly[b] = *instance;
            }
            poly
        })
        .map(MultilinearPolynomial::new)
        .collect_vec()
}

#[allow(clippy::type_complexity)]
pub(super) fn lookup_permuted_polys<F: PrimeField + Ord + Hash>(
    lookups: &[Vec<(Expression<F>, Expression<F>)>],
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    theta: &F,
) -> Result<
    (
        Vec<[MultilinearPolynomial<F>; 2]>,
        Vec<[MultilinearPolynomial<F>; 2]>,
    ),
    Error,
> {
    if lookups.is_empty() {
        return Ok(Default::default());
    }

    let num_vars = polys[0].num_vars();
    let nth_map = BooleanHypercube::new(num_vars).nth_map();
    let expression = lookups
        .iter()
        .flat_map(|lookup| lookup.iter().map(|(input, table)| (input + table)))
        .sum::<Expression<_>>();
    let lagranges = {
        let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
        expression
            .used_langrange()
            .into_iter()
            .map(|i| (i, bh[i.rem_euclid(1 << num_vars) as usize]))
            .collect::<HashSet<_>>()
    };
    let identities = {
        let max_used_identity = expression
            .used_identity()
            .into_iter()
            .max()
            .unwrap_or_default();
        (0..=max_used_identity)
            .map(|idx| (idx as u64) << num_vars)
            .collect_vec()
    };
    Ok(lookups
        .iter()
        .map(|lookup| {
            lookup_permuted_poly(
                lookup,
                &nth_map,
                &lagranges,
                &identities,
                polys,
                challenges,
                theta,
            )
        })
        .try_collect::<_, Vec<_>, _>()?
        .into_iter()
        .unzip::<_, _, Vec<_>, Vec<_>>())
}

#[allow(clippy::type_complexity)]
fn lookup_permuted_poly<F: PrimeField + Ord + Hash>(
    lookup: &[(Expression<F>, Expression<F>)],
    nth_map: &[usize],
    lagranges: &HashSet<(i32, usize)>,
    identities: &[u64],
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    theta: &F,
) -> Result<([MultilinearPolynomial<F>; 2], [MultilinearPolynomial<F>; 2]), Error> {
    let num_vars = polys[0].num_vars();
    let bh = BooleanHypercube::new(num_vars);
    let compress = |expressions: &[&Expression<F>]| {
        expressions
            .iter()
            .map(|expression| {
                let mut compressed = vec![F::zero(); 1 << num_vars];
                parallelize(&mut compressed, |(compressed, start)| {
                    for (b, compressed) in (start..).zip(compressed) {
                        *compressed = expression.evaluate(
                            &|constant| constant,
                            &|common_poly| match common_poly {
                                CommonPolynomial::Lagrange(i) => lagranges
                                    .contains(&(i, b))
                                    .then(F::one)
                                    .unwrap_or_else(F::zero),
                                CommonPolynomial::Identity(idx) => {
                                    F::from(b as u64 + identities[idx])
                                }
                                CommonPolynomial::EqXY(_) => unreachable!(),
                            },
                            &|query| polys[query.poly()][bh.rotate(b, query.rotation())],
                            &|challenge| challenges[challenge],
                            &|value| -value,
                            &|lhs, rhs| lhs + &rhs,
                            &|lhs, rhs| lhs * &rhs,
                            &|value, scalar| value * &scalar,
                        );
                    }
                });
                MultilinearPolynomial::new(compressed)
            })
            .reduce(|mut acc, poly| {
                acc *= theta;
                &acc + &poly
            })
            .unwrap()
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

    let timer = start_timer(|| "permuted_input_poly");
    let permuted_input_poly = {
        let mut permuted_input_poly = compressed_input_poly.evals().to_vec();
        par_sort_unstable(&mut permuted_input_poly[1..]);
        MultilinearPolynomial::new(permuted_input_poly)
    };
    end_timer(timer);

    let timer = start_timer(|| "permuted_table_poly");
    let permuted_table_poly = {
        let timer = start_timer(|| "table_count");
        let mut table_count = compressed_table_poly[1..].iter().fold(
            HashMap::<_, u32>::new(),
            |mut table_count, value| {
                table_count
                    .entry(value)
                    .and_modify(|table_count| *table_count += 1)
                    .or_insert(1);
                table_count
            },
        );
        end_timer(timer);

        let timer = start_timer(|| "repeated_indices");
        let mut permuted_table_poly = vec![F::zero(); 1 << num_vars];
        let mut repeated_indices = permuted_table_poly
            .iter_mut()
            .zip(permuted_input_poly.iter())
            .enumerate()
            .skip(1)
            .filter_map(|(idx, (table, input))| {
                if idx == 1 || *input != permuted_input_poly[idx - 1] {
                    *table = *input;
                    if let Some(count) = table_count.get_mut(input) {
                        assert!(*count > 0);
                        *count -= 1;
                        None
                    } else {
                        Some(Err(Error::InvalidSnark("Invalid lookup input".to_string())))
                    }
                } else {
                    Some(Ok(idx))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (value, count) in table_count.into_iter() {
            for _ in 0..count {
                permuted_table_poly[repeated_indices.pop().unwrap()] = *value;
            }
        }
        end_timer(timer);

        if cfg!(feature = "sanity-check") {
            assert!(repeated_indices.is_empty());
        }

        MultilinearPolynomial::new(permuted_table_poly)
    };
    end_timer(timer);

    if cfg!(feature = "sanity-check") {
        let mut last = None;
        for (input, table) in permuted_input_poly
            .iter()
            .zip(permuted_table_poly.iter())
            .skip(1)
        {
            if input != table {
                assert_eq!(input, last.unwrap());
            }
            last = Some(input);
        }
    }

    let timer = start_timer(|| "into_bh_order");
    let [permuted_input_poly, permuted_table_poly] = [&permuted_input_poly, &permuted_table_poly]
        .map(|poly| MultilinearPolynomial::new(par_map_collect(nth_map, |b| poly[*b])));
    end_timer(timer);

    Ok((
        [compressed_input_poly, compressed_table_poly],
        [permuted_input_poly, permuted_table_poly],
    ))
}

pub(super) fn lookup_z_polys<F: PrimeField>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    permuted_polys: &[[MultilinearPolynomial<F>; 2]],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    compressed_polys
        .iter()
        .zip_eq(permuted_polys.iter())
        .map(
            |(
                [compressed_input_poly, compressed_table_poly],
                [permuted_input_poly, permuted_table_poly],
            )| {
                lookup_z_poly(
                    compressed_input_poly,
                    compressed_table_poly,
                    permuted_input_poly,
                    permuted_table_poly,
                    beta,
                    gamma,
                )
            },
        )
        .collect()
}

fn lookup_z_poly<F: PrimeField>(
    compressed_input_poly: &MultilinearPolynomial<F>,
    compressed_table_poly: &MultilinearPolynomial<F>,
    permuted_input_poly: &MultilinearPolynomial<F>,
    permuted_table_poly: &MultilinearPolynomial<F>,
    beta: &F,
    gamma: &F,
) -> MultilinearPolynomial<F> {
    let num_vars = compressed_input_poly.num_vars();
    let mut product = vec![F::zero(); 1 << num_vars];

    let timer = start_timer(|| "product");
    parallelize(&mut product, |(product, start)| {
        for ((product, permuted_input), permuted_table) in product
            .iter_mut()
            .zip(permuted_input_poly[start..].iter())
            .zip(permuted_table_poly[start..].iter())
        {
            *product = (*beta + permuted_input) * &(*gamma + permuted_table);
        }
    });

    product.iter_mut().batch_invert();

    parallelize(&mut product, |(product, start)| {
        for ((product, compressed_input), compressed_table) in product
            .iter_mut()
            .zip(compressed_input_poly[start..].iter())
            .zip(compressed_table_poly[start..].iter())
        {
            *product *= (*beta + compressed_input) * &(*gamma + compressed_table);
        }
    });
    end_timer(timer);

    let _timer = start_timer(|| "z_poly");
    let mut z_poly = MultilinearPolynomial::new(vec![F::zero(); 1 << num_vars]);
    z_poly[1] = F::one();
    for (b, b_next) in BooleanHypercube::new(num_vars)
        .iter()
        .skip(1)
        .tuple_windows()
    {
        z_poly[b_next] = z_poly[b] * &product[b];
    }

    if cfg!(feature = "sanity-check") {
        let b_last = BooleanHypercube::new(num_vars).iter().last().unwrap();
        assert_eq!(z_poly[b_last] * &product[b_last], F::one());
    }

    z_poly
}

pub(super) fn permutation_z_polys<F: PrimeField>(
    max_degree: usize,
    permutation_polys: &[(usize, MultilinearPolynomial<F>)],
    polys: &[&MultilinearPolynomial<F>],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    if permutation_polys.is_empty() {
        return Vec::new();
    }

    let chunk_size = max_degree - 1;
    let num_chunks = div_ceil(permutation_polys.len(), chunk_size);
    let num_vars = polys[0].num_vars();

    let timer = start_timer(|| "products");
    let products = permutation_polys
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, permutation_polys)| {
            let mut product = vec![F::one(); 1 << num_vars];

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

            product.iter_mut().batch_invert();

            for ((poly, _), idx) in permutation_polys.iter().zip(chunk_idx * chunk_size..) {
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), id) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(((idx << num_vars) + start) as u64..)
                    {
                        *product *= (F::from(id) * beta) + gamma + value;
                    }
                });
            }

            product
        })
        .collect_vec();
    end_timer(timer);

    let timer = start_timer(|| "z_polys");
    let z = iter::empty()
        .chain(iter::repeat_with(F::zero).take(num_chunks))
        .chain(Some(F::one()))
        .chain(
            BooleanHypercube::new(num_vars)
                .iter()
                .skip(1)
                .flat_map(|b| iter::repeat(b).take(num_chunks))
                .zip(products.iter().cycle())
                .scan(F::one(), |state, (b, product)| {
                    *state *= &product[b];
                    Some(*state)
                }),
        )
        .take(num_chunks << num_vars)
        .collect_vec();

    if cfg!(feature = "sanity-check") {
        let b_last = BooleanHypercube::new(num_vars).iter().last().unwrap();
        assert_eq!(
            *z.last().unwrap() * products.last().unwrap()[b_last],
            F::one()
        );
    }

    drop(products);
    end_timer(timer);

    let _timer = start_timer(|| "into_bh_order");
    let nth_map = BooleanHypercube::new(num_vars)
        .nth_map()
        .into_iter()
        .map(|b| num_chunks * b)
        .collect_vec();
    (0..num_chunks)
        .map(|offset| MultilinearPolynomial::new(par_map_collect(&nth_map, |b| z[offset + b])))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_sum_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let x =
        VanillaSumCheck::<EvaluationsProver<_>>::prove(&(), num_vars, virtual_poly, transcript)?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(polys[query.poly()].evaluate_for_rotation(&x, query.rotation()))
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    for eval in evals.iter() {
        transcript.write_scalar(*eval.value())?;
    }

    Ok((points(&pcs_query, &x), evals))
}
