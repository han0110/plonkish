//! Implementation of multilinear polynomial commitment scheme described in
//! [GLSTW21].
//! Most part are ported from https://github.com/conroi/lcpc with reorganization
//! to fit [`PolynomialCommitmentScheme`].
//!
//! [GLSTW21]: https://eprint.iacr.org/2021/1043.pdf

use crate::{
    pcs::{multilinear::validate_input, Evaluation, Point, PolynomialCommitmentScheme},
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, inner_product, PrimeField},
        code::{Brakedown, BrakedownSpec, LinearCodes},
        hash::{Hash, Output},
        parallel::{num_threads, parallelize, parallelize_iter},
        transcript::{FieldTranscript, TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{borrow::Cow, marker::PhantomData, mem::size_of, slice};

#[derive(Debug)]
pub struct MultilinearBrakedown<F: PrimeField, H: Hash, S: BrakedownSpec>(PhantomData<(F, H, S)>);

impl<F: PrimeField, H: Hash, S: BrakedownSpec> Clone for MultilinearBrakedown<F, H, S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultilinearBrakedownParam<F: PrimeField> {
    num_vars: usize,
    num_rows: usize,
    brakedown: Brakedown<F>,
}

impl<F: PrimeField> MultilinearBrakedownParam<F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn brakedown(&self) -> &Brakedown<F> {
        &self.brakedown
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct MultilinearBrakedownCommitment<F, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

impl<F: PrimeField, H: Hash> MultilinearBrakedownCommitment<F, H> {
    fn from_root(root: Output<H>) -> Self {
        Self {
            root,
            ..Default::default()
        }
    }

    pub fn rows(&self) -> &[F] {
        &self.rows
    }

    pub fn intermediate_hashes(&self) -> &[Output<H>] {
        &self.intermediate_hashes
    }

    pub fn root(&self) -> &Output<H> {
        &self.root
    }
}

impl<F: PrimeField, H: Hash> AsRef<[Output<H>]> for MultilinearBrakedownCommitment<F, H> {
    fn as_ref(&self) -> &[Output<H>] {
        slice::from_ref(&self.root)
    }
}

impl<F, H, S> PolynomialCommitmentScheme<F> for MultilinearBrakedown<F, H, S>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakedownSpec,
{
    type Param = MultilinearBrakedownParam<F>;
    type ProverParam = MultilinearBrakedownParam<F>;
    type VerifierParam = MultilinearBrakedownParam<F>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = MultilinearBrakedownCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(poly_size: usize, _: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;
        let brakedown = Brakedown::new_multilinear::<S>(num_vars, 20.min((1 << num_vars) - 1), rng);
        Ok(MultilinearBrakedownParam {
            num_vars,
            num_rows: (1 << num_vars) / brakedown.row_len(),
            brakedown,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        _: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        assert!(poly_size.is_power_of_two());
        if poly_size == 1 << param.num_vars {
            Ok((param.clone(), param.clone()))
        } else {
            Err(Error::InvalidPcsParam(
                "Can't trim MultilinearBrakedownParam into different poly_size".to_string(),
            ))
        }
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.brakedown.row_len();
        let codeword_len = pp.brakedown.codeword_len();
        let mut rows = vec![F::ZERO; pp.num_rows * codeword_len];

        // encode rows
        let chunk_size = div_ceil(pp.num_rows, num_threads());
        parallelize_iter(
            rows.chunks_exact_mut(chunk_size * codeword_len)
                .zip(poly.evals().chunks_exact(chunk_size * row_len)),
            |(rows, evals)| {
                for (row, evals) in rows
                    .chunks_exact_mut(codeword_len)
                    .zip(evals.chunks_exact(row_len))
                {
                    row[..evals.len()].copy_from_slice(evals);
                    pp.brakedown.encode(row);
                }
            },
        );

        // hash columns
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut hashes = vec![Output::<H>::default(); (2 << depth) - 1];
        parallelize(&mut hashes[..codeword_len], |(hashes, start)| {
            let mut hasher = H::new();
            for (hash, column) in hashes.iter_mut().zip(start..) {
                rows.iter()
                    .skip(column)
                    .step_by(codeword_len)
                    .for_each(|item| hasher.update_field_element(item));
                hasher.finalize_into_reset(hash);
            }
        });

        // merklize column hashes
        let mut offset = 0;
        for width in (1..=depth).rev().map(|depth| 1 << depth) {
            let (input, output) = hashes[offset..].split_at_mut(width);
            let chunk_size = div_ceil(output.len(), num_threads());
            parallelize_iter(
                input
                    .chunks(2 * chunk_size)
                    .zip(output.chunks_mut(chunk_size)),
                |(input, output)| {
                    let mut hasher = H::new();
                    for (input, output) in input.chunks_exact(2).zip(output.iter_mut()) {
                        hasher.update(&input[0]);
                        hasher.update(&input[1]);
                        hasher.finalize_into_reset(output);
                    }
                },
            );
            offset += width;
        }

        let (intermediate_hashes, root) = {
            let mut intermediate_hashes = hashes;
            let root = intermediate_hashes.pop().unwrap();
            (intermediate_hashes, root)
        };

        Ok(MultilinearBrakedownCommitment {
            rows,
            intermediate_hashes,
            root,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        validate_input("open", pp.num_vars(), [poly], [point])?;

        let row_len = pp.brakedown.row_len();
        let codeword_len = pp.brakedown.codeword_len();

        // prove proximity
        let (t_0, t_1) = point_to_tensor(pp.num_rows, point);
        let t_0_combined_row = if pp.num_rows > 1 {
            let combine = |combined_row: &mut [F], coeffs: &[F]| {
                parallelize(combined_row, |(combined_row, offset)| {
                    combined_row
                        .iter_mut()
                        .zip(offset..)
                        .for_each(|(combined, column)| {
                            *combined = F::ZERO;
                            coeffs
                                .iter()
                                .zip(poly.evals().iter().skip(column).step_by(row_len))
                                .for_each(|(coeff, eval)| {
                                    *combined += *coeff * eval;
                                });
                        })
                });
            };
            let mut combined_row = vec![F::ZERO; row_len];
            for _ in 0..pp.brakedown.num_proximity_testing() {
                let coeffs = transcript.squeeze_challenges(pp.num_rows);
                combine(&mut combined_row, &coeffs);
                transcript.write_field_elements(&combined_row)?;
            }
            combine(&mut combined_row, &t_0);
            Cow::Owned(combined_row)
        } else {
            Cow::Borrowed(poly.evals())
        };
        transcript.write_field_elements(t_0_combined_row.iter())?;
        if cfg!(feature = "sanity-check") {
            assert_eq!(inner_product(t_0_combined_row.as_ref(), &t_1), *eval);
        }

        // open merkle tree
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        for _ in 0..pp.brakedown.num_column_opening() {
            let column = squeeze_challenge_idx(transcript, codeword_len);

            transcript.write_field_elements(comm.rows.iter().skip(column).step_by(codeword_len))?;

            let mut offset = 0;
            for (idx, width) in (1..=depth).rev().map(|depth| 1 << depth).enumerate() {
                let neighbor_idx = (column >> idx) ^ 1;
                transcript.write_commitment(&comm.intermediate_hashes[offset + neighbor_idx])?;
                offset += width;
            }
        }

        Ok(())
    }

    // TODO: Apply 2022/1355
    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        for eval in evals {
            Self::open(
                pp,
                polys[eval.poly()],
                comms[eval.poly()],
                &points[eval.point()],
                eval.value(),
                transcript,
            )?;
        }
        Ok(())
    }

    fn read_commitments(
        _: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        transcript.read_commitments(num_polys).map(|roots| {
            roots
                .into_iter()
                .map(MultilinearBrakedownCommitment::from_root)
                .collect_vec()
        })
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        validate_input("verify", vp.num_vars(), [], [point])?;

        let row_len = vp.brakedown.row_len();
        let codeword_len = vp.brakedown.codeword_len();

        let (t_0, t_1) = point_to_tensor(vp.num_rows, point);
        let mut combined_rows = Vec::with_capacity(vp.brakedown.num_proximity_testing() + 1);
        if vp.num_rows > 1 {
            let coeffs = transcript.squeeze_challenges(vp.num_rows);
            let mut combined_row = transcript.read_field_elements(row_len)?;
            combined_row.resize(codeword_len, F::ZERO);
            vp.brakedown.encode(&mut combined_row);
            combined_rows.push((coeffs, combined_row));
        }
        combined_rows.push({
            let mut combined_row = transcript.read_field_elements(row_len)?;
            combined_row.resize(codeword_len, F::ZERO);
            vp.brakedown.encode(&mut combined_row);
            (t_0, combined_row)
        });

        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        for _ in 0..vp.brakedown.num_column_opening() {
            let column = squeeze_challenge_idx(transcript, codeword_len);
            let items = transcript.read_field_elements(vp.num_rows)?;
            let path = transcript.read_commitments(depth)?;

            // verify proximity
            for (coeff, encoded) in combined_rows.iter() {
                let item = if vp.num_rows > 1 {
                    inner_product(coeff, &items)
                } else {
                    items[0]
                };
                if item != encoded[column] {
                    return Err(Error::InvalidPcsOpen("Proximity failure".to_string()));
                }
            }

            // verify merkle tree opening
            let mut hasher = H::new();
            let mut output = {
                for item in items.iter() {
                    hasher.update_field_element(item);
                }
                hasher.finalize_fixed_reset()
            };
            for (idx, neighbor) in path.iter().enumerate() {
                if (column >> idx) & 1 == 0 {
                    hasher.update(&output);
                    hasher.update(neighbor);
                } else {
                    hasher.update(neighbor);
                    hasher.update(&output);
                }
                output = hasher.finalize_fixed_reset();
            }
            if &output != comm.root() {
                return Err(Error::InvalidPcsOpen(
                    "Invalid merkle tree opening".to_string(),
                ));
            }
        }

        // verify consistency
        let t_0_combined_row = combined_rows
            .last()
            .map(|(_, combined_row)| &combined_row[..row_len])
            .unwrap();
        if inner_product(t_0_combined_row, &t_1) != *eval {
            return Err(Error::InvalidPcsOpen("Consistency failure".to_string()));
        }

        Ok(())
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let comms = comms.into_iter().collect_vec();
        for eval in evals {
            Self::verify(
                vp,
                comms[eval.poly()],
                &points[eval.point()],
                eval.value(),
                transcript,
            )?;
        }
        Ok(())
    }
}

fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    let t_0 = MultilinearPolynomial::eq_xy(lo).into_evals();
    let t_1 = MultilinearPolynomial::eq_xy(hi).into_evals();
    (t_0, t_1)
}

fn squeeze_challenge_idx<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    cap: usize,
) -> usize {
    let challenge = transcript.squeeze_challenge();
    let mut bytes = [0; size_of::<u32>()];
    bytes.copy_from_slice(&challenge.to_repr().as_ref()[..size_of::<u32>()]);
    u32::from_le_bytes(bytes) as usize % cap
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::brakedown::MultilinearBrakedown,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
        },
        util::{code::BrakedownSpec6, hash::Keccak256, transcript::Keccak256Transcript},
    };
    use halo2_curves::bn256::Fr;

    type Pcs = MultilinearBrakedown<Fr, Keccak256, BrakedownSpec6>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
