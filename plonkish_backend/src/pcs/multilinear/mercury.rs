//! Implementation of https://eprint.iacr.org/2025/385

use crate::{
    pcs::{
        multilinear::additive,
        univariate::{err_too_large_deree, UnivariateKzg, UnivariateKzgCommitment},
        Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{
        multilinear::{merge_into, MultilinearPolynomial},
        univariate::UnivariatePolynomial,
    },
    util::{
        arithmetic::{
            horner_univariate_div, radix2_fft, root_of_unity, root_of_unity_inv, squares,
            transpose, Field, MultiMillerLoop,
        },
        izip_eq,
        parallel::parallelize,
        transcript::{TranscriptRead, TranscriptWrite},
        DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use halo2_curves::{ff::PrimeField, CurveAffine};
use rand::RngCore;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct Mercury<Pcs>(PhantomData<Pcs>);

impl<M> PolynomialCommitmentScheme<M::Scalar> for Mercury<UnivariateKzg<M>>
where
    M: MultiMillerLoop,
    M::Scalar: Serialize + DeserializeOwned,
    M::G1Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Scalar>,
    M::G2Affine: Serialize + DeserializeOwned + CurveAffine<ScalarExt = M::Scalar>,
{
    type Param = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Param;
    type ProverParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::ProverParam;
    type VerifierParam = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::VerifierParam;
    type Polynomial = MultilinearPolynomial<M::Scalar>;
    type Commitment = <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::Commitment;
    type CommitmentChunk =
        <UnivariateKzg<M> as PolynomialCommitmentScheme<M::Scalar>>::CommitmentChunk;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        UnivariateKzg::<M>::setup(poly_size, batch_size, rng)
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        UnivariateKzg::<M>::trim(param, poly_size, batch_size)
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("commit", pp.degree(), got));
        }

        Ok(UnivariateKzg::commit_monomial(pp, poly.evals()))
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        polys
            .into_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        if pp.degree() + 1 < poly.evals().len() {
            let got = poly.evals().len() - 1;
            return Err(err_too_large_deree("open", pp.degree(), got));
        }

        if cfg!(feature = "sanity-check") {
            assert_eq!(Self::commit(pp, poly).unwrap().0, comm.0);
            assert_eq!(poly.evaluate(point), *eval);
        }

        let t = num_vars.div_ceil(2);
        let b: usize = 1 << t; // Folding factor
        let f = UnivariatePolynomial::monomial(poly.evals().to_vec());

        // Compute `h(X)`
        let h = {
            let mut h = f.clone();
            for x_i in &point[..t] {
                let f_i_minus_one = h.coeffs();
                let mut f_i = Vec::with_capacity(f_i_minus_one.len() >> 1);
                merge_into(&mut f_i, f_i_minus_one, x_i, 1, 0);
                h = UnivariatePolynomial::monomial(f_i);
            }

            h
        };

        // Commit to `h(X)`
        let h_comm = UnivariateKzg::commit_and_write(pp, &h, transcript)?;

        // Compute `g(X) = f(X) (mod (X^b - \alpha))`
        let alpha = transcript.squeeze_challenge();
        let f_is = {
            let f_is = transpose(&f.coeffs().chunks(b).collect_vec());
            f_is.into_iter()
                .map(UnivariatePolynomial::monomial)
                .collect_vec()
        };
        let (g, q) = {
            // f(X) = ∑ X^i • f_i(X^b)
            //      = ∑ X^i • ((X^b - \alpha) • q_i(X^b) + f_i(\alpha))
            //      = (X^b - \alpha) • ∑ X^i • q_i(X^b)  + ∑ X^i • f_i(\alpha)
            //      = (X^b - \alpha) • q(X)              + g(X)
            let mut g_coeffs = vec![M::Scalar::ZERO; b];
            parallelize(&mut g_coeffs, |(g_coeffs, start)| {
                for (g_coeff, f_i) in g_coeffs.iter_mut().zip(f_is[start..].iter()) {
                    *g_coeff = f_i.evaluate(&alpha);
                }
            });
            let g = UnivariatePolynomial::monomial(g_coeffs);
            let mut q_is = vec![vec![]; b];
            parallelize(&mut q_is, |(q_is, start)| {
                for (q_i, f_i) in q_is.iter_mut().zip(f_is[start..].iter()) {
                    *q_i = horner_univariate_div(f_i.coeffs(), &alpha);
                }
            });
            let q_coeffs = transpose(&q_is.iter().map(|q_i| q_i.as_slice()).collect_vec())
                .into_iter()
                .flatten()
                .collect_vec();
            let q = UnivariatePolynomial::monomial(q_coeffs);
            (g, q)
        };

        // Commit to `g(X)`, and `q(X)`
        let comms = UnivariateKzg::<M>::batch_commit_and_write(pp, vec![&g, &q], transcript)?;
        let [g_comm, q_comm] = comms.try_into().unwrap();

        let gamma = transcript.squeeze_challenge();

        // IPA polynomial for `˜g(u_1), ˜h(u_2)`
        // X^{b-1} • (
        //      g(X) • P_{u_1}(1 / X) + g(1 / X) • P_{u_1}(X) +
        //      \gamma • (h(X) • P_{u_2}(1 / X) + h(1 / X) • P_{u_2}(X))
        // )
        let mut batched_ipa_poly = {
            let omega = root_of_unity(t + 1);
            // X^{b-1} • ( g(X) • P_{u_1}(1 / X) + g(1 / X) • P_{u_1}(X) )
            let mut lhs = {
                // g(X) • ( X^{b-1} • P_{u_1}(1 / X) ) + ( X^{b-1} • g(1 / X) ) • P_{u_1}(X)
                let p_u1 = MultilinearPolynomial::eq_xy(&point[..t]).into_evals();
                // fft ( X^{b-1} • P_{u_1}(1 / X) )
                let mut p_u1_inv_evals = p_u1.iter().rev().copied().collect_vec();
                p_u1_inv_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut p_u1_inv_evals, omega, t + 1);
                // fft ( g(X) )
                let mut g_evals = g.coeffs().to_vec();
                g_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut g_evals, omega, t + 1);
                // fft ( g(X) • ( X^{b-1} • P_{u_1}(1 / X) ) )
                let g_times_p_u1_inv = izip_eq!(g_evals, p_u1_inv_evals)
                    .map(|(g_eval, p_u1_inv_eval)| g_eval * p_u1_inv_eval)
                    .collect_vec();

                // fft ( X^{b-1} • g(1 / X) )
                let mut g_inv_evals = g.coeffs().iter().rev().copied().collect_vec();
                g_inv_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut g_inv_evals, omega, t + 1);
                // fft ( P_{u_1}(X) )
                let mut p_u1_evals = p_u1.clone();
                p_u1_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut p_u1_evals, omega, t + 1);
                // fft ( ( X^{b-1} • g(1 / X) ) • P_{u_1}(X) )
                let g_inv_times_p_u1 = izip_eq!(g_inv_evals, p_u1_evals)
                    .map(|(g_inv_eval, p_u1_eval)| g_inv_eval * p_u1_eval)
                    .collect_vec();

                izip_eq!(g_times_p_u1_inv, g_inv_times_p_u1)
                    .map(|(a, b)| a + b)
                    .collect_vec()
            };
            // X^{b-1} • ( h(X) • P_{u_2}(1 / X) + h(1 / X) • P_{u_2}(X) )
            let rhs = {
                // h(X) • ( X^{b-1} • P_{u_2}(1 / X) ) + ( X^{b-1} • h(1 / X) ) • P_{u_2}(X)
                let mut p_u2 = MultilinearPolynomial::eq_xy(&point[t..]).into_evals();
                p_u2.resize(b, M::Scalar::ZERO);
                // fft ( X^{b-1} • P_{u_2}(1 / X) )
                let mut p_u2_inv_evals = p_u2.iter().rev().copied().collect_vec();
                p_u2_inv_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut p_u2_inv_evals, omega, t + 1);
                // fft ( h(X) )
                let mut h_evals = h.coeffs().to_vec();
                h_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut h_evals, omega, t + 1);
                // fft ( h(X) • ( X^{b-1} • P_{u_1}(1 / X) ) )
                let h_times_p_u1_inv = izip_eq!(h_evals, p_u2_inv_evals)
                    .map(|(h_eval, p_u2_inv_eval)| h_eval * p_u2_inv_eval)
                    .collect_vec();

                // fft ( X^{b-1} • h(1 / X) )
                let mut h = h.coeffs().to_vec();
                h.resize(b, M::Scalar::ZERO);
                let mut h_inv_evals = h.iter().rev().copied().collect_vec();
                h_inv_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut h_inv_evals, omega, t + 1);
                // fft ( P_{u_2}(X) )
                let mut p_u2_evals = p_u2.clone();
                p_u2_evals.resize(2 * b, M::Scalar::ZERO);
                radix2_fft(&mut p_u2_evals, omega, t + 1);
                // fft ( ( X^{b-1} • h(1 / X) ) • P_{u_2}(X) )
                let h_inv_times_p_u2 = izip_eq!(h_inv_evals, p_u2_evals)
                    .map(|(h_inv_eval, p_u2_eval)| h_inv_eval * p_u2_eval)
                    .collect_vec();

                izip_eq!(h_times_p_u1_inv, h_inv_times_p_u2)
                    .map(|(a, b)| a + b)
                    .collect_vec()
            };
            lhs.iter_mut()
                .zip(rhs)
                .for_each(|(lhs, rhs)| *lhs += gamma * rhs);
            lhs
        };

        // `batched_ipa_poly` = X^{b-1} • (2(˜g(u_1) + \gamma • ˜h(u_2)) + X • s(X) + (1 / X) • s(1 / X))
        let s = {
            // size `2b` ifft for `batched_ipa_poly`
            let omega_inv = root_of_unity_inv(t + 1);
            let n_inv = M::Scalar::TWO_INV.pow_vartime([(t + 1) as u64]);
            radix2_fft(&mut batched_ipa_poly, omega_inv, t + 1);
            batched_ipa_poly
                .iter_mut()
                .for_each(|coeff| *coeff *= n_inv);
            UnivariatePolynomial::monomial(batched_ipa_poly[b..=2 * b - 2].to_vec())
        };

        // degree of `g(X)` <= b
        let d = {
            let d_coeffs = g.coeffs().iter().rev().copied().collect::<Vec<_>>();
            UnivariatePolynomial::monomial(d_coeffs)
        };

        // Commit to `s(X)`, and `d(X)`
        let comms = UnivariateKzg::<M>::batch_commit_and_write(pp, vec![&s, &d], transcript)?;
        let [s_comm, d_comm] = comms.try_into().unwrap();

        let zeta = transcript.squeeze_challenge();
        let zeta_inv = zeta.invert().unwrap();

        // f(X) - q(X) • (X^b - \alpha)
        let (phi_poly, phi_comm) = {
            let c = zeta.pow([b as u64]) - alpha;
            let phi_poly = &f - &q * &c;
            let phi_comm: M::G1Affine = (comm.0 - (q_comm.0 * c).into()).into();
            (phi_poly, phi_comm.into())
        };

        let polys = vec![&g, &h, &s, &d, &phi_poly];
        let comms = vec![&g_comm, &h_comm, &s_comm, &d_comm, &phi_comm];
        let points = vec![zeta, zeta_inv, alpha];
        let g_zeta = g.evaluate(&zeta);
        let evals = vec![
            Evaluation::new(0, 0, g_zeta),                // g(ζ)
            Evaluation::new(0, 1, g.evaluate(&zeta_inv)), // g(1/ζ)
            Evaluation::new(1, 0, h.evaluate(&zeta)),     // h(ζ)
            Evaluation::new(1, 1, h.evaluate(&zeta_inv)), // h(1/ζ)
            Evaluation::new(2, 0, s.evaluate(&zeta)),     // s(ζ)
            Evaluation::new(2, 1, s.evaluate(&zeta_inv)), // s(1/ζ)
            Evaluation::new(3, 0, d.evaluate(&zeta)),     // d(ζ)
            Evaluation::new(1, 2, h.evaluate(&alpha)),    // h(α)
            Evaluation::new(4, 0, g_zeta),                // φ(ζ)
        ];
        transcript.write_field_elements(evals[..6].iter().map(Evaluation::value))?;

        UnivariateKzg::batch_open(pp, polys, comms, &points, &evals, transcript)
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        let polys = polys.into_iter().collect_vec();
        let comms = comms.into_iter().collect_vec();
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        additive::batch_open::<_, Self>(pp, num_vars, polys, comms, points, evals, transcript)
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        UnivariateKzg::read_commitments(vp, num_polys, transcript)
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<M::Scalar, Self::Polynomial>,
        eval: &M::Scalar,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = point.len();
        let t = num_vars.div_ceil(2);

        let h_comm = transcript.read_commitment()?;

        let alpha = transcript.squeeze_challenge();
        let [g_comm, q_comm] = transcript.read_commitments(2)?.try_into().unwrap();

        let gamma = transcript.squeeze_challenge();
        let [s_comm, d_comm] = transcript.read_commitments(2)?.try_into().unwrap();

        let zeta = transcript.squeeze_challenge();
        let zeta_inv = zeta.invert().unwrap();

        // g(ζ), g(1/ζ), h(ζ), h(1/ζ), s(ζ), s(1/ζ)
        let evals = transcript.read_field_elements(6)?;

        let zeta_squares = squares(zeta).take(t).collect_vec();
        let zeta_inv_squares = squares(zeta_inv).take(t).collect_vec();
        let zeta_pow_to_b = zeta_squares.last().unwrap().square();
        let expected_d_zeta = { zeta_pow_to_b * zeta_inv * evals[1] };
        // h(α) = (g(ζ) P_{u_1}(1 / ζ) + g(1 / ζ) P_{u_1}(ζ) + γ • (h(ζ) P_{u_2}(1 / ζ) + h(1 / ζ) P_{u_2}(ζ) - 2 * v)
        //        - ζ * s(ζ) - (1 / ζ) * s(1 / ζ)) / 2
        let expected_h_alpha = {
            // P_{u_1}(ζ) = ∏_{i=0}^{t - 1} (u_i * ζ^{2^i} + (1 - u_i))
            let p_u1_zeta = {
                let mut acc = M::Scalar::ONE;
                for (i, u_i) in point[..t].iter().enumerate() {
                    acc *= *u_i * zeta_squares[i] + (M::Scalar::ONE - *u_i);
                }
                acc
            };
            // P_{u_1}(1 / ζ) = ∏_{i=0}^{t - 1} (u_i * (1 / ζ)^{2^i} + (1 - u_i))
            let p_u1_zeta_inv = {
                let mut acc = M::Scalar::ONE;
                for (i, u_i) in point[..t].iter().enumerate() {
                    acc *= *u_i * zeta_inv_squares[i] + (M::Scalar::ONE - *u_i);
                }
                acc
            };
            // P_{u_2}(ζ) = ∏_{i=t}^{n - 1} (u_i * ζ^{2^i} + (1 - u_i))
            let p_u2_zeta = {
                let mut acc = M::Scalar::ONE;
                for (i, u_i) in point[t..].iter().enumerate() {
                    acc *= *u_i * zeta_squares[i] + (M::Scalar::ONE - *u_i);
                }
                acc
            };
            // P_{u_2}(1 / ζ) = ∏_{i=t}^{n - 1} (u_i * (1 / ζ)^{2^i} + (1 - u_i))
            let p_u2_zeta_inv = {
                let mut acc = M::Scalar::ONE;
                for (i, u_i) in point[t..].iter().enumerate() {
                    acc *= *u_i * zeta_inv_squares[i] + (M::Scalar::ONE - *u_i);
                }
                acc
            };
            let acc = evals[0] * p_u1_zeta_inv
                + evals[1] * p_u1_zeta
                + gamma
                    * (evals[2] * p_u2_zeta_inv + evals[3] * p_u2_zeta
                        - M::Scalar::ONE.double() * eval);
            (acc - zeta * evals[4] - zeta_inv * evals[5]) * M::Scalar::TWO_INV
        };

        // opening check
        let phi_comm = comm.0 - (q_comm * (zeta_pow_to_b - alpha)).into();
        let comms = vec![g_comm, h_comm, s_comm, d_comm, phi_comm.into()]
            .into_iter()
            .map(UnivariateKzgCommitment)
            .collect_vec();
        let points = vec![zeta, zeta_inv, alpha];
        let evals = vec![
            Evaluation::new(0, 0, evals[0]),         // g(ζ)
            Evaluation::new(0, 1, evals[1]),         // g(1/ζ)
            Evaluation::new(1, 0, evals[2]),         // h(ζ)
            Evaluation::new(1, 1, evals[3]),         // h(1/ζ)
            Evaluation::new(2, 0, evals[4]),         // s(ζ)
            Evaluation::new(2, 1, evals[5]),         // s(1/ζ)
            Evaluation::new(3, 0, expected_d_zeta),  // d(ζ)
            Evaluation::new(1, 2, expected_h_alpha), // h(α)
            Evaluation::new(4, 0, evals[0]),         // φ(ζ)
        ];
        UnivariateKzg::batch_verify(vp, &comms, &points, &evals, transcript)?;
        Ok(())
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<M::Scalar, Self::Polynomial>],
        evals: &[Evaluation<M::Scalar>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, M::Scalar>,
    ) -> Result<(), Error> {
        let num_vars = points.first().map(|point| point.len()).unwrap_or_default();
        let comms = comms.into_iter().collect_vec();
        additive::batch_verify::<_, Self>(vp, num_vars, comms, points, evals, transcript)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            multilinear::mercury::Mercury,
            test::{run_batch_commit_open_verify, run_commit_open_verify},
            univariate::UnivariateKzg,
        },
        util::transcript::Keccak256Transcript,
    };
    use halo2_curves::bn256::Bn256;

    type Pcs = Mercury<UnivariateKzg<Bn256>>;

    #[test]
    fn commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }

    #[test]
    fn batch_commit_open_verify() {
        run_batch_commit_open_verify::<_, Pcs, Keccak256Transcript<_>>();
    }
}
