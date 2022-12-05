use crate::{
    util::{
        arithmetic::{fe_from_bytes_le, Coordinates, CurveAffine, MultiMillerLoop, PrimeField},
        Itertools,
    },
    Error,
};
use sha3::{Digest, Keccak256};
use std::{io, marker::PhantomData};

pub trait Transcript<F> {
    type Commitment;

    fn squeeze_challenge(&mut self) -> F;

    fn squeeze_n_challenges(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.squeeze_challenge()).collect()
    }

    fn common_commitment(&mut self, comm: &Self::Commitment) -> Result<(), Error>;

    fn common_scalar(&mut self, scalar: &F) -> Result<(), Error>;
}

pub trait TranscriptRead<F>: Transcript<F> {
    fn read_commitment(&mut self) -> Result<Self::Commitment, Error>;

    fn read_n_commitments(&mut self, n: usize) -> Result<Vec<Self::Commitment>, Error> {
        (0..n).map(|_| self.read_commitment()).collect()
    }

    fn read_scalar(&mut self) -> Result<F, Error>;

    fn read_n_scalars(&mut self, n: usize) -> Result<Vec<F>, Error> {
        (0..n).map(|_| self.read_scalar()).collect()
    }
}

pub trait TranscriptWrite<F>: Transcript<F> {
    fn write_commitment(&mut self, comm: Self::Commitment) -> Result<(), Error>;

    fn write_scalar(&mut self, scalar: F) -> Result<(), Error>;
}

#[derive(Debug)]
pub struct Keccak256Transcript<S, T> {
    buf: Vec<u8>,
    stream: S,
    _marker: PhantomData<T>,
}

impl<S, T> Keccak256Transcript<S, T> {
    pub fn new(stream: S) -> Self {
        Self {
            buf: Vec::new(),
            stream,
            _marker: PhantomData,
        }
    }
}

impl<T> Keccak256Transcript<Vec<u8>, T> {
    pub fn finalize(self) -> Vec<u8> {
        self.stream
    }
}

impl<M: MultiMillerLoop, S> Transcript<M::Scalar> for Keccak256Transcript<S, M> {
    type Commitment = M::G1Affine;

    fn squeeze_challenge(&mut self) -> M::Scalar {
        let empty_buf = self.buf.len() <= 0x20;
        let data = self
            .buf
            .drain(..)
            .chain(empty_buf.then_some(1))
            .collect_vec();
        self.buf = Keccak256::digest(data).to_vec();
        fe_from_bytes_le(&self.buf)
    }

    fn common_commitment(&mut self, comm: &Self::Commitment) -> Result<(), Error> {
        let coordinates = Option::<Coordinates<_>>::from(comm.coordinates()).ok_or_else(|| {
            Error::Transcript(
                io::ErrorKind::Other,
                "Cannot write points at infinity to the transcript".to_string(),
            )
        })?;
        self.buf.extend(
            [coordinates.x(), coordinates.y()]
                .map(PrimeField::to_repr)
                .iter()
                .flat_map(|repr| repr.as_ref().iter().rev())
                .cloned(),
        );
        Ok(())
    }

    fn common_scalar(&mut self, scalar: &M::Scalar) -> Result<(), Error> {
        self.buf
            .extend(scalar.to_repr().as_ref().iter().rev().cloned());
        Ok(())
    }
}

impl<M: MultiMillerLoop, R: io::Read> TranscriptRead<M::Scalar> for Keccak256Transcript<R, M> {
    fn read_commitment(&mut self) -> Result<Self::Commitment, Error> {
        let mut reprs = [<<M::G1Affine as CurveAffine>::Base as PrimeField>::Repr::default(); 2];
        for repr in &mut reprs {
            self.stream
                .read_exact(repr.as_mut())
                .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
            repr.as_mut().reverse();
        }
        let [x, y] =
            reprs.map(<<M::G1Affine as CurveAffine>::Base as PrimeField>::from_repr_vartime);
        let ec_point = x
            .zip(y)
            .and_then(|(x, y)| CurveAffine::from_xy(x, y).into())
            .ok_or_else(|| {
                Error::Transcript(
                    io::ErrorKind::Other,
                    "Invalid elliptic curve point encoding in proof".to_string(),
                )
            })?;
        self.common_commitment(&ec_point)?;
        Ok(ec_point)
    }

    fn read_scalar(&mut self) -> Result<M::Scalar, Error> {
        let mut repr = <M::Scalar as PrimeField>::Repr::default();
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        repr.as_mut().reverse();
        let scalar = M::Scalar::from_repr_vartime(repr).ok_or_else(|| {
            Error::Transcript(
                io::ErrorKind::Other,
                "Invalid scalar encoding in proof".to_string(),
            )
        })?;
        self.common_scalar(&scalar)?;
        Ok(scalar)
    }
}

impl<M: MultiMillerLoop, W: io::Write> TranscriptWrite<M::Scalar> for Keccak256Transcript<W, M> {
    fn write_commitment(&mut self, ec_point: Self::Commitment) -> Result<(), Error> {
        self.common_commitment(&ec_point)?;
        let coordinates = ec_point.coordinates().unwrap();
        for coordinate in [coordinates.x(), coordinates.y()] {
            let mut repr = coordinate.to_repr();
            repr.as_mut().reverse();
            self.stream
                .write_all(repr.as_ref())
                .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        }
        Ok(())
    }

    fn write_scalar(&mut self, scalar: M::Scalar) -> Result<(), Error> {
        self.common_scalar(&scalar)?;
        let mut repr = scalar.to_repr();
        repr.as_mut().reverse();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))
    }
}
