#![allow(clippy::op_ref)]

pub mod pcs;
pub mod plonk;
pub mod poly;
pub mod sum_check;
pub mod util;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidSumcheck(String),
    InvalidPcsParam(String),
    InvalidPcsProof(String),
    Serialization(String),
    Transcript(std::io::ErrorKind, String),
}
