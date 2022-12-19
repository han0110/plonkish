#![allow(clippy::op_ref)]

pub mod pcs;
pub mod piop;
pub mod poly;
pub mod snark;
pub mod util;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidSumcheck(String),
    InvalidPcsParam(String),
    InvalidPcsProof(String),
    Serialization(String),
    Transcript(std::io::ErrorKind, String),
}
