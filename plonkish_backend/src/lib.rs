#![allow(clippy::op_ref)]

pub mod backend;
pub mod pcs;
pub mod piop;
pub mod poly;
pub mod util;

pub use halo2_curves;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidSumcheck(String),
    InvalidPcsParam(String),
    InvalidPcsOpen(String),
    InvalidSnark(String),
    Serialization(String),
    Transcript(std::io::ErrorKind, String),
}
