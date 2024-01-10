use core::num::TryFromIntError;
use core::str::Utf8Error;
use protobuf::Error as ProtobufError;
use std::io;
use std::io::Error as IOError;
use thiserror::Error;
pub type LNResult<T> = Result<T, LNError>;
use galois::error::GError;
use galois::DType;
#[derive(Error, Debug)]
pub enum LNError {
    #[error("could not deserialize model: {0}")]
    ModelDeserializationError(#[from] ProtobufError),
    #[error("Unexpected io: {0}, {1}")]
    UnexpectIO(String, io::Error),
    #[error("Unexpected: {0}")]
    Unexpected(String),
    #[error("load onnx fail: {0}")]
    LoadOnnxFail(String),
    #[error("parse onnx model fail: {0}")]
    ParseOnnxFail(&'static str),
    #[error("convert fail: {0}")]
    ConvertFail(String),
    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },
    #[error("galois tensor error:: {0}")]
    GTensorError(GError),
}

impl From<&str> for LNError {
    fn from(e: &str) -> Self {
        LNError::Unexpected(e.to_string())
    }
}

impl From<(&str, io::Error)> for LNError {
    fn from(e: (&str, io::Error)) -> Self {
        LNError::UnexpectIO(e.0.to_string(), e.1)
    }
}

impl From<String> for LNError {
    fn from(e: String) -> Self {
        LNError::Unexpected(e)
    }
}

impl From<IOError> for LNError {
    fn from(e: IOError) -> Self {
        LNError::Unexpected(e.to_string())
    }
}

impl From<TryFromIntError> for LNError {
    fn from(e: TryFromIntError) -> Self {
        LNError::ConvertFail(e.to_string())
    }
}

impl From<LNError> for String {
    fn from(e: LNError) -> Self {
        format!("{}", e)
    }
}

impl From<Utf8Error> for LNError {
    fn from(e: Utf8Error) -> Self {
        LNError::Unexpected(e.to_string())
    }
}

impl From<GError> for LNError {
    fn from(e: GError) -> Self {
        LNError::GTensorError(e)
    }
}
