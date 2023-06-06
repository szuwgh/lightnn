// @generated

pub mod onnx;
use crate::util::error::LNResult;
use protobuf::{self, Message};
use std::path::Path;

pub fn load<P: AsRef<Path>>(path: P) -> LNResult<onnx::ModelProto> {
    let m = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
    Ok(m)
}
