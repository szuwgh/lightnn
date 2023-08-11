use super::onnx::*;
use super::op::Op;
use super::tensor::Tensor;
use super::util::error::LNResult;

pub struct Session {
    m: ModelProto,
    param_tensor: Vec<Tensor>,
    nodes: Vec<Node>,
}

impl Session {
    pub fn load(m: ModelProto) -> LNResult<Session> {
        todo!()
    }

    pub fn run(&self) -> LNResult<()> {
        Ok(())
    }
}

pub(crate) struct Node {
    inputs: Vec<String>,
    name: String,
    op: Box<dyn Op>,
    outputs: Vec<String>,
}

pub trait Parser<T> {
    fn parse(&self) -> LNResult<T>;
}
