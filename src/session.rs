use super::onnx::*;
use super::op::Op;
use super::tensor::Tensor;
use super::util::error::LNResult;
use std::collections::HashMap;

pub struct Session {
    m: ModelProto,
    param_tensor: Vec<Tensor>,
    nodes: Vec<Node>,
    tensors: HashMap<String, Tensor>, //统计所有的输入 权重也是当成输入
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
