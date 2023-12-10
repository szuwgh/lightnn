use super::{op::Op, tensor::Tensors};
use crate::core::Tensor;
use crate::util::error::LNResult;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
pub(crate) struct Node {
    pub(crate) id: usize,
    pub(crate) name: Option<String>,
    pub(crate) op: Op,
    pub(crate) inputs: Box<[usize]>, //Box<[String]>, //Box<[usize]>,  //Box<[String]>,
    pub(crate) outputs: Box<[usize]>, //Box<[String]>, //Box<[usize]>, //Box<[String]>,
    pub(crate) weights: Box<[(String, Tensor)]>,
}

impl Node {
    pub(crate) fn apply(&self, inputs: Tensors) -> LNResult<Tensors> {
        self.op.infer(inputs)
    }

    pub(crate) fn get_input(&self, value: &mut Vec<Option<Tensor>>) -> LNResult<Tensors> {
        let inputs = self
            .inputs
            .iter()
            .map(|i| value.get(*i).unwrap().take().unwrap())
            .collect::<Tensors>();
        Ok(inputs)
    }

    pub(crate) fn get_weight(&self) -> &[(String, Tensor)] {
        &self.weights
    }

    pub(crate) fn get_output(&self) -> &[usize] {
        &self.outputs
    }
}

#[derive(Default)]
pub(crate) struct NodeBuilder {
    name: Option<String>,
    op: Op,
    weights: Box<[(String, Tensor)]>,
    inputs: Box<[usize]>,  //Box<[String]>,
    outputs: Box<[usize]>, //Box<[String]>,
}

impl NodeBuilder {
    pub fn name(mut self, n: String) -> NodeBuilder {
        self.name = Some(n);
        self
    }

    pub fn op(mut self, op_type: &str) -> NodeBuilder {
        self.op = Op::parse(op_type);
        self
    }

    pub fn weights(mut self, weights: Box<[(String, Tensor)]>) -> NodeBuilder {
        self.weights = weights;
        self
    }

    pub fn inputs(mut self, inputs: Box<[usize]>) -> NodeBuilder {
        self.inputs = inputs;
        self
    }

    pub fn outputs(mut self, outputs: Box<[usize]>) -> NodeBuilder {
        self.outputs = outputs;
        self
    }

    pub(crate) fn build(self) -> Node {
        Node {
            id: 0,
            name: self.name,
            op: self.op,
            inputs: self.inputs,
            weights: self.weights,
            outputs: self.outputs,
        }
    }
}
