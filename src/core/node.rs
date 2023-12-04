use super::{op::Op, tensor::Tensors};
use crate::core::Tensor;
use crate::util::error::LNResult;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
pub(crate) struct Node {
    id: usize,
    name: Option<String>,
    op: Op,
    inputs: Box<[String]>,
    weights: Box<[Tensor]>,
    outputs: Box<[String]>,
}

impl Node {
    pub(crate) fn apply(&self, inputs: Tensors) -> LNResult<Tensors> {
        self.op.infer(inputs)
    }

    pub(crate) fn get_input(&self, value: &mut HashMap<String, Tensor>) -> LNResult<Tensors> {
        let inputs = self
            .inputs
            .iter()
            .map(|i| value.remove(i).unwrap())
            .collect::<Tensors>();
        Ok(inputs)
    }

    pub(crate) fn get_weight(&self, value: &HashMap<String, Tensor>) -> LNResult<Tensors> {
        let inputs = self
            .inputs
            .iter()
            .map(|i| value.get(i).unwrap().clone())
            .collect::<Tensors>();
        Ok(inputs)
    }
}

#[derive(Default)]
pub(crate) struct NodeBuilder {
    name: Option<String>,
    op: Op,
    weights: Box<[Tensor]>,
    inputs: Box<[String]>,
    outputs: Box<[String]>,
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

    pub fn weights(mut self, weights: Box<[Tensor]>) -> NodeBuilder {
        self.weights = weights;
        self
    }

    pub fn inputs(mut self, inputs: Box<[String]>) -> NodeBuilder {
        self.inputs = inputs;
        self
    }

    pub fn outputs(mut self, outputs: Box<[String]>) -> NodeBuilder {
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
