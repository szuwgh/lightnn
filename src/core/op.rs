use ndarray::iter::Windows;

use super::tensor::{Tensor, Value};
use crate::core::tensor::Tensors;
use crate::util::error::LNResult;
use smallvec::SmallVec;

#[derive(Debug)]
pub enum Op {
    Add,
    Reshape,
    Conv,
    Relu,
    MaxPool,
    MatMul,
    Empty,
}

impl Default for Op {
    fn default() -> Self {
        Op::Empty
    }
}

impl Op {
    pub(crate) fn parse(s: &str) -> Self {
        match s {
            "Add" => Op::Add,
            "Reshape" => Op::Reshape,
            "Conv" => Op::Conv,
            "Relu" => Op::Relu,
            "MaxPool" => Op::MaxPool,
            "MatMul" => Op::MatMul,
            _ => Op::Empty,
        }
    }

    pub(crate) fn infer(&self, inputs: Tensors) -> LNResult<Tensors> {
        match self {
            Op::Add => Ok(Tensors::from_elem(Tensor::Own(add(inputs)?), 1)),
            _ => {
                todo!()
            }
        }
    }
}

fn add(inputs: Tensors) -> LNResult<Value> {
    let (a1, a2) = args_2(inputs);
    Ok(a1.as_value_ref() + a2.as_value_ref())
}

fn args_2(mut inputs: Tensors) -> (Tensor, Tensor) {
    if inputs.len() < 2 {
        panic!("tensor input smaller than 2")
    }
    let (a1, a2) = (inputs.pop().unwrap(), inputs.pop().unwrap());
    drop(inputs);
    (a1, a2)
}
