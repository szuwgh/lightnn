use super::tensor::{TenVec, Tensor};
use crate::util::error::LNResult;
use std::collections::HashMap;
use std::sync::Arc;
use std::vec;
pub enum Op {
    Add,
    Empty,
}

impl Default for Op {
    fn default() -> Self {
        Op::Empty
    }
}

impl Op {
    pub(crate) fn infer(&self, inputs: TenVec) -> LNResult<TenVec> {
        match self {
            Op::Add => add(inputs),
            _ => {
                todo!()
            }
        }
    }

    pub fn parse(s: &str) -> Op {
        todo!()
    }
}

fn add(inputs: TenVec) -> LNResult<TenVec> {
    let (a1, a2) = args_2(inputs);
    Ok(vec![a1 + a2])
}

fn args_2(mut inputs: TenVec) -> (Tensor, Tensor) {
    if inputs.len() < 2 {
        panic!("tensor input smaller than 2")
    }
    let (a1, a2) = (inputs.pop().unwrap(), inputs.pop().unwrap());
    drop(inputs);
    (a1, a2)
}
