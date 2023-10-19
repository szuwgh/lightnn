use crate::tensor::{TenVec, Tensor};
use std::collections::HashMap;
pub type OpRegister = HashMap<&'static str, Box<dyn Op>>;
use std::vec;
pub fn register_ops(register: &mut OpRegister) {
    register.insert("Add", Box::new(Add::new()));
}

pub trait Op {
    fn infer(&self, inputs: TenVec) -> TenVec;
}

pub struct Add {}

impl Add {
    fn new() -> Self {
        Self {}
    }
}

impl Op for Add {
    fn infer(&self, inputs: TenVec) -> TenVec {
        let (a1, a2) = args_2(inputs);
        vec![a1 + a2]
    }
}

fn args_2(mut inputs: TenVec) -> (Tensor, Tensor) {
    if inputs.len() < 2 {
        panic!("tensor input smaller than 2")
    }
    let (a1, a2) = (inputs.pop().unwrap(), inputs.pop().unwrap());
    drop(inputs);
    (a1, a2)
}
