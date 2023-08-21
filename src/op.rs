use crate::tensor::TensorVec;
use std::collections::HashMap;
pub type OpRegister = HashMap<&'static str, Box<dyn Op>>;

pub trait Op {
    fn eval(&self, inputs: TensorVec);
}

pub struct Add {}

// impl Op for Add {
//     fn eval(&self) {}
// }
