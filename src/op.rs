use crate::tensor::TensorVec;

pub trait Op {
    fn eval(&self, inputs: TensorVec);
}

pub struct Add {}

// impl Op for Add {
//     fn eval(&self) {}
// }
