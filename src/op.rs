pub trait Op {
    fn eval(&self, inputs: VecLnnTensor);
}

pub struct Add {}

// impl Op for Add {
//     fn eval(&self) {}
// }
