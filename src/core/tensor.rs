use galois::Shape;
use galois::{DType, Tensor as GTensor};
use std::fmt::Debug;

use std::sync::Arc;
pub type F16 = half::f16;
use smallvec::SmallVec;

pub type Tensors = SmallVec<[Tensor; 4]>;

#[derive(Clone)]
pub enum Tensor {
    Own(Value),
    Share(Arc<Value>),
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Tensor::Own(_) => f.write_str("own"),
            Tensor::Share(_) => f.write_str("share"),
        }
    }
}

impl Tensor {
    pub fn new(t: GTensor) -> Self {
        Tensor::Own(Value(t))
    }

    pub fn as_value_ref(&self) -> &Value {
        match self {
            Tensor::Own(v) => &v,
            Tensor::Share(v) => v.as_ref(),
        }
    }

    pub fn as_value(self) -> Value {
        match self {
            Tensor::Own(v) => v,
            Tensor::Share(_) => panic!("not get share value"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value(pub(crate) GTensor);

impl Value {
    pub fn as_tensor_ref(&self) -> &GTensor {
        &self.0
    }

    pub fn as_tensor(self) -> GTensor {
        self.0
    }

    pub fn from_raw(dim: Vec<usize>, raw: Vec<u8>, d: DType) -> Value {
        let t = GTensor::from_raw_data(raw.as_ptr() as _, raw.len(), Shape::from_vec(dim), d);
        ::std::mem::forget(raw);
        Value(t)
    }

    pub(crate) fn to_shape(&self) -> Vec<usize> {
        return match &self.0 {
            GTensor::U8(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::I8(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::I16(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::U16(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::I32(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::U32(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::I64(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            GTensor::U64(t1) => t1.as_slice().iter().map(|e| *e as usize).collect(),
            _ => Vec::new(),
        };
    }
}
