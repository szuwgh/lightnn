use super::onnx::f16;
use galois::{Shape, Tensor as GTensor};

pub type TenVec = Vec<Tensor>;

#[derive(Debug)]
pub enum Tensor {
    U8(GTensor<u8>),
    I8(GTensor<i8>),
    I16(GTensor<i16>),
    U16(GTensor<u16>),
    F16(GTensor<f16>),
    F32(GTensor<f32>),
    I32(GTensor<i32>),
    U32(GTensor<u32>),
    I64(GTensor<i64>),
    F64(GTensor<f64>),
    U64(GTensor<u64>),
    Bool(GTensor<bool>),
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let a = match (self, rhs) {
            (Tensor::U8(t1), Tensor::U8(t2)) => Tensor::U8(t1 + t2),
            (Tensor::I8(t1), Tensor::I8(t2)) => Tensor::I8(t1 + t2),
            (Tensor::I16(t1), Tensor::I16(t2)) => Tensor::I16(t1 + t2),
            (Tensor::U16(t1), Tensor::U16(t2)) => Tensor::U16(t1 + t2),
            (Tensor::F32(t1), Tensor::F32(t2)) => Tensor::F32(t1 + t2),
            (Tensor::I32(t1), Tensor::I32(t2)) => Tensor::I32(t1 + t2),
            (Tensor::U32(t1), Tensor::U32(t2)) => Tensor::U32(t1 + t2),
            (Tensor::I64(t1), Tensor::I64(t2)) => Tensor::I64(t1 + t2),
            (Tensor::F64(t1), Tensor::F64(t2)) => Tensor::F64(t1 + t2),
            (Tensor::U64(t1), Tensor::U64(t2)) => Tensor::U64(t1 + t2),
            _ => {
                panic!("types do not match");
            }
        };
        a
    }
}

impl Tensor {
    pub fn from_raw<T: TenType>(dim: &[usize], raw: &[u8]) -> Tensor {
        let t = unsafe {
            GTensor::<T>::from_raw_data(
                raw.as_ptr() as _,
                raw.len() / ::std::mem::size_of::<T>(),
                Shape::from_slice(dim),
            )
        };
        T::into_tensor(t)
    }
}

pub trait TenType: Clone {
    fn into_tensor(t: GTensor<Self>) -> Tensor;
}

macro_rules! tensor_type {
    ($trt:ident, $mth:ident) => {
        impl TenType for $mth {
            fn into_tensor(t: GTensor<Self>) -> Tensor {
                Tensor::$trt(t)
            }
        }
    };
}

//      U8(GTensor<u8>),
//     I8(GTensor<i8>),
//     I16(GTensor<i16>),
//     U16(GTensor<u16>),
//     F32(GTensor<f32>),
//     I32(GTensor<i32>),
//     U32(GTensor<u32>),
//     I64(GTensor<i64>),
//     F64(GTensor<f64>),
//     U64(GTensor<u64>),
tensor_type!(U8, u8);
tensor_type!(I8, i8);
tensor_type!(I16, i16);
tensor_type!(U16, u16);
tensor_type!(F16, f16);
tensor_type!(F32, f32);
tensor_type!(I32, i32);
tensor_type!(U32, u32);
tensor_type!(I64, i64);
tensor_type!(F64, f64);
tensor_type!(U64, u64);
tensor_type!(Bool, bool);
