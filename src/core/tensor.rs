use galois::Shape;
use galois::Tensor as GTensor;
use std::sync::Arc;
pub type F16 = half::f16;
use smallvec::SmallVec;

pub type Tensors = SmallVec<[Tensor; 4]>;

#[derive(Clone)]
pub enum Tensor {
    Own(Value),
    Share(Arc<Value>),
}

impl Tensor {
    pub(crate) fn as_value_ref(&self) -> &Value {
        match self {
            Tensor::Own(v) => &v,
            Tensor::Share(v) => v.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(GTensor<u8>),
    I8(GTensor<i8>),
    I16(GTensor<i16>),
    U16(GTensor<u16>),
    F16(GTensor<F16>),
    F32(GTensor<f32>),
    I32(GTensor<i32>),
    U32(GTensor<u32>),
    I64(GTensor<i64>),
    F64(GTensor<f64>),
    U64(GTensor<u64>),
    Bool(GTensor<bool>),
}

impl std::ops::Add<&Value> for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        let a = match (self, rhs) {
            (Value::U8(t1), Value::U8(t2)) => Value::U8(t1 + t2),
            (Value::I8(t1), Value::I8(t2)) => Value::I8(t1 + t2),
            (Value::I16(t1), Value::I16(t2)) => Value::I16(t1 + t2),
            (Value::U16(t1), Value::U16(t2)) => Value::U16(t1 + t2),
            (Value::F32(t1), Value::F32(t2)) => Value::F32(t1 + t2),
            (Value::I32(t1), Value::I32(t2)) => Value::I32(t1 + t2),
            (Value::U32(t1), Value::U32(t2)) => Value::U32(t1 + t2),
            (Value::I64(t1), Value::I64(t2)) => Value::I64(t1 + t2),
            (Value::F64(t1), Value::F64(t2)) => Value::F64(t1 + t2),
            (Value::U64(t1), Value::U64(t2)) => Value::U64(t1 + t2),
            _ => {
                panic!("types do not match");
            }
        };
        a
    }
}

impl std::ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        let a = match (self, rhs) {
            (Value::U8(t1), Value::U8(t2)) => Value::U8(t1 + t2),
            (Value::I8(t1), Value::I8(t2)) => Value::I8(t1 + t2),
            (Value::I16(t1), Value::I16(t2)) => Value::I16(t1 + t2),
            (Value::U16(t1), Value::U16(t2)) => Value::U16(t1 + t2),
            (Value::F32(t1), Value::F32(t2)) => Value::F32(t1 + t2),
            (Value::I32(t1), Value::I32(t2)) => Value::I32(t1 + t2),
            (Value::U32(t1), Value::U32(t2)) => Value::U32(t1 + t2),
            (Value::I64(t1), Value::I64(t2)) => Value::I64(t1 + t2),
            (Value::F64(t1), Value::F64(t2)) => Value::F64(t1 + t2),
            (Value::U64(t1), Value::U64(t2)) => Value::U64(t1 + t2),
            _ => {
                panic!("types do not match");
            }
        };
        a
    }
}

impl std::ops::Add<&Value> for Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Self::Output {
        let a = match (self, rhs) {
            (Value::U8(t1), Value::U8(t2)) => Value::U8(t1 + t2),
            (Value::I8(t1), Value::I8(t2)) => Value::I8(t1 + t2),
            (Value::I16(t1), Value::I16(t2)) => Value::I16(t1 + t2),
            (Value::U16(t1), Value::U16(t2)) => Value::U16(t1 + t2),
            (Value::F32(t1), Value::F32(t2)) => Value::F32(t1 + t2),
            (Value::I32(t1), Value::I32(t2)) => Value::I32(t1 + t2),
            (Value::U32(t1), Value::U32(t2)) => Value::U32(t1 + t2),
            (Value::I64(t1), Value::I64(t2)) => Value::I64(t1 + t2),
            (Value::F64(t1), Value::F64(t2)) => Value::F64(t1 + t2),
            (Value::U64(t1), Value::U64(t2)) => Value::U64(t1 + t2),
            _ => {
                panic!("types do not match");
            }
        };
        a
    }
}

impl Value {
    pub fn from_raw<T: TenType>(dim: &[usize], raw: &[u8]) -> Value {
        let t = unsafe {
            GTensor::<T>::from_raw_data(
                raw.as_ptr() as _,
                raw.len() / ::std::mem::size_of::<T>(),
                Shape::from_slice(dim),
            )
        };
        T::into_Value(t)
    }
}

pub trait TenType: Clone {
    fn into_Value(t: GTensor<Self>) -> Value;
}

macro_rules! Value_type {
    ($trt:ident, $mth:ident) => {
        impl TenType for $mth {
            fn into_Value(t: GTensor<Self>) -> Value {
                Value::$trt(t)
            }
        }
    };
}

Value_type!(U8, u8);
Value_type!(I8, i8);
Value_type!(I16, i16);
Value_type!(U16, u16);
Value_type!(F16, F16);
Value_type!(F32, f32);
Value_type!(I32, i32);
Value_type!(U32, u32);
Value_type!(I64, i64);
Value_type!(F64, f64);
Value_type!(U64, u64);
Value_type!(Bool, bool);
