use galois::Tensor as GTensor;

pub type TensorVec = Vec<Tensor>;

pub enum Tensor {
    F32(GTensor<f32>),
    F64(GTensor<f64>),
    I8(GTensor<i8>),
    I16(GTensor<i16>),
    I32(GTensor<i32>),
    I64(GTensor<i64>),
}

impl Tensor {
    pub unsafe fn from_raw<T: TenType>(dim: &[usize], raw: &[u8]) -> Tensor {
        let value: Vec<T> =
            ::std::slice::from_raw_parts(raw.as_ptr() as _, raw.len() / ::std::mem::size_of::<T>())
                .to_vec();
        todo!()
        //let t = GTensor::<T>::from_vec(value, dim);
    }
}

pub trait TenType: Clone {
    fn into_tensor(t: GTensor<Self>) -> Tensor;
}
