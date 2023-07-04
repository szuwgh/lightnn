use galois::Tensor;

pub type VecLnnTensor = Vec<LnnTensor>;

pub enum LnnTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
}
