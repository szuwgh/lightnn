use super::tensor::{Tensor, Value};
use crate::core::tensor::Tensors;
use crate::util::error::LNResult;
use galois::shape::Dim;
use galois::DTensor;
use galois::Tensor as GTensor;
use galois::TensorType;
use galois::{Shape, TensorValue};
use smallvec::{smallvec, SmallVec};
#[derive(Debug)]
pub(crate) enum Op {
    Add,
    Reshape,
    Conv,
    Relu, //
    MaxPool(MaxPool2D),
    MatMul,
    Empty,
}

impl Default for Op {
    fn default() -> Self {
        Op::Empty
    }
}

impl Op {
    pub(crate) fn infer(&self, inputs: Tensors) -> LNResult<Tensors> {
        match self {
            Op::Add => Ok(smallvec![add(inputs)?]),
            Op::Reshape => Ok(smallvec![reshape(inputs)?]),
            Op::Relu => Ok(smallvec![relu(inputs)?]),
            Op::MaxPool(t) => Ok(smallvec![t.apply(inputs)?]),
            Op::MatMul => Ok(smallvec![reshape(inputs)?]),
            _ => {
                todo!()
            }
        }
    }
}

fn add(inputs: Tensors) -> LNResult<Tensor> {
    let (a1, a2) = args_2(inputs);
    let a3 = a1.as_value_ref().as_tensor() + a2.as_value_ref().as_tensor();
    Ok(Tensor::Own(Value(a3)))
}

fn reshape(inputs: Tensors) -> LNResult<Tensor> {
    let (a1, a2) = args_2(inputs);
    let shape = a2.as_value_ref().to_shape();
    Ok(Tensor::Own(Value(
        a1.as_value_ref()
            .as_tensor()
            .reshape(Shape::from_vec(shape)),
    )))
}

fn relu(inputs: Tensors) -> LNResult<Tensor> {
    let a1 = args_1(inputs);
    a1.as_value_ref().as_tensor().iter().for_each(|v| match v {
        TensorValue::U8(v) => {}
        TensorValue::U16(v) => {}
        TensorValue::U32(v) => {}
        TensorValue::U64(v) => {}
        TensorValue::I8(v) => {
            if *v < 0 {
                *v = 0 as _;
            }
        }
        TensorValue::I16(v) => {
            if *v < 0 {
                *v = 0 as _;
            }
        }
        TensorValue::I32(v) => {
            if *v < 0 {
                *v = 0 as _;
            }
        }
        TensorValue::I64(v) => {
            if *v < 0 as _ {
                *v = 0 as _;
            }
        }
        TensorValue::F16(v) => {
            // if *v < 0 as _ {
            //     *v = 0 as _;
            // }
        }
        TensorValue::F32(v) => {
            if *v < 0 as _ {
                *v = 0 as _;
            }
        }
        TensorValue::F64(v) => {
            if *v < 0 as _ {
                *v = 0 as _;
            }
        }
    });
    Ok(a1)
}

trait Map {
    fn f<T: TensorType>(&self, src: &[T], shape: &Dim) -> LNResult<Vec<T>>;

    fn map(&self, t: Tensor, new_dim: Shape) -> LNResult<GTensor> {
        let new_t = match t.as_value_ref().as_tensor() {
            GTensor::U8(t1) => GTensor::U8(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::I8(t1) => GTensor::I8(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::I16(t1) => GTensor::I16(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::U16(t1) => GTensor::U16(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::I32(t1) => GTensor::I32(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::U32(t1) => GTensor::U32(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::I64(t1) => GTensor::I64(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::U64(t1) => GTensor::U64(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::F16(t1) => GTensor::F16(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::F32(t1) => GTensor::F32(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
            GTensor::F64(t1) => GTensor::F64(DTensor::with_shape(
                self.f(t1.as_slice(), t1.dim())?,
                new_dim,
            )),
        };
        Ok(new_t)
    }
}

/// 在卷积神经网络（CNN）中，MaxPool2d等卷积操作通常需要四维的输入数据，这四个维度分别是：批量大小（batch size）、通道数（channels）、图像高度（height）和图像宽度（width），通常表示为NCHW12。

/// 批量大小（N）：这是一次处理的图像数量。通过一次处理多个图像，可以利用并行计算来提高效率。
/// 通道数（C）：对于彩色图像，通常有三个通道（红、绿、蓝）。对于灰度图像，只有一个通道。在网络的中间层，通道数可能会更大，如256，512，2024等2。
/// 图像高度（H）和图像宽度（W）：这是图像的尺寸。例如，一个28x28的图像的高度和宽度都是28
#[derive(Debug)]
pub(crate) struct MaxPool2D(pub(crate) (usize, usize), pub(crate) (usize, usize));

impl MaxPool2D {
    pub(crate) fn apply(&self, inputs: Tensors) -> LNResult<Tensor> {
        let a1 = args_1(inputs);
        let (kernel_size, stride) = (self.0, self.1);
        let (n, c, h, w) = a1.as_value_ref().as_tensor().shape().dims4();
        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;
        let new_shape = Shape::from_array([n, c, h_out, w_out]);
        Ok(Tensor::Own(Value(self.map(a1, new_shape)?)))
    }
}

impl Map for MaxPool2D {
    fn f<T: TensorType>(&self, src: &[T], shape: &Dim) -> LNResult<Vec<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        let (k_h, k_w) = self.0;
        let (s_h, s_w) = self.1;
        let (b_sz, c, h, w) = shape.shape().dims4();
        let stride = shape.stride();
        let (stride_h, stride_w) = (stride[2], stride[3]);
        let h_out = (h - k_h) / s_h + 1;
        let w_out = (w - k_w) / s_w + 1;
        let src_index = 0;
        let mut dst = vec![T::zero(); b_sz * c * h_out * w_out];
        for b_idx in 0..b_sz {
            let dst = &mut dst[b_idx * c * h_out * w_out..];
            let src_index = src_index + b_idx * stride[0];
            for c_idx in 0..c {
                let dst = &mut dst[c_idx * h_out * w_out..];
                let src_index = src_index + c_idx * stride[1];
                for h_idx in 0..h_out {
                    for w_idx in 0..w_out {
                        let mut largest =
                            src[src_index + s_h * h_idx * stride_h + s_w * w_idx * stride_w];
                        for m in 0..k_h {
                            for n in 0..k_w {
                                let m = s_h * h_idx + m;
                                let n = s_w * w_idx + n;
                                if largest < src[src_index + m * stride_h + n * stride_w] {
                                    largest = src[src_index + m * stride_h + n * stride_w]
                                }
                            }
                        }
                        dst[h_idx * w_out + w_idx] = largest;
                    }
                }
            }
        }
        Ok(dst)
    }
}

fn args_1(mut inputs: Tensors) -> Tensor {
    if inputs.len() < 1 {
        panic!("tensor input smaller than 2")
    }
    inputs.pop().unwrap()
}

fn args_2(mut inputs: Tensors) -> (Tensor, Tensor) {
    if inputs.len() < 2 {
        panic!("tensor input smaller than 2")
    }
    let (a2, a1) = (inputs.pop().unwrap(), inputs.pop().unwrap());
    drop(inputs);
    (a1, a2)
}

mod tests {
    use super::*;
    use galois::Tensor as GTensor;
    use galois::{arr, mat};
    #[test]
    fn op_add() {
        let mut inputs = Tensors::new();
        let a = Tensor::Own(Value(GTensor::I32(mat(&[
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]))));
        let b = Tensor::Own(Value(GTensor::I32(arr(&[1, 1, 1]))));

        inputs.push(a);
        inputs.push(b);
        println!("{:?}", inputs);

        let c = add(inputs).unwrap();
        println!("{:?}", c);
    }
}
