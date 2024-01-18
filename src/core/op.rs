use super::tensor::{Tensor, Value};
use crate::core::tensor::Tensors;
use crate::util::error::LNError;
use crate::util::error::LNResult;
use galois::DTensor;
use galois::Tensor as GTensor;
use galois::TensorType;
use galois::{Shape, TensorValue};
use rayon::prelude::*;
use smallvec::smallvec;
#[derive(Debug)]
pub(crate) enum Op {
    Add,
    Reshape,
    Conv(Conv2D),
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
            Op::MaxPool(op) => Ok(smallvec![op.apply(inputs)?]),
            Op::MatMul => Ok(smallvec![reshape(inputs)?]),
            Op::Conv(op) => Ok(smallvec![op.apply(inputs)?]),
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

trait Map2 {
    const OP: &'static str;
    fn f<T: TensorType>(&self, t: &DTensor<T>, w: &DTensor<T>) -> LNResult<DTensor<T>>;

    fn out_shape<T: TensorType>(&self, t1: &DTensor<T>, t1: &DTensor<T>) -> LNResult<Shape>;

    fn map(&self, inp1: Tensor, inp2: Tensor) -> LNResult<GTensor> {
        let (t1, t2) = (
            inp1.as_value_ref().as_tensor(),
            inp2.as_value_ref().as_tensor(),
        );
        match (t1, t2) {
            (GTensor::U8(dt1), GTensor::U8(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::U8(self.f(dt1, dt2)?))
            }
            (GTensor::I8(dt1), GTensor::I8(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::I8(self.f(dt1, dt2)?))
            }
            (GTensor::I16(dt1), GTensor::I16(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::I16(self.f(dt1, dt2)?))
            }
            (GTensor::U16(dt1), GTensor::U16(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::U16(self.f(dt1, dt2)?))
            }
            (GTensor::I32(dt1), GTensor::I32(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::I32(self.f(dt1, dt2)?))
            }
            (GTensor::U32(dt1), GTensor::U32(dt2)) => {
                //let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::U32(self.f(dt1, dt2)?))
            }
            (GTensor::I64(dt1), GTensor::I64(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::I64(self.f(dt1, dt2)?))
            }
            (GTensor::U64(dt1), GTensor::U64(dt2)) => {
                //let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::U64(self.f(dt1, dt2)?))
            }
            (GTensor::F16(dt1), GTensor::F16(dt2)) => {
                //  let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::F16(self.f(dt1, dt2)?))
            }
            (GTensor::F32(dt1), GTensor::F32(dt2)) => {
                //  let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::F32(self.f(dt1, dt2)?))
            }
            (GTensor::F64(dt1), GTensor::F64(dt2)) => {
                // let new_dim = self.out_shape(dt1, dt2)?;
                Ok(GTensor::F64(self.f(dt1, dt2)?))
            }
            _ => Err(LNError::DTypeMismatchBinaryOp {
                lhs: t1.dtype(),
                rhs: t2.dtype(),
                op: Self::OP,
            }),
        }
    }
}

trait Map {
    fn f<T: TensorType>(&self, t: &DTensor<T>) -> LNResult<DTensor<T>>;

    fn out_shape<T: TensorType>(&self, t: &DTensor<T>) -> LNResult<Shape>;

    fn map(&self, t: Tensor) -> LNResult<GTensor> {
        let new_t = match t.as_value_ref().as_tensor() {
            GTensor::U8(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::U8(self.f(t1)?)
            }
            GTensor::I8(t1) => {
                //let new_dim = self.out_shape(t1)?;
                GTensor::I8(self.f(t1)?)
            }
            GTensor::I16(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::I16(self.f(t1)?)
            }
            GTensor::U16(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::U16(self.f(t1)?)
            }
            GTensor::I32(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::I32(self.f(t1)?)
            }
            GTensor::U32(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::U32(self.f(t1)?)
            }
            GTensor::I64(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::I64(self.f(t1)?)
            }
            GTensor::U64(t1) => {
                // let new_dim = self.out_shape(t1)?;
                GTensor::U64(self.f(t1)?)
            }
            GTensor::F16(t1) => {
                //  let new_dim = self.out_shape(t1)?;
                GTensor::F16(self.f(t1)?)
            }
            GTensor::F32(t1) => {
                //  let new_dim = self.out_shape(t1)?;
                GTensor::F32(self.f(t1)?)
            }
            GTensor::F64(t1) => {
                //  let new_dim = self.out_shape(t1)?;
                GTensor::F64(self.f(t1)?)
            }
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
        Ok(Tensor::Own(Value(self.map(a1)?)))
    }
}

impl Map for MaxPool2D {
    fn out_shape<T: TensorType>(&self, t: &DTensor<T>) -> LNResult<Shape> {
        let (kernel_size, stride) = (self.0, self.1);
        let (n, c, h, w) = t.shape().dims4();
        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;
        let new_shape = Shape::from_array([n, c, h_out, w_out]);
        Ok(new_shape)
    }

    fn f<T: TensorType>(&self, t: &DTensor<T>) -> LNResult<DTensor<T>> {
        let src = t.as_slice();
        let shape = t.dim();
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
        let new_dim = self.out_shape(t)?;
        Ok(DTensor::with_shape(dst, new_dim))
    }
}

pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug)]
pub(crate) struct Conv2D(pub(crate) Conv2DParam);

impl Map2 for Conv2D {
    const OP: &'static str = "conv2d";

    fn out_shape<T: TensorType>(&self, t1: &DTensor<T>, t2: &DTensor<T>) -> LNResult<Shape> {
        let (b_size, c_in, i_h, i_w) = t1.shape().dims4();
        let (c_out, c_in_k, k_h, k_w) = t2.shape().dims4();
        let groups = self.0.groups;
        if c_in != c_in_k * groups {
            panic!("in_channel mismatch between input")
        }
        let (p1, p2, p3, p4) = (self.0.pads.0, self.0.pads.1, self.0.pads.2, self.0.pads.3);
        let padding = if p1 != p2 || p1 != p3 || p1 != p4 {
            0
        } else {
            p1
        };
        let p = ParamsConv2D {
            b_size: b_size,
            i_h: i_h,
            i_w: i_w,
            k_h: k_h,
            k_w: k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding: padding,
            stride: self.0.stride,
            dilation: self.0.dilation,
        };
        let new_shape = p.out_dims();
        Ok(Shape::from_vec(new_shape))
    }

    fn f<T: TensorType>(&self, t1: &DTensor<T>, w: &DTensor<T>) -> LNResult<DTensor<T>> {
        let (p1, p2, p3, p4) = (self.0.pads.0, self.0.pads.1, self.0.pads.2, self.0.pads.3);
        let (padding, t) = if p1 != p2 || p1 != p3 || p1 != p4 {
            (0, t1.pad(2, p1, p3, T::zero())?.pad(3, p2, p4, T::zero())?)
        } else {
            (p1, t1.view())
        };
        let (b_size, c_in, i_h, i_w) = t.shape().dims4();
        let (c_out, c_in_k, k_h, k_w) = w.shape().dims4();
        let groups = self.0.groups;
        if c_in != c_in_k * groups {
            panic!("in_channel mismatch between input")
        }
        let p = ParamsConv2D {
            b_size: b_size,
            i_h: i_h,
            i_w: i_w,
            k_h: k_h,
            k_w: k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding: padding,
            stride: self.0.stride,
            dilation: self.0.dilation,
        };
        let new_dim = Shape::from_vec(p.out_dims());

        let (inp, inp_d, k, k_d) = (t.as_slice(), t.dim(), w.as_slice(), w.dim());

        let (inp_s0, inp_s1, inp_s2, inp_s3) = galois::shape::dims4(inp_d.stride());
        let (k_s0, k_s1, k_s2, k_s3) = galois::shape::dims4(k_d.stride());
        let (out_h, out_w) = (p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_h * out_w];

        // TODO: Avoid making this copy if `inp` already has the appropriate layout.
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_h * p.i_w];
        let cont_s0 = p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_w * p.c_in;
        let cont_s2 = p.c_in;
        for b_idx in 0..p.b_size {
            for h_idx in 0..p.i_h {
                for w_idx in 0..p.i_w {
                    for c_idx in 0..p.c_in {
                        let src_idx =
                            b_idx * inp_s0 + c_idx * inp_s1 + h_idx * inp_s2 + w_idx * inp_s3;
                        let dst_idx = b_idx * cont_s0 + h_idx * cont_s1 + w_idx * cont_s2 + c_idx;
                        inp_cont[dst_idx] = inp[src_idx]
                    }
                }
            }
        }

        for offset_h in 0..p.k_h {
            for offset_w in 0..p.k_w {
                (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                    let dst_idx = dst_c_idx * out_w * out_h;
                    let k_cont = (0..p.c_in)
                        .map(|c_in_idx| {
                            k[dst_c_idx * k_s0
                                + c_in_idx * k_s1
                                + offset_h * k_s2
                                + offset_w * k_s3]
                        })
                        .collect::<Vec<_>>();
                    for b_idx in 0..p.b_size {
                        let dst_idx = dst_idx + b_idx * p.c_out * out_h * out_w;
                        for dst_h in 0..out_h {
                            let dst_idx = dst_idx + dst_h * out_w;
                            let src_h = p.stride * dst_h + offset_h * p.dilation;
                            if src_h < p.padding || src_h >= p.i_h + p.padding {
                                continue;
                            }
                            let src_h = src_h - p.padding;
                            for dst_w in 0..out_w {
                                let dst_idx = dst_idx + dst_w;
                                let src_w = p.stride * dst_w + offset_w * p.dilation;
                                if src_w < p.padding || src_w >= p.i_w + p.padding {
                                    continue;
                                }
                                let src_w = src_w - p.padding;
                                let inp_cont = &inp_cont
                                    [b_idx * cont_s0 + src_h * cont_s1 + src_w * cont_s2..];
                                assert!(inp_cont.len() >= p.c_in);
                                assert!(k_cont.len() >= p.c_in);
                                let mut d = T::zero();
                                unsafe {
                                    T::vec_dot(inp_cont.as_ptr(), k_cont.as_ptr(), &mut d, p.c_in)
                                }
                                let dst_p = dst.as_ptr();
                                // Safety: dst_idx are uniques per dst_c_idx which is used to parallelise
                                // the different tasks so no two threads can try to write at the same
                                // location.
                                unsafe {
                                    let ptr = dst_p.add(dst_idx) as *mut T;
                                    *ptr += d
                                }
                            }
                        }
                    }
                });
            }
        }

        Ok(DTensor::with_shape(dst, new_dim))
    }
}

#[derive(Debug)]
pub(crate) struct Conv2DParam {
    pads: (usize, usize, usize, usize),
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl Conv2DParam {
    pub(crate) fn new(
        pads: (usize, usize, usize, usize),
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Conv2DParam {
        Self {
            pads,
            stride,
            dilation,
            groups,
        }
    }
}

// #[derive(Debug)]
// pub(crate) struct Conv2D(pub(crate) Conv2DParam);

impl Conv2D {
    pub(crate) fn apply(&self, inputs: Tensors) -> LNResult<Tensor> {
        let (a1, a2) = args_2(inputs);
        Ok(Tensor::Own(Value(self.map(a1, a2)?)))
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
