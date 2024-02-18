// @generated

pub mod onnx;
use crate::core::node::NodeBuilder;

use crate::core::op::MaxPool2D;
use crate::core::op::{Op, *};
use crate::core::tensor::Value;
use crate::core::ParserMut;
use crate::core::{Node, Tensor};
use crate::util::error::{LNError, LNResult};
use galois::DTensor;
use galois::Shape;
use galois::Tensor as GTensor;
use galois::TensorType;
use onnx::tensor_proto::DataType;
pub use onnx::{ModelProto, TensorProto};
use protobuf::Enum;
use protobuf::{self, Message};
use std::collections::HashMap;
use std::path::Path;

pub enum TensorDataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
}

//导入onnx模型
pub fn load<P: AsRef<Path>>(path: P) -> LNResult<ModelProto> {
    let m = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
    Ok(m)
}

impl ParserMut<Value> for onnx::TensorProto {
    fn parse_mut(&mut self) -> LNResult<Value> {
        let t = DataType::from_i32(
            self.data_type
                .ok_or(LNError::ParseOnnxFail("tensor data type is none"))?,
        )
        .ok_or(LNError::ParseOnnxFail("tensor data type not found"))?;
        let dim: Vec<usize> = self.dims.iter().map(|&i| i as usize).collect();

        if self.has_raw_data() {
            match t {
                DataType::UNDEFINED => Err(LNError::ParseOnnxFail("tensor data type is none")),
                DataType::FLOAT => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::F32,
                )),
                DataType::UINT8 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::U8,
                )),
                DataType::INT8 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::I8,
                )),
                DataType::UINT16 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::U16,
                )),
                DataType::INT16 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::I16,
                )),
                DataType::INT32 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::I32,
                )),
                DataType::INT64 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::I64,
                )),
                DataType::STRING => Err(LNError::ParseOnnxFail("tensor string type not support")),

                DataType::FLOAT16 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::F16,
                )),
                DataType::DOUBLE => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::F64,
                )),
                DataType::UINT32 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::U32,
                )),
                DataType::UINT64 => Ok(Value::from_raw(
                    dim,
                    self.take_raw_data(),
                    galois::DType::U64,
                )),
                _ => Err(LNError::ParseOnnxFail("tensor data type is none")),
            }
        } else {
            match t {
                DataType::UINT8 => Ok(Value(GTensor::U8(values_to_dtensor::<u8>(
                    dim,
                    self.i32_data().cast_to::<u8>()?,
                )))),
                DataType::UINT16 => Ok(Value(GTensor::U16(values_to_dtensor::<u16>(
                    dim,
                    self.i32_data().cast_to::<u16>()?,
                )))),
                DataType::INT8 => Ok(Value(GTensor::I8(values_to_dtensor::<i8>(
                    dim,
                    self.i32_data().cast_to::<i8>()?,
                )))),
                DataType::INT16 => Ok(Value(GTensor::I16(values_to_dtensor::<i16>(
                    dim,
                    self.i32_data().cast_to::<i16>()?,
                )))),
                DataType::INT32 => Ok(Value(GTensor::I32(values_to_dtensor::<i32>(
                    dim,
                    self.take_i32_data(),
                )))),
                DataType::INT64 => Ok(Value(GTensor::I64(values_to_dtensor::<i64>(
                    dim,
                    self.take_i64_data(),
                )))),
                DataType::FLOAT => Ok(Value(GTensor::F32(values_to_dtensor::<f32>(
                    dim,
                    self.take_f32_data(),
                )))),
                DataType::DOUBLE => Ok(Value(GTensor::F64(values_to_dtensor::<f64>(
                    dim,
                    self.take_f64_data(),
                )))),
                _ => unimplemented!("FIXME, tensor loading"),
            }
        }
    }
}

pub fn values_to_dtensor<T: TensorType>(dim: Vec<usize>, values: Vec<T>) -> DTensor<T> {
    let t = DTensor::<T>::with_shape(values, Shape::from_vec(dim));
    t
}

pub trait Cast<U> {
    fn cast_to<T: TryFrom<U>>(&self) -> Result<Vec<T>, T::Error>;
}

impl Cast<i32> for &[i32] {
    fn cast_to<T: TryFrom<i32>>(&self) -> Result<Vec<T>, T::Error> {
        let vec = self
            .iter()
            .map(|x| TryInto::<T>::try_into(*x))
            .collect::<Result<Vec<T>, T::Error>>();
        vec
    }
}

impl Cast<i64> for &[i64] {
    fn cast_to<T: TryFrom<i64>>(&self) -> Result<Vec<T>, T::Error> {
        let vec = self
            .iter()
            .map(|x| TryInto::<T>::try_into(*x))
            .collect::<Result<Vec<T>, T::Error>>();
        vec
    }
}

impl onnx::NodeProto {
    pub(crate) fn parse_mut(
        &mut self,
        initialize: &mut HashMap<String, Tensor>,
        values: &mut HashMap<String, usize>,
    ) -> LNResult<Node> {
        let mut weights: Vec<(String, Tensor)> = Vec::new();
        let mut inputs: Vec<usize> = Vec::new();
        let total_inputs = self.take_input();
        let name = self.take_name();
        for i in total_inputs.iter() {
            if initialize.contains_key(i) {
                //这个就是权重信息
                let w = initialize.remove(i).unwrap();
                if &name == "mobilenetv20_output_flatten0_reshape0" {
                    println!("w:{:?}", w.as_value_ref().as_tensor_ref());
                }
                weights.push((i.to_string(), w));
            } else {
                inputs.push(*values.get(i).unwrap());
            }
        }

        let output = self
            .take_output()
            .into_iter()
            .map(|i| {
                let id = values.len();
                values.insert(i, id);
                id
            })
            .collect::<Vec<usize>>();

        let node = NodeBuilder::default()
            .name(name)
            .op(self.get_op()?)
            .weights(weights.into_boxed_slice())
            .inputs(inputs.into_boxed_slice())
            .outputs(output.into_boxed_slice())
            .build();
        Ok(node)
    }

    pub(crate) fn get_op(&self) -> LNResult<Op> {
        let op = match self.op_type() {
            "Add" => Op::Add(Add::default()),
            "Reshape" => Op::Reshape(Reshape::default()),
            "Conv" => {
                let dilations = self.get_attr_pro::<[i64]>("dilations");
                let groups = self.get_attr_pro::<i64>("group").copied().unwrap_or(1);
                let _kernel_shape = self.get_attr_pro::<[i64]>("kernel_shape");
                let pads = self.get_attr_pro::<[i64]>("pads");
                let strides = self.get_attr_pro::<[i64]>("strides");
                let auto_pad = self.get_attr_pro::<str>("auto_pad");
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => panic!("unsupported auto_pad {s}"),
                };
                let pads = match pads {
                    Some([p]) => (*p as usize, *p as usize, *p as usize, *p as usize),
                    Some(&[p1, p2, p3, p4]) => {
                        let p1 = p1 as usize;
                        let p2 = p2 as usize;
                        let p3 = p3 as usize;
                        let p4 = p4 as usize;
                        (p1, p2, p3, p4)
                    }
                    Some(_) => {
                        panic!("more pads than expected in conv2d {pads:?} {}", self.name())
                    }
                    None => (0, 0, 0, 0),
                };
                let stride = match strides {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some([p1, p2]) => {
                        if p1 != p2 {
                            panic!(
                                "strides have to be the same on both axis {pads:?} {}",
                                self.name()
                            )
                        }
                        *p1 as usize
                    }
                    Some(s) => {
                        panic!("more strides than expected in conv2d {s:?} {}", self.name())
                    }
                };
                let dilation = match dilations {
                    None => 1,
                    Some([p]) => *p as usize,
                    Some([p1, p2]) => {
                        if p1 != p2 {
                            panic!(
                                "dilations have to be the same on both axis {pads:?} {}",
                                self.name()
                            )
                        }
                        *p1 as usize
                    }
                    Some(s) => {
                        panic!(
                            "more dilations than expected in conv2d {s:?} {}",
                            self.name()
                        )
                    }
                };

                Op::Conv(Conv2D(Conv2DParam::new(
                    pads,
                    stride,
                    dilation,
                    groups as usize,
                )))
            }
            "Relu" => Op::Relu(Relu::default()),
            "MaxPool" => {
                let kernel_shape = self
                    .get_attr_pro::<[i64]>("kernel_shape")
                    .ok_or("get kernel_shape attribute fail")?;

                let strides = self.get_attr_pro::<[i64]>("strides");

                let (k1, k2) = match kernel_shape {
                    &[k1, k2] => (k1 as usize, k2 as usize),
                    _ => panic!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}"),
                };

                let (s1, s2) = match strides {
                    None => (1, 1),
                    Some(a) => {
                        // let s = a.cast_to::<usize>()?;
                        if a.len() != 2 {
                            panic!("only 2d MaxPool is supported, strides {a:?}");
                        }
                        (a[0] as usize, a[1] as usize)
                    }
                };
                let max_pool = MaxPool2D((k1, k2), (s1, s2));
                Op::MaxPool(max_pool)
            }
            "MatMul" => Op::MatMul(MatMul::default()),
            "BatchNormalization" => {
                let training_mode = self.get_attr_pro::<i64>("training_mode");
                if training_mode.copied().unwrap_or(0) != 0 {
                    panic!("training mode is not supported for BatchNorm")
                }
                let eps = self.get_attr_pro::<f32>("epsilon").unwrap_or(&1e-5);
                Op::BatchNormalization(BatchNormalization::new(*eps))
            }
            "GlobalAveragePool" => Op::GlobalAvgPool2D(GlobalAvgPool2D::default()),
            _ => panic!("not suppert op node {}", self.op_type()),
        };
        Ok(op)
    }
}

#[cfg(test)]
mod tests {

    use super::Cast;
    use super::*;

    #[test]
    fn test_i32_to_u8() {
        let i32_arr = [1, 2, 3, 6];
        let i32_a = &i32_arr[..];
        let u8_arr = i32_a.cast_to::<u8>().unwrap();
        println!("{:?}", u8_arr);
    }

    #[test]
    fn test_read_simple() {
        let m = load("/opt/rsproject/gptgrep/lightnn/src/model/simple.onnx").unwrap();

        println!("version:{}", m.ir_version.unwrap());
        for input in m.get_graph().get_input() {
            println!("input name:{:?}", input.name());
            let typ_ = input.type_.0.as_ref().unwrap();
            let value = typ_.value.as_ref().unwrap();
            match value {
                onnx::type_proto::Value::TensorType(tensor) => {}
                _ => {}
            }
        }

        for tensor in m.get_graph().get_initializer() {
            println!("tensor name:{:?}", tensor.name());
            println!("tensor dim:{:?}", tensor.dims);
        }

        for n in m.get_graph().get_node() {
            println!(
                "name:{}, input:{:?},op_type:{:?}",
                n.get_name(),
                n.input,
                n.get_op_type()
            );
        }
    }

    #[test]
    fn test_read_model() {
        let m = load("/opt/rsproject/gptgrep/lightnn/model/mobilenetv2-7.onnx").unwrap();

        println!("version:{}", m.ir_version.unwrap());
        for input in m.get_graph().get_input() {
            println!("input name:{:?}", input.name());
            let typ_ = input.type_.0.as_ref().unwrap();
            let value = typ_.value.as_ref().unwrap();
            match value {
                onnx::type_proto::Value::TensorType(tensor) => {}
                _ => {}
            }
        }

        for output in m.get_graph().get_ouput() {
            println!("output name:{:?}", output.name());
            let typ_ = output.type_.0.as_ref().unwrap();
            let value = typ_.value.as_ref().unwrap();
            match value {
                onnx::type_proto::Value::TensorType(tensor) => {}
                _ => {}
            }
        }

        for tensor in m.get_graph().get_initializer() {
            println!("init tensor name:{:?}", tensor.name());
            println!("init tensor dim:{:?}", tensor.dims);
        }

        for n in m.get_graph().get_node() {
            println!(
                "name:{}\n input:{:?}\nouput:{:?}\nop_type:{:?}\nattr:{:?}\n",
                n.get_name(),
                n.input,
                n.output,
                n.get_op_type(),
                n.attribute,
            );
            println!("================================");
        }
    }
}

#[cfg(test)]
mod op_tests {
    use smallvec::smallvec;

    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;
    const TEST_DATA_PATH: &str = "./test_data/node";
    use crate::core::tensor::Tensors;

    fn input1(op: &str) -> (Tensor, Value) {
        let input0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_0.pb", op));
        let output0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/output_0.pb", op));

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::with_value((m.parse_mut().unwrap()));

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();
        println!("t3{:?}", t3.as_tensor_ref());
        (t1, t3)
    }

    fn input2(op: &str) -> (Tensor, Tensor, Value) {
        let input0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_0.pb", op));
        let input1 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_1.pb", op));
        let output0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/output_0.pb", op));

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(input1).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t2 = Tensor::with_value(m.parse_mut().unwrap());
        println!("t1{:?}", t1.as_value_ref().as_tensor_ref());
        println!("t2{:?}", t2.as_value_ref().as_tensor_ref());

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();
        println!("t3{:?}", t3.as_tensor_ref());

        (t1, t2, t3)
    }

    fn input5(op: &str) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Value) {
        let input0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_0.pb", op));
        let input1 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_1.pb", op));

        let input2 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_2.pb", op));
        let input3 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_3.pb", op));
        let input4 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/input_4.pb", op));

        let output0 =
            PathBuf::from(TEST_DATA_PATH).join(format!("{}/test_data_set_0/output_0.pb", op));

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(input1).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t2 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(input2).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(input3).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t4 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(input4).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t5 = Tensor::with_value(m.parse_mut().unwrap());

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let out: Value = m.parse_mut().unwrap();
        println!("t3{:?}", out.as_tensor_ref());

        (t1, t2, t3, t4, t5, out)
    }

    fn input2_infer(test_op: &str) {
        let (t1, t2, output) = input2(test_op);
        let model = PathBuf::from(TEST_DATA_PATH).join(format!("{}/{}", test_op, "model.onnx"));
        let file = File::open(model).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = ModelProto::parse_from_reader(&mut buf_reader).unwrap();
        let op = m.get_graph_mut().get_node_mut()[0].get_op().unwrap();
        let mut t3_output = op.infer(smallvec![t1.clone(), t2.clone()]).unwrap();
        let t33 = t3_output.pop().unwrap();
        println!("infer output{:?}", t33.as_value_ref().as_tensor_ref());
        assert!(*(t33.as_value_ref().as_tensor_ref()) == *output.as_tensor_ref());
        println!("pass {} success", test_op);
    }

    fn input5_infer(test_op: &str) {
        let (t1, t2, t3, t4, t5, output) = input5(test_op);
        let model = PathBuf::from(TEST_DATA_PATH).join(format!("{}/{}", test_op, "model.onnx"));
        let file = File::open(model).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = ModelProto::parse_from_reader(&mut buf_reader).unwrap();
        let op = m.get_graph_mut().get_node_mut()[0].get_op().unwrap();
        let mut t3_output = op
            .infer(smallvec![
                t1.clone(),
                t2.clone(),
                t3.clone(),
                t4.clone(),
                t5.clone()
            ])
            .unwrap();
        let t33 = t3_output.pop().unwrap();
        println!("infer output{:?}", t33.as_value_ref().as_tensor_ref());
        assert!(*(t33.as_value_ref().as_tensor_ref()) == *output.as_tensor_ref());
        println!("pass {} success", test_op);
    }

    fn input1_infer(test_op: &str) {
        let (t1, output) = input1(test_op);
        let model = PathBuf::from(TEST_DATA_PATH).join(format!("{}/{}", test_op, "model.onnx"));
        let file = File::open(model).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = ModelProto::parse_from_reader(&mut buf_reader).unwrap();
        let op = m.get_graph_mut().get_node_mut()[0].get_op().unwrap();
        let mut t3_output = op.infer(smallvec![t1.clone(),]).unwrap();
        let t33 = t3_output.pop().unwrap();
        println!("infer output{:?}", t33.as_value_ref().as_tensor_ref());
        assert!(*(t33.as_value_ref().as_tensor_ref()) == *output.as_tensor_ref());
        println!("pass {} success", test_op);
    }

    #[test]
    fn test_add() {
        input2_infer("test_add");
    }

    #[test]
    fn test_reshape_extended_dims() {
        input2_infer("test_reshape_extended_dims");
    }

    #[test]
    fn test_reshape_negative_dim() {
        input2_infer("test_reshape_negative_dim");
    }

    #[test]
    fn test_reshape_reduced_dims() {
        input2_infer("test_reshape_reduced_dims");
    }

    // #[test]
    // fn test_reshape_allowzero_reordered() {
    //     input2_infer("test_reshape_allowzero_reordered");
    // }

    #[test]
    fn test_reshape_one_dim() {
        input2_infer("test_reshape_one_dim");
    }

    #[test]
    fn test_relu() {
        input1_infer("test_relu");
    }

    #[test]
    fn test_maxpool_2d_default() {
        input1_infer("test_maxpool_2d_default");
    }

    #[test]
    fn test_maxpool_2d_strides() {
        input1_infer("test_maxpool_2d_strides");
    }

    #[test]
    fn test_conv_with_strides_and_asymmetric_padding() {
        input2_infer("test_conv_with_strides_and_asymmetric_padding");
    }

    #[test]
    fn test_conv_with_strides_no_padding() {
        input2_infer("test_conv_with_strides_no_padding");
    }

    // #[test]
    // fn test_conv_with_autopad_same() {
    //     input2_infer("test_conv_with_autopad_same");
    // }

    #[test]
    fn test_conv_with_strides_padding() {
        input2_infer("test_conv_with_strides_padding");
    }

    #[test]
    fn test_matmul_2d() {
        input2_infer("test_matmul_2d");
    }

    #[test]
    fn test_matmul_3d() {
        input2_infer("test_matmul_3d");
    }

    #[test]
    fn test_matmul_4d() {
        input2_infer("test_matmul_4d");
    }

    #[test]
    fn test_batchnorm_epsilon() {
        input5_infer("test_batchnorm_epsilon");
    }

    #[test]
    fn test_batchnorm_example() {
        input5_infer("test_batchnorm_example");
    }

    #[test]
    fn test_globalaveragepool() {
        input1_infer("test_globalaveragepool");
    }

    // #[test]
    // fn test_globalaveragepool() {
    //     input1_infer("test_globalaveragepool");
    // }
}
