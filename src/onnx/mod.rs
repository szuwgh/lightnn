// @generated

pub mod onnx;
use crate::core::node::NodeBuilder;

use crate::core::op::MaxPool2D;
use crate::core::op::Op;
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
        for i in total_inputs.iter() {
            if initialize.contains_key(i) {
                //这个就是权重信息
                weights.push((i.to_string(), initialize.remove(i).unwrap()));
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
            .name(self.take_name())
            .op(self.get_op()?)
            .weights(weights.into_boxed_slice())
            .inputs(inputs.into_boxed_slice())
            .outputs(output.into_boxed_slice())
            .build();
        Ok(node)
    }

    pub(crate) fn get_op(&self) -> LNResult<Op> {
        let op = match self.op_type() {
            "Add" => Op::Add,
            "Reshape" => Op::Reshape,
            "Conv" => Op::Conv,
            "Relu" => Op::Relu,
            "MaxPool" => {
                let kernel_shape = self
                    .get_attr_pro("kernel_shape")
                    .ok_or("get kernel_shape attribute fail")?
                    .get_ints()
                    .cast_to::<usize>()?;

                let strides = self.get_attr_pro("strides");

                let (k1, k2) = match kernel_shape.as_slice() {
                    &[k1, k2] => (k1, k2),
                    _ => panic!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}"),
                };

                let (s1, s2) = match strides {
                    None => (1, 1),
                    Some(a) => {
                        let s = a.get_ints().cast_to::<usize>()?;
                        if s.len() != 2 {
                            panic!("only 2d MaxPool is supported, strides {s:?}");
                        }
                        (s[0], s[1])
                    }
                };
                let max_pool = MaxPool2D((k1, k2), (s1, s2));
                Op::MaxPool(max_pool)
            }
            "MatMul" => Op::MatMul,
            _ => Op::Empty,
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
        let i32_arr = [1, 2, 3, 6, 500];
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
        let m = load("/opt/rsproject/gptgrep/lightnn/model/mnist-8.onnx").unwrap();

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
    use crate::core::op::Op;
    use crate::core::tensor::Tensors;
    #[test]
    fn test_op_add() {
        let input0 = PathBuf::from(TEST_DATA_PATH).join("test_add/test_data_set_0/input_0.pb");
        let input1 = PathBuf::from(TEST_DATA_PATH).join("test_add/test_data_set_0/input_1.pb");
        let output0 = PathBuf::from(TEST_DATA_PATH).join("test_add/test_data_set_0/output_0.pb");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());

        let file = File::open(input1).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t2 = Tensor::Own(m.parse_mut().unwrap());

        // println!("t1:{:?}", t1.as_value_ref().as_tensor());
        // println!("t2:{:?}", t2.as_value_ref().as_tensor());

        let add_op: Op = Op::Add;
        let mut input = Tensors::new();
        input.push(t1);
        input.push(t2);
        let mut t3_output = add_op.infer(input).unwrap();

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();

        let t33 = t3_output.pop().unwrap();

        // println!("t3:{:?}\n", t3);
        // println!("t3_output:{:?}", *(t33.as_value_ref().as_tensor()));
        assert!(*(t33.as_value_ref().as_tensor()) == *t3.as_tensor());
        println!("pass add success")
    }

    #[test]
    fn test_op_reshape() {
        let input0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_reshape_extended_dims/test_data_set_0/input_0.pb");
        let input1 = PathBuf::from(TEST_DATA_PATH)
            .join("test_reshape_extended_dims/test_data_set_0/input_1.pb");

        let output0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_reshape_extended_dims/test_data_set_0/output_0.pb");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());

        let file = File::open(input1).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t2 = Tensor::Own(m.parse_mut().unwrap());

        let reshape_op: Op = Op::Reshape;
        let mut input = Tensors::new();
        input.push(t1);
        input.push(t2);
        //  let shape = t2.to_shape();
        let mut t3_output = reshape_op.infer(input).unwrap();

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();

        let t33 = t3_output.pop().unwrap();

        println!("t3:{:?}\n", t3);
        println!("t3_output:{:?}", t33.as_value_ref().as_tensor());
        assert!(*(t33.as_value_ref().as_tensor()) == *t3.as_tensor());
        println!("pass reshape success")
    }

    #[test]
    fn test_op_relu() {
        let input0 = PathBuf::from(TEST_DATA_PATH).join("test_relu/test_data_set_0/input_0.pb");
        let output0 = PathBuf::from(TEST_DATA_PATH).join("test_relu/test_data_set_0/output_0.pb");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());

        let relu_op: Op = Op::Relu;
        let mut input = Tensors::new();
        input.push(t1);

        let mut t3_output = relu_op.infer(input).unwrap();

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();
        let t33 = t3_output.pop().unwrap();

        println!("t3:{:?}", t3);
        println!("t3_output:{:?}", t33.as_value_ref().as_tensor());
        assert!(*(t33.as_value_ref().as_tensor()) == *t3.as_tensor());
        println!("pass relu success")
    }

    #[test]
    fn test_op_maxpool_1d() {
        let input0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_1d_default/test_data_set_0/input_0.pb");
        let output0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_1d_default/test_data_set_0/output_0.pb");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());
        println!("t1:{:?}\n", t1.as_value_ref().as_tensor());
        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();

        println!("t3:{:?}", t3);
        println!("pass relu success")
    }

    #[test]
    fn test_op_maxpool_2d_default() {
        let input0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_2d_default/test_data_set_0/input_0.pb");
        let output0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_2d_default/test_data_set_0/output_0.pb");

        let model = PathBuf::from(TEST_DATA_PATH).join("test_maxpool_2d_default/model.onnx");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);

        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());
        println!("t1:{:?}\n", t1);

        let file = File::open(model).unwrap();
        let mut buf_reader = BufReader::new(file);

        let mut m = ModelProto::parse_from_reader(&mut buf_reader).unwrap();

        let op = m.get_graph_mut().get_node_mut()[0].get_op().unwrap();

        let output = op.infer(smallvec![t1]).unwrap();

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();

        println!("t3:{:?}\n", t3);
        println!("t3_output:{:?}", output[0].as_value_ref().as_tensor());
        assert!(*(output[0].as_value_ref().as_tensor()) == *t3.as_tensor());
        println!("pass maxpool_2d_default success")
    }

    #[test]
    fn test_op_maxpool_2d_strides() {
        let input0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_2d_strides/test_data_set_0/input_0.pb");
        let output0 = PathBuf::from(TEST_DATA_PATH)
            .join("test_maxpool_2d_strides/test_data_set_0/output_0.pb");

        let model = PathBuf::from(TEST_DATA_PATH).join("test_maxpool_2d_strides/model.onnx");

        let file = File::open(input0).unwrap();
        let mut buf_reader = BufReader::new(file);

        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t1 = Tensor::Own(m.parse_mut().unwrap());
        println!("t1:{:?}\n", t1);

        let file = File::open(model).unwrap();
        let mut buf_reader = BufReader::new(file);

        let mut m = ModelProto::parse_from_reader(&mut buf_reader).unwrap();

        // let mut t3_output = relu_op.infer(input).unwrap();

        let op = m.get_graph_mut().get_node_mut()[0].get_op().unwrap();

        let output = op.infer(smallvec![t1]).unwrap();

        let file = File::open(output0).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut m = TensorProto::parse_from_reader(&mut buf_reader).unwrap();
        let t3: Value = m.parse_mut().unwrap();
        //    let t33 = t3_output.pop().unwrap();

        println!("t3:{:?}\n", t3);
        println!("t3_output:{:?}", output[0].as_value_ref().as_tensor());
        assert!(*(output[0].as_value_ref().as_tensor()) == *t3.as_tensor());
        println!("pass maxpool_2d_strides success")
    }
}
