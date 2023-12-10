// @generated

pub mod onnx;
use crate::core::node::NodeBuilder;
use crate::core::tensor::Value;
use crate::core::tensor::F16;
use crate::core::ParserMut;
use crate::core::{Node, Parser, Tensor};
use crate::util::error::{LNError, LNResult};
pub use onnx::{ModelProto, TensorProto};
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

use onnx::tensor_proto::DataType;
use protobuf::Enum;

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
                DataType::FLOAT => Ok(Value::from_raw::<f32>(dim, self.take_raw_data())),
                DataType::UINT8 => Ok(Value::from_raw::<u8>(dim, self.take_raw_data())),
                DataType::INT8 => Ok(Value::from_raw::<i8>(dim, self.take_raw_data())),
                DataType::UINT16 => Ok(Value::from_raw::<u16>(dim, self.take_raw_data())),
                DataType::INT16 => Ok(Value::from_raw::<i16>(dim, self.take_raw_data())),
                DataType::INT32 => Ok(Value::from_raw::<i32>(dim, self.take_raw_data())),
                DataType::INT64 => Ok(Value::from_raw::<i64>(dim, self.take_raw_data())),
                DataType::STRING => Err(LNError::ParseOnnxFail("tensor string type not support")),

                DataType::BOOL => Ok(Value::from_raw::<bool>(dim, self.take_raw_data())),
                DataType::FLOAT16 => Ok(Value::from_raw::<F16>(dim, self.take_raw_data())),
                DataType::DOUBLE => Ok(Value::from_raw::<f64>(dim, self.take_raw_data())),
                DataType::UINT32 => Ok(Value::from_raw::<u32>(dim, self.take_raw_data())),
                DataType::UINT64 => Ok(Value::from_raw::<u64>(dim, self.take_raw_data())),
                _ => Err(LNError::ParseOnnxFail("tensor data type is none")),
            }
        } else {
            match t {
                DataType::UINT8 => Ok(Value::from_values::<u8>(
                    dim,
                    self.i32_data().cast_to::<u8>()?,
                )),
                DataType::UINT16 => Ok(Value::from_values::<u16>(
                    dim,
                    self.i32_data().cast_to::<u16>()?,
                )),
                DataType::INT8 => Ok(Value::from_values::<i8>(
                    dim,
                    self.i32_data().cast_to::<i8>()?,
                )),
                DataType::INT16 => Ok(Value::from_values::<i16>(
                    dim,
                    self.i32_data().cast_to::<i16>()?,
                )),
                DataType::INT32 => Ok(Value::from_values(dim, self.take_i32_data())),
                DataType::INT64 => Ok(Value::from_values(dim, self.take_i64_data())),
                DataType::FLOAT => Ok(Value::from_values(dim, self.take_f32_data())),
                DataType::DOUBLE => Ok(Value::from_values(dim, self.take_f64_data())),
                _ => unimplemented!("FIXME, tensor loading"),
            }
        }
    }
}

pub trait Cast {
    fn cast_to<T: TryFrom<i32>>(&self) -> Result<Vec<T>, T::Error>;
}

impl Cast for &[i32] {
    fn cast_to<T: TryFrom<i32>>(&self) -> Result<Vec<T>, T::Error> {
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
            .op(self.op_type())
            .weights(weights.into_boxed_slice())
            .inputs(inputs.into_boxed_slice())
            .outputs(output.into_boxed_slice())
            .build();
        Ok(node)
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
        let m = load("/opt/rsproject/gptgrep/lightnn/src/model/mnist-8.onnx").unwrap();

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
                "name:{}, input:{:?},ouput:{:?},op_type:{:?}",
                n.get_name(),
                n.input,
                n.output,
                n.get_op_type()
            );
        }
    }
}
