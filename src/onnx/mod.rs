// @generated

pub mod onnx;
use crate::core::node::NodeBuilder;
use crate::core::tensor::Value;
use crate::core::tensor::F16;
use crate::core::{Node, Parser, ParserMut, Tensor};
use crate::util::error::{LNError, LNResult};
pub use onnx::{ModelProto, TensorProto};
use protobuf::{self, Message};
use rand::distributions::weighted;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
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

use ::protobuf::Enum;
use onnx::tensor_proto::DataType;

impl Parser<Value> for onnx::TensorProto {
    fn parse(&self) -> LNResult<Value> {
        let t = DataType::from_i32(
            self.data_type
                .ok_or(LNError::ParseOnnxFail("tensor data type is none"))?,
        )
        .ok_or(LNError::ParseOnnxFail("tensor data type not found"))?;
        let dim: Vec<usize> = self.dims.iter().map(|&i| i as usize).collect();
        match &self.raw_data {
            Some(raw_data) => match t {
                DataType::UNDEFINED => Err(LNError::ParseOnnxFail("tensor data type is none")),
                DataType::FLOAT => Ok(Value::from_raw::<f32>(&dim, raw_data)),
                DataType::UINT8 => Ok(Value::from_raw::<u8>(&dim, raw_data)),
                DataType::INT8 => Ok(Value::from_raw::<i8>(&dim, raw_data)),
                DataType::UINT16 => Ok(Value::from_raw::<u16>(&dim, raw_data)),
                DataType::INT16 => Ok(Value::from_raw::<i16>(&dim, raw_data)),
                DataType::INT32 => Ok(Value::from_raw::<i32>(&dim, raw_data)),
                DataType::INT64 => Ok(Value::from_raw::<i64>(&dim, raw_data)),
                DataType::STRING => Err(LNError::ParseOnnxFail("tensor string type not support")),
                DataType::BOOL => Ok(Value::from_raw::<bool>(&dim, raw_data)),
                DataType::FLOAT16 => Ok(Value::from_raw::<F16>(&dim, raw_data)),
                DataType::DOUBLE => Ok(Value::from_raw::<f64>(&dim, raw_data)),
                DataType::UINT32 => Ok(Value::from_raw::<u32>(&dim, raw_data)),
                DataType::UINT64 => Ok(Value::from_raw::<u64>(&dim, raw_data)),
                _ => Err(LNError::ParseOnnxFail("tensor data type is none")),
            },
            None => Err(LNError::ParseOnnxFail("tensor data type is none")),
        }
    }
}

impl onnx::NodeProto {
    pub(crate) fn parse_mut(&mut self, initialize: &HashMap<String, Tensor>) -> LNResult<Node> {
        let mut weights: Vec<Tensor> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        let total_inputs = self.take_input();
        for i in total_inputs.iter() {
            if initialize.contains_key(i) {
                //那么这个就是权重信息
                weights.push(initialize.get(i).unwrap().clone());
            } else {
                inputs.push(i.to_string());
            }
        }
        let node = NodeBuilder::default()
            .name(self.take_name())
            .op(self.op_type())
            .weights(weights.into_boxed_slice())
            .inputs(inputs.into_boxed_slice())
            .outputs(self.take_output())
            .build();
        Ok(node)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

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
