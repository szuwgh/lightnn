// @generated

pub mod onnx;
use crate::core::tensor::F16;
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

// impl Parser<Session> for onnx::ModelProto {
//     fn parse(&self) -> LNResult<Session> {
//         todo!()
//     }
// }

use ::protobuf::Enum;
use onnx::tensor_proto::DataType;

impl Parser<HashMap<String, Tensor>> for Vec<TensorProto> {
    fn parse(&self) -> LNResult<HashMap<String, Tensor>> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for t in self.iter() {
            let tensor = t.parse()?;
            map.insert(t.name().to_string(), tensor);
        }
        Ok(map)
    }
}

impl Parser<Tensor> for onnx::TensorProto {
    fn parse(&self) -> LNResult<Tensor> {
        let t = DataType::from_i32(
            self.data_type
                .ok_or(LNError::ParseOnnxFail("tensor data type is none"))?,
        )
        .ok_or(LNError::ParseOnnxFail("tensor data type not found"))?;
        let dim: Vec<usize> = self.dims.iter().map(|&i| i as usize).collect();
        match &self.raw_data {
            Some(raw_data) => match t {
                DataType::UNDEFINED => Err(LNError::ParseOnnxFail("tensor data type is none")),
                DataType::FLOAT => Ok(Tensor::from_raw::<f32>(&dim, raw_data)),
                DataType::UINT8 => Ok(Tensor::from_raw::<u8>(&dim, raw_data)),
                DataType::INT8 => Ok(Tensor::from_raw::<i8>(&dim, raw_data)),
                DataType::UINT16 => Ok(Tensor::from_raw::<u16>(&dim, raw_data)),
                DataType::INT16 => Ok(Tensor::from_raw::<i16>(&dim, raw_data)),
                DataType::INT32 => Ok(Tensor::from_raw::<i32>(&dim, raw_data)),
                DataType::INT64 => Ok(Tensor::from_raw::<i64>(&dim, raw_data)),
                DataType::STRING => Err(LNError::ParseOnnxFail("tensor string type not support")),
                DataType::BOOL => Ok(Tensor::from_raw::<bool>(&dim, raw_data)),
                DataType::FLOAT16 => Ok(Tensor::from_raw::<F16>(&dim, raw_data)),
                DataType::DOUBLE => Ok(Tensor::from_raw::<f64>(&dim, raw_data)),
                DataType::UINT32 => Ok(Tensor::from_raw::<u32>(&dim, raw_data)),
                DataType::UINT64 => Ok(Tensor::from_raw::<u64>(&dim, raw_data)),
                _ => Err(LNError::ParseOnnxFail("tensor data type is none")),
            },
            None => Err(LNError::ParseOnnxFail("tensor data type is none")),
        }
    }
}

impl Parser<Node> for onnx::NodeProto {
    fn parse(&self) -> LNResult<Node> {
        todo!()
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
}
