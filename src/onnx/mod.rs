// @generated

pub mod onnx;
use crate::session::{Node, Parser, Session};
use crate::tensor::Tensor;
use crate::util::error::{LNError, LNResult};
pub use onnx::ModelProto;
use protobuf::{self, Message};
use std::path::Path;

//      "UNDEFINED":  0,
// 		"FLOAT":      1,
// 		"UINT8":      2,
// 		"INT8":       3,
// 		"UINT16":     4,
// 		"INT16":      5,
// 		"INT32":      6,
// 		"INT64":      7,
// 		"STRING":     8,
// 		"BOOL":       9,
// 		"FLOAT16":    10,
// 		"DOUBLE":     11,
// 		"UINT32":     12,
// 		"UINT64":     13,
// 		"COMPLEX64":  14,
// 		"COMPLEX128": 15,
// 		"BFLOAT16":   16,

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

impl Parser<Session> for onnx::ModelProto {
    fn parse(&self) -> LNResult<Session> {
        todo!()
    }
}

use ::protobuf::Enum;
use onnx::tensor_proto::DataType;

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
                DataType::UNDEFINED => {}
                DataType::FLOAT => {}
                DataType::UINT8 => {}
                DataType::INT8 => {}
                DataType::UINT16 => {}
                DataType::INT16 => {}
                DataType::INT32 => {}
                DataType::INT64 => {}
                DataType::STRING => {}
                DataType::BOOL => {}
                DataType::FLOAT16 => {}
                DataType::DOUBLE => {}
                DataType::UINT32 => {}
                DataType::UINT64 => {}
                _ => {}
            },
            None => {}
        }

        todo!()
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
