// @generated

pub mod onnx;
use crate::util::error::LNResult;
pub use onnx::ModelProto;
use protobuf::{self, Message};
use std::path::Path;

//导入onnx模型
pub fn load<P: AsRef<Path>>(path: P) -> LNResult<ModelProto> {
    let m = onnx::ModelProto::parse_from_bytes(&std::fs::read(path)?)?;
    Ok(m)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_read_model() {
        let m = load("/opt/rsproject/zeusgrep/lightnn/src/model/mnist-8.onnx").unwrap();

        println!("version:{}", m.ir_version.unwrap());
        for input in m.get_graph().get_input() {
            println!("input:{:?}", input);
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
