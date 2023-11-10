use super::core::op::Op;
use super::onnx::*;
use super::util::error::LNResult;
use crate::core::tensor::{TenVec, Tensor};
use crate::core::Node;
use crate::core::Parser;
use std::collections::HashMap;
pub struct Session {
    m: ModelProto,
    param_tensor: Vec<Tensor>,
    nodes: Vec<Node>,
    tensors: HashMap<String, Tensor>, //统计所有的输入 权重也是当成输入
}

impl Session {
    pub fn load(m: ModelProto) -> LNResult<Session> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        //获取所有权重值
        for tp in m.get_graph().get_initializer() {
            let tensor = tp.parse()?;
            tensors.insert(tp.name().to_string(), tensor);
        }

        let mut nodes: Vec<Node> = Vec::new();

        for n in m.get_graph().get_node() {
            println!(
                "name:{}, input:{:?},op_type:{:?}",
                n.get_name(),
                n.input,
                n.get_op_type()
            );
        }
        todo!()
    }

    pub fn run(&self, input: TenVec) -> LNResult<()> {
        Ok(())
    }

    pub fn get_input_tensor(&self) -> TenVec {
        todo!()
    }
}

mod tests {

    use super::*;
    #[test]
    fn test_read_simple() {}
}
