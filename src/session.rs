use super::onnx::*;
use super::op::Op;
use super::tensor::Tensor;
use super::util::error::LNResult;
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

    pub fn run(&self) -> LNResult<()> {
        Ok(())
    }
}

pub(crate) struct Node {
    name: String,
    op: Box<dyn Op>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

pub trait Parser<T> {
    fn parse(&self) -> LNResult<T>;
}

mod tests {

    use super::*;
    #[test]
    fn test_read_simple() {}
}
