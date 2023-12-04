use rand::distributions::weighted::alias_method::Weight;

use super::core::op::Op;
use super::onnx::*;
use super::util::error::LNResult;
use crate::core::tensor::{Tensor, Tensors};
use crate::core::Node;
use crate::core::{Parser, ParserMut};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Model(Arc<Inner>);

struct Inner {
    nodes: Vec<Node>,
    input: String,
    initialize: HashMap<String, Tensor>, //统计所有的输入 权重也是当成输入
}

impl Inner {
    pub fn load(mut m: ModelProto) -> LNResult<Inner> {
        let mut initialize: HashMap<String, Tensor> = HashMap::new();
        //获取所有权重值
        for tp in m.get_graph().get_initializer() {
            let value = Arc::new(tp.parse()?);
            initialize.insert(tp.name().to_string(), Tensor::Share(value));
        }

        let mut nodes: Vec<Node> = Vec::new();
        for n in m.get_graph_mut().get_node_mut() {
            nodes.push(n.parse_mut(&initialize)?);
        }
        let input = m.get_graph_mut().get_input_mut()[0].take_name();
        drop(m);
        Ok(Inner {
            nodes,
            input,
            initialize,
        })
    }
}

impl Model {
    fn session(&self) -> LNResult<Session> {
        Ok(Session {
            values: HashMap::new(),
            model: self.clone(),
        })
    }
}

pub struct Session {
    values: HashMap<String, Tensor>,
    model: Model,
}

impl Session {
    pub fn set_input(&mut self, input: Tensor) {
        self.values.insert(self.model.0.input.clone(), input);
    }

    pub fn run(&mut self) -> LNResult<()> {
        for n in self.model.0.nodes.iter() {
            let mut inputs = n.get_input(&mut self.values)?;
            let mut weights = n.get_weight(&self.model.0.initialize)?;
            weights.pop().map(|v| inputs.push(v));
            drop(weights);
            let ouput = n.apply(inputs)?;
            //  self.values.insert(k, v)
        }
        Ok(())
    }
}

mod tests {

    use super::*;
    #[test]
    fn test_read_simple() {}
}
