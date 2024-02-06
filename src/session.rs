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

impl Model {
    pub fn load(mut m: ModelProto) -> LNResult<Model> {
        let inner = Arc::new(Inner::load(m)?);
        Ok(Model(inner))
    }
}

struct Inner {
    //  m: ModelProto,
    nodes: Vec<Node>,
    //  initialize: HashMap<String, Tensor>, //统计所有的输入 权重也是当成输入
    input: Box<[usize]>,

    output: Box<[usize]>,

    values_len: usize,
}

impl Inner {
    pub fn load(mut m: ModelProto) -> LNResult<Inner> {
        //获取所有权重值
        let mut initialize = m
            .get_graph_mut()
            .get_initializer_mut()
            .iter_mut()
            .map(|tp| {
                let value = Arc::new(tp.parse_mut()?);
                Ok((tp.take_name(), Tensor::Share(value)))
            })
            .collect::<LNResult<HashMap<String, Tensor>>>()?;

        let input_name: Vec<String> = m
            .get_graph_mut()
            .get_input_mut()
            .iter_mut()
            .filter_map(|inp| {
                if !initialize.contains_key(inp.name()) {
                    return Some(inp.take_name());
                }
                None
            })
            .collect::<Vec<String>>();

        let mut values_map: HashMap<String, usize> = HashMap::new();
        let mut input: Vec<usize> = Vec::new();
        for i in input_name {
            let id = values_map.len();
            values_map.insert(i, id);
            input.push(id);
        }

        let mut nodes: Vec<Node> = Vec::new();
        for n in m.get_graph_mut().get_node_mut() {
            nodes.push(n.parse_mut(&mut initialize, &mut values_map)?);
        }
        let output = m
            .get_graph_mut()
            .get_ouput_mut()
            .iter_mut()
            .map(|e| *values_map.get(e.name()).unwrap())
            .collect::<Vec<usize>>();
        println!("{:?}", values_map);
        Ok(Inner {
            nodes: nodes,
            input: input.into_boxed_slice(),
            output: output.into_boxed_slice(),
            values_len: values_map.len(),
        })
    }
}

impl Model {
    fn session(&self) -> LNResult<Session> {
        Ok(Session {
            values: vec![None; self.0.values_len],
            model: self.clone(),
        })
    }
}

pub struct Session {
    values: Vec<Option<Tensor>>, // Vec<Option<Tensor>>,/
    model: Model,
}

impl Session {
    pub fn set_input(&mut self, mut input: Tensors) {
        while let Some(v) = input.pop() {
            self.values[input.len()] = Some(v);
        }
    }

    pub fn run(&mut self) -> LNResult<Tensors> {
        for n in self.model.0.nodes.iter() {
            let mut inputs = n.get_input(&mut self.values)?;
            n.get_weight().iter().for_each(|v| inputs.push(v.1.clone()));
            let mut output = n.apply(inputs)?;
            let output_name = n.get_output();
            assert!(output.len() == output_name.len());
            while let Some(v) = output.pop() {
                let i = output_name[output.len()];
                self.values[i] = Some(v);
            }
        }
        let output = self
            .model
            .0
            .output
            .iter()
            .map(|i| self.values[*i].take().unwrap())
            .collect::<Tensors>();
        Ok(output)
    }
}

mod tests {

    use std::vec;

    use super::*;
    #[test]
    fn test_read_mnist() {
        let mut m = load("/opt/rsproject/gptgrep/lightnn/model/mnist-8.onnx").unwrap();
        let inner = Inner::load(m).unwrap();
        for input in inner.input.iter() {
            println!("input name:{:?}", input);
        }
        println!("===============================");
        for node in inner.nodes.iter() {
            println!("node name:{:?}", node.name);
            println!("node op:{:?}", node.op);
            println!("node input :{:?}", node.inputs);
            println!("node weights :{:?}", node.weights);
            println!("node output :{:?}", node.outputs);
            println!("===============================");
        }
        println!("last output:{:?}", inner.output);
    }

    use super::*;
    #[test]
    fn test_read_simple() {
        let mut m = load("/opt/rsproject/gptgrep/lightnn/model/simple.onnx").unwrap();
        let inner = Inner::load(m).unwrap();
        for input in inner.input.iter() {
            println!("input name:{:?}", input);
        }
        println!("===============================");
        for node in inner.nodes.iter() {
            println!("node name:{:?}", node.name);
            println!("node op:{:?}", node.op);
            println!("node input :{:?}", node.inputs);
            println!("node weights :{:?}", node.weights);
            println!("===============================");
        }
    }

    #[test]
    fn test_infer_mobilenet() {
        let mut m = load("/opt/rsproject/gptgrep/lightnn/model/mobilenetv2-7.onnx").unwrap();
        let inner = Inner::load(m).unwrap();
        for input in inner.input.iter() {
            println!("input name:{:?}", input);
        }
        println!("===============================");
        for node in inner.nodes.iter() {
            println!("node name:{:?}", node.name);
            println!("node op:{:?}", node.op);
            println!("node input :{:?}", node.inputs);
            println!("node weights :{:?}", node.weights);
            println!("node output :{:?}", node.outputs);
            println!("===============================");
        }
        println!("last output:{:?}", inner.output);
    }

    #[test]
    fn test_i32_to_u16() {
        let i32_array: [i32; 4] = [1, 2, 3, 4];

        // 使用unsafe代码将i32数组转换为u16数组
        let u16_array: &[u16] = unsafe {
            std::slice::from_raw_parts(
                i32_array.as_ptr() as *const u16,
                i32_array.len() * std::mem::size_of::<i32>() / std::mem::size_of::<u16>(),
            )
        };

        // 打印u16_array的内容
        for &value in u16_array {
            println!("{}", value);
        }
    }
}
