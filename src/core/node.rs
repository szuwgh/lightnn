use super::op::Op;

pub(crate) struct Node {
    name: Option<String>,
    op: Op,
    inputs: Box<[String]>,
    outputs: Box<[String]>,
}

#[derive(Default)]
pub(crate) struct NodeBuilder {
    name: Option<String>,
    op: Op,
    inputs: Box<[String]>,
    outputs: Box<[String]>,
}

impl NodeBuilder {
    pub fn name(mut self, n: String) -> NodeBuilder {
        self.name = Some(n);
        self
    }

    pub(crate) fn build(self) -> Node {
        Node {
            name: self.name,
            op: self.op,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

impl Node {}
