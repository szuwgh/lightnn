use crate::op::Op;
use crate::util::error::LNResult;
pub struct Session {
    nodes: Vec<Node>,
}

impl Session {
    pub fn load() -> LNResult<Session> {
        todo!()
    }

    pub fn run() {}
}

struct Node {
    op: Box<dyn Op>,
}
