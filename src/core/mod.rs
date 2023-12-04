pub(crate) mod node;
pub(crate) mod op;
pub(crate) mod tensor;

pub(crate) use node::Node;
pub(crate) use tensor::Tensor;

use super::util::error::LNResult;
pub(crate) trait Parser<T> {
    fn parse(&self) -> LNResult<T>;
}

pub(crate) trait ParserMut<T> {
    fn parse_mut(&mut self) -> LNResult<T>;
}
