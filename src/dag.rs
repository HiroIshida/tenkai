#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Every node produces one scalar. Operands refer to earlier nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Node {
    Constant(f64),
    Input(usize),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Sin(NodeId),
    Cos(NodeId),
}

#[derive(Debug, Default, Clone)]
pub struct Dag {
    nodes: Vec<Node>,
}

impl Dag {
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub(crate) fn required_nodes(&self, outputs: &[NodeId]) -> Vec<bool> {
        let nodes = self.nodes();
        let mut needed = vec![false; nodes.len()];
        for output in outputs {
            needed[output.index()] = true;
        }
        // Operands precede their users, so no fixed-point iteration is needed.
        for (index, node) in nodes.iter().enumerate().rev() {
            if !needed[index] {
                continue;
            }
            match *node {
                Node::Constant(_) | Node::Input(_) => {}
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    needed[a.index()] = true;
                    needed[b.index()] = true;
                }
                Node::Sin(a) | Node::Cos(a) => needed[a.index()] = true,
            }
        }

        needed
    }

    fn push(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub(crate) fn constant(&mut self, value: f64) -> NodeId {
        self.push(Node::Constant(value))
    }

    pub(crate) fn input(&mut self, index: usize) -> NodeId {
        self.push(Node::Input(index))
    }

    pub(crate) fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Add(a, b))
    }

    pub(crate) fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Sub(a, b))
    }

    pub(crate) fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Node::Mul(a, b))
    }

    pub(crate) fn sin(&mut self, angle: NodeId) -> NodeId {
        self.push(Node::Sin(angle))
    }

    pub(crate) fn cos(&mut self, angle: NodeId) -> NodeId {
        self.push(Node::Cos(angle))
    }
}
