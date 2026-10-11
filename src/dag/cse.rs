// Copyright (C) 2026 Hirokazu Ishida. SPDX-License-Identifier: MPL-2.0
// Adapted from tenkai2/src/tenkai/expressions.py; see licenses/tenkai2-MPL-2.0.txt.

use std::collections::HashMap;

use blake2::{Blake2b, Digest, digest::consts::U8};

use super::{Dag, Node, NodeId, Precision};

/// Simplification and sharing before scheduling. Enabled modes permit changes
/// to signed zero and exceptional arithmetic; algebraic CSE also reassociates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Optimization {
    None,
    /// Zero/one identities, constant folding and exact structural sharing.
    Basic,
    /// Basic optimizations plus probabilistic polynomial equivalence detection.
    #[default]
    Algebraic,
}

impl Dag {
    /// Outputs must belong to this DAG. The original DAG and its IDs stay valid.
    pub(crate) fn optimize(
        &self,
        outputs: &[NodeId],
        precision: Precision,
        mode: Optimization,
    ) -> (Self, Vec<NodeId>) {
        if mode == Optimization::None {
            return (self.clone(), outputs.to_vec());
        }
        let mut optimizer = Optimizer {
            dag: Self::default(),
            precision,
            structural: HashMap::new(),
            algebraic: (mode == Optimization::Algebraic).then(Algebraic::default),
        };
        if optimizer.algebraic.is_some() {
            // Cancellation may find zero or one before either literal occurs.
            optimizer.intern(Node::Constant(0.0));
            optimizer.intern(Node::Constant(1.0));
        }
        let outputs = rebuild(self, outputs, |node| optimizer.insert(node));
        // Scheduling already excludes operands made unreachable by simplification.
        (optimizer.dag, outputs)
    }
}

fn rebuild(dag: &Dag, outputs: &[NodeId], mut insert: impl FnMut(Node) -> NodeId) -> Vec<NodeId> {
    let needed = dag.required_nodes(outputs);
    let mut mapping = vec![None; dag.nodes().len()];
    for (index, &node) in dag.nodes().iter().enumerate() {
        if !needed[index] {
            continue;
        }
        let operand =
            |id: NodeId| mapping[id.index()].expect("reachable operands precede their users");
        let node = match node {
            Node::Constant(_) | Node::Input(_) => node,
            Node::Add(a, b) => Node::Add(operand(a), operand(b)),
            Node::Sub(a, b) => Node::Sub(operand(a), operand(b)),
            Node::Mul(a, b) => Node::Mul(operand(a), operand(b)),
            Node::Sin(a) => Node::Sin(operand(a)),
            Node::Cos(a) => Node::Cos(operand(a)),
        };
        mapping[index] = Some(insert(node));
    }
    outputs
        .iter()
        .map(|id| mapping[id.index()].expect("all outputs were rebuilt"))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Constant(u64),
    Input(usize),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Sin(NodeId),
    Cos(NodeId),
}

impl From<Node> for Key {
    fn from(node: Node) -> Self {
        match node {
            Node::Constant(value) => Self::Constant(value.to_bits()),
            Node::Input(index) => Self::Input(index),
            Node::Add(a, b) => Self::Add(a, b),
            Node::Sub(a, b) => Self::Sub(a, b),
            Node::Mul(a, b) => Self::Mul(a, b),
            Node::Sin(a) => Self::Sin(a),
            Node::Cos(a) => Self::Cos(a),
        }
    }
}

struct Optimizer {
    dag: Dag,
    precision: Precision,
    structural: HashMap<Key, NodeId>,
    algebraic: Option<Algebraic>,
}

impl Optimizer {
    fn insert(&mut self, node: Node) -> NodeId {
        if let Node::Constant(value) = node {
            let rounded = match self.precision {
                Precision::F32 => f64::from(value as f32),
                Precision::F64 => value,
            };
            // Non-finite literals cannot be emitted as ordinary CUDA numbers.
            if rounded.is_finite() {
                return self.intern(Node::Constant(rounded));
            }
        }
        let (a, b) = match node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => (a, b),
            _ => return self.intern(node),
        };
        let x = self.constant(a);
        let y = self.constant(b);
        if let (Some(x), Some(y)) = (x, y) {
            let value = match self.precision {
                Precision::F32 => {
                    let (x, y) = (x as f32, y as f32);
                    f64::from(match node {
                        Node::Add(_, _) => x + y,
                        Node::Sub(_, _) => x - y,
                        Node::Mul(_, _) => x * y,
                        _ => unreachable!("only binary arithmetic reaches constant folding"),
                    })
                }
                Precision::F64 => match node {
                    Node::Add(_, _) => x + y,
                    Node::Sub(_, _) => x - y,
                    Node::Mul(_, _) => x * y,
                    _ => unreachable!("only binary arithmetic reaches constant folding"),
                },
            };
            if value.is_finite() {
                return self.intern(Node::Constant(value));
            }
            return self.intern(node);
        }
        match node {
            Node::Add(_, _) if x == Some(0.0) => b,
            Node::Add(_, _) | Node::Sub(_, _) if y == Some(0.0) => a,
            Node::Mul(_, _) if x == Some(0.0) || y == Some(0.0) => self.intern(Node::Constant(0.0)),
            Node::Mul(_, _) if x == Some(1.0) => b,
            Node::Mul(_, _) if y == Some(1.0) => a,
            _ => self.intern(node),
        }
    }

    fn constant(&self, id: NodeId) -> Option<f64> {
        match self.dag.nodes()[id.index()] {
            Node::Constant(value) => Some(value),
            _ => None,
        }
    }

    fn intern(&mut self, node: Node) -> NodeId {
        let key = Key::from(node);
        if let Some(&id) = self.structural.get(&key) {
            return id;
        }
        let fingerprint = self.algebraic.as_ref().map(|cse| cse.fingerprint(node));
        if let (Some(cse), Some(fingerprint)) = (&self.algebraic, fingerprint)
            && let Some(&id) = cse.canonical.get(&fingerprint)
        {
            self.structural.insert(key, id);
            return id;
        }
        let id = self.dag.push(node);
        self.structural.insert(key, id);
        if let (Some(cse), Some(fingerprint)) = (&mut self.algebraic, fingerprint) {
            cse.fingerprints.push(fingerprint);
            cse.canonical.insert(fingerprint, id);
        }
        id
    }
}

const FIELD: u64 = (1 << 61) - 1;
type Fingerprint = [u64; 3];

#[derive(Default)]
struct Algebraic {
    fingerprints: Vec<Fingerprint>,
    canonical: HashMap<Fingerprint, NodeId>,
}

impl Algebraic {
    fn fingerprint(&self, node: Node) -> Fingerprint {
        let (a, b) = match node {
            Node::Constant(value) => {
                if value == 0.0 {
                    return [0; 3];
                }
                let magnitude = if value.abs() == 1.0 {
                    [1; 3]
                } else {
                    // Arbitrary floats stay symbolic: reducing their numeric
                    // values modulo FIELD would introduce systematic aliases.
                    atom(b"constant", value.abs().to_bits())
                };
                return magnitude.map(|v| {
                    if value.is_sign_negative() {
                        FIELD - v
                    } else {
                        v
                    }
                });
            }
            Node::Input(index) => return atom(b"input", index as u64),
            // Trig results are opaque atoms keyed by the canonical argument.
            // No trig or unit-quaternion identities are assumed.
            Node::Sin(a) => return atom(b"sin", a.index() as u64),
            Node::Cos(a) => return atom(b"cos", a.index() as u64),
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                (self.fingerprints[a.index()], self.fingerprints[b.index()])
            }
        };
        std::array::from_fn(|i| match node {
            Node::Add(_, _) => (a[i] + b[i]) % FIELD,
            Node::Sub(_, _) => (a[i] + FIELD - b[i]) % FIELD,
            Node::Mul(_, _) => ((u128::from(a[i]) * u128::from(b[i])) % u128::from(FIELD)) as u64,
            _ => unreachable!("atoms returned before field arithmetic"),
        })
    }
}

fn atom(kind: &[u8], value: u64) -> Fingerprint {
    std::array::from_fn(|lane| {
        let mut hash = Blake2b::<U8>::new();
        hash.update(b"rkern-cse");
        hash.update([lane as u8]);
        hash.update(kind);
        hash.update(value.to_le_bytes());
        u64::from_le_bytes(hash.finalize().into()) % (FIELD - 1) + 1
    })
}
