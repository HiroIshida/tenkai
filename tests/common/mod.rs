use rkern::dag::{Node, NodeId};
use rkern::fk::FkDag;

pub fn evaluate(fk: &FkDag, inputs: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), fk.joint_names().len());
    let mut values: Vec<f64> = Vec::with_capacity(fk.dag().nodes().len());
    for (index, node) in fk.dag().nodes().iter().enumerate() {
        let get = |id: NodeId| {
            assert!(id.index() < index, "operand must precede its user");
            values[id.index()]
        };
        let value = match *node {
            Node::Constant(value) => value,
            Node::Input(index) => inputs[index],
            Node::Add(a, b) => get(a) + get(b),
            Node::Sub(a, b) => get(a) - get(b),
            Node::Mul(a, b) => get(a) * get(b),
            Node::Sin(angle) => get(angle).sin(),
            Node::Cos(angle) => get(angle).cos(),
        };
        assert!(value.is_finite());
        values.push(value);
    }
    values
}
