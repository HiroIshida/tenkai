use std::fmt::Write;

use super::MathMode;
use crate::dag::{Dag, Node, NodeId};

pub(super) struct Kernel<'a> {
    pub(super) name: &'static str,
    pub(super) dag: &'a Dag,
    pub(super) dof: usize,
    pub(super) outputs: &'a [NodeId],
    pub(super) math: MathMode,
}

impl Kernel<'_> {
    pub(super) fn source(&self) -> String {
        let needed = self.dag.required_nodes(self.outputs);
        let mut source = self.header();
        self.write_nodes(&mut source, &needed);
        self.write_outputs(&mut source);
        source.push_str("}\n");
        source
    }

    fn header(&self) -> String {
        let name = self.name;
        let (scalar, _) = self.math.precision().cuda_spelling();
        format!(
            r#"#include <stddef.h>
#include <math.h>

extern "C" __global__ void {name}(
    const {scalar}* configurations, {scalar}* output, size_t count) {{
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch >= count) return;
"#,
        )
    }

    fn write_nodes(&self, source: &mut String, needed: &[bool]) {
        let (scalar, suffix) = self.math.precision().cuda_spelling();
        let dof = self.dof;
        let (sin, cos) = self.math.trig_functions();
        for (index, node) in self.dag.nodes().iter().enumerate() {
            if !needed[index] {
                continue;
            }
            write!(source, "    const {scalar} v{index} = ")
                .expect("writing to String is infallible");
            match *node {
                Node::Constant(value) => writeln!(source, "{value:.17e}{suffix};"),
                Node::Input(input) => writeln!(source, "configurations[batch * {dof} + {input}];"),
                Node::Add(a, b) => writeln!(source, "v{} + v{};", a.index(), b.index()),
                Node::Sub(a, b) => writeln!(source, "v{} - v{};", a.index(), b.index()),
                Node::Mul(a, b) => writeln!(source, "v{} * v{};", a.index(), b.index()),
                Node::Sin(a) => writeln!(source, "{sin}(v{});", a.index()),
                Node::Cos(a) => writeln!(source, "{cos}(v{});", a.index()),
            }
            .expect("writing to String is infallible");
        }
    }

    fn write_outputs(&self, source: &mut String) {
        let components = self.outputs.len();
        for (component, output) in self.outputs.iter().enumerate() {
            writeln!(
                source,
                "    output[batch * {components} + {component}] = v{};",
                output.index()
            )
            .expect("writing to String is infallible");
        }
    }
}
