use std::fmt::Write;

use super::MathMode;
use crate::dag::{Dag, Node, NodeId};
use crate::scheduling::{Schedule, Step};

pub(super) struct Kernel<'a> {
    pub(super) name: &'static str,
    pub(super) dag: &'a Dag,
    pub(super) dof: usize,
    pub(super) schedule: Schedule,
    pub(super) math: MathMode,
}

pub(super) enum Output {
    Batched,
    LocalCenters,
}

impl Kernel<'_> {
    pub(super) fn source(&self) -> String {
        let mut source = self.header();
        self.write_body(&mut source, Output::Batched);
        source.push_str("}\n");
        source
    }

    pub(super) fn write_body(&self, source: &mut String, output: Output) {
        for &step in self.schedule.steps() {
            match step {
                Step::Evaluate(id) => self.write_node(source, id),
                Step::StoreOutput { index, value } => {
                    let target = match output {
                        Output::Batched => {
                            format!("output[batch * {} + {index}]", self.schedule.output_count())
                        }
                        Output::LocalCenters => format!("centers[{index}]"),
                    };
                    writeln!(source, "    {target} = v{};", value.index())
                        .expect("writing to String is infallible");
                }
            }
        }
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

    fn write_node(&self, source: &mut String, id: NodeId) {
        let (scalar, suffix) = self.math.precision().cuda_spelling();
        let dof = self.dof;
        let (sin, cos) = self.math.trig_functions();
        let index = id.index();
        let node = self.dag.nodes()[index];
        write!(source, "    const {scalar} v{index} = ").expect("writing to String is infallible");
        match node {
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
