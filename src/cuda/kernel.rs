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

impl Kernel<'_> {
    pub(super) fn source(&self) -> String {
        let mut source = self.header();
        let input = format!("(configurations + batch * {})", self.dof);
        self.write_body(&mut source, &input, false, |source, index, value| {
            writeln!(
                source,
                "    output[batch * {} + {index}] = v{};",
                self.schedule.output_count(),
                value.index()
            )
            .expect("writing to String is infallible");
        });
        source.push_str("}\n");
        source
    }

    pub(super) fn write_body(
        &self,
        source: &mut String,
        input: &str,
        pair_trig: bool,
        mut store: impl FnMut(&mut String, usize, NodeId),
    ) {
        let mut steps = self.schedule.steps().iter().copied().peekable();
        while let Some(step) = steps.next() {
            match step {
                Step::Evaluate(id) => {
                    if pair_trig
                        && self.math == MathMode::F32
                        && let Some(&Step::Evaluate(next)) = steps.peek()
                    {
                        let pair =
                            match (self.dag.nodes()[id.index()], self.dag.nodes()[next.index()]) {
                                (Node::Sin(a), Node::Cos(b)) if a == b => Some((a, id, next)),
                                (Node::Cos(a), Node::Sin(b)) if a == b => Some((a, next, id)),
                                _ => None,
                            };
                        if let Some((angle, sin, cos)) = pair {
                            writeln!(
                                source,
                                "    float v{}, v{};\n    sincosf(v{}, &v{}, &v{});",
                                sin.index(),
                                cos.index(),
                                angle.index(),
                                sin.index(),
                                cos.index(),
                            )
                            .expect("writing to String is infallible");
                            steps.next();
                            continue;
                        }
                    }
                    self.write_node(source, input, id);
                }
                Step::StoreOutput { index, value } => {
                    store(source, index, value);
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

    fn write_node(&self, source: &mut String, input: &str, id: NodeId) {
        let (scalar, suffix) = self.math.precision().cuda_spelling();
        let (sin, cos) = self.math.trig_functions();
        let index = id.index();
        let node = self.dag.nodes()[index];
        write!(source, "    const {scalar} v{index} = ").expect("writing to String is infallible");
        match node {
            Node::Constant(value) => writeln!(source, "{value:.17e}{suffix};"),
            Node::Input(index) => writeln!(source, "{input}[{index}];"),
            Node::Add(a, b) => writeln!(source, "v{} + v{};", a.index(), b.index()),
            Node::Sub(a, b) => writeln!(source, "v{} - v{};", a.index(), b.index()),
            Node::Mul(a, b) => writeln!(source, "v{} * v{};", a.index(), b.index()),
            Node::Sin(a) => writeln!(source, "{sin}(v{});", a.index()),
            Node::Cos(a) => writeln!(source, "{cos}(v{});", a.index()),
        }
        .expect("writing to String is infallible");
    }
}
