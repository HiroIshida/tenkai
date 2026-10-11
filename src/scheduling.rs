// Copyright (C) 2026 Hirokazu Ishida. SPDX-License-Identifier: MPL-2.0
// Adapted from tenkai2/src/tenkai/scheduling.py; see licenses/tenkai2-MPL-2.0.txt.

//! Backend-independent scheduling of scalar DAG evaluations and ordered output stores.

use std::cmp::Reverse;
use std::fmt;

use crate::dag::{Dag, Node, NodeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Prefer completing the next output when this many scalar values are live.
    /// This is a soft threshold, not a physical register limit. Constants are excluded;
    /// inputs and arithmetic results each count as one value, regardless of precision.
    /// Zero always prioritizes the next output's dependencies.
    pub live_budget: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Evaluate reachable nodes in DAG order, then store all outputs.
    #[default]
    DagOrder,
    Live(Options),
}

impl Default for Options {
    fn default() -> Self {
        // GPU tuning is deferred; the CPU scheduler's budget of 16 is not reused.
        Self { live_budget: 64 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Evaluate(NodeId),
    StoreOutput { index: usize, value: NodeId },
}

#[derive(Debug)]
pub struct Schedule {
    steps: Vec<Step>,
    output_count: usize,
    peak_live_values: usize,
}

impl Schedule {
    /// Schedules only the nodes reachable from `outputs`, without changing the DAG.
    /// Output stores retain their slice order and may appear between evaluations.
    /// Node IDs must belong to `dag`; out-of-range IDs are rejected.
    pub fn new(dag: &Dag, outputs: &[NodeId], options: Options) -> Result<Self, InvalidOutput> {
        Self::with_mode(dag, outputs, Mode::Live(options))
    }

    /// Selects liveness scheduling or the original DAG-order evaluation and final stores.
    /// Node IDs must belong to `dag`; out-of-range IDs are rejected.
    pub fn with_mode(dag: &Dag, outputs: &[NodeId], mode: Mode) -> Result<Self, InvalidOutput> {
        for (index, value) in outputs.iter().enumerate() {
            if value.index() >= dag.nodes().len() {
                return Err(InvalidOutput {
                    index,
                    node: *value,
                    node_count: dag.nodes().len(),
                });
            }
        }
        Ok(Scheduler::new(dag, outputs).run(mode))
    }

    /// Consumes each output group before evaluating work needed only by later groups.
    /// Shared nodes are evaluated once; within each group, the selected mode's
    /// evaluation order is retained. Stores follow the flattened group order.
    pub(crate) fn for_output_groups(
        dag: &Dag,
        groups: &[&[NodeId]],
        mode: Mode,
    ) -> Result<Self, InvalidOutput> {
        let outputs = groups.concat();
        let order = Self::with_mode(dag, &outputs, mode)?;
        let mut scheduler = Scheduler::new(dag, &outputs);
        for group in groups {
            let needed = dag.required_nodes(group);
            for &step in order.steps() {
                if let Step::Evaluate(id) = step
                    && needed[id.index()]
                    && !scheduler.defined[id.index()]
                {
                    scheduler.evaluate(id);
                }
            }
            scheduler.store_ready_outputs(scheduler.next_output + group.len());
        }
        Ok(scheduler.finish())
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Maximum live scalar values across step boundaries, excluding constants.
    /// Compiler temporaries, register widths, and instruction-level overlap are not modeled.
    pub fn peak_live_values(&self) -> usize {
        self.peak_live_values
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidOutput {
    pub index: usize,
    pub node: NodeId,
    pub node_count: usize,
}

impl fmt::Display for InvalidOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "output {} references node {} outside a DAG with {} nodes",
            self.index,
            self.node.index(),
            self.node_count
        )
    }
}

impl std::error::Error for InvalidOutput {}

struct Scheduler<'a> {
    dag: &'a Dag,
    outputs: &'a [NodeId],
    users: Vec<Vec<NodeId>>,
    pending: Vec<usize>,
    remaining: Vec<usize>,
    heights: Vec<usize>,
    defined: Vec<bool>,
    ready: Vec<NodeId>,
    next_output: usize,
    live: usize,
    schedule: Schedule,
}

impl<'a> Scheduler<'a> {
    fn new(dag: &'a Dag, outputs: &'a [NodeId]) -> Self {
        let nodes = dag.nodes();
        let needed = dag.required_nodes(outputs);
        let mut users = vec![Vec::new(); nodes.len()];
        let mut pending = vec![0; nodes.len()];
        let mut remaining = vec![0; nodes.len()];
        let mut ready = Vec::new();
        for (index, node) in nodes.iter().enumerate().filter(|(i, _)| needed[*i]) {
            for dependency in node.dependencies() {
                users[dependency.index()].push(NodeId(index));
                pending[index] += 1;
                remaining[dependency.index()] += 1;
            }
            if pending[index] == 0 {
                ready.push(NodeId(index));
            }
        }
        for output in outputs {
            remaining[output.index()] += 1;
        }
        let mut heights = vec![0; nodes.len()];
        // DAG node IDs are topological, so users' heights are already known here.
        for index in (0..nodes.len()).rev().filter(|&i| needed[i]) {
            heights[index] = 1 + users[index]
                .iter()
                .map(|id| heights[id.index()])
                .max()
                .unwrap_or(0);
        }
        Self {
            dag,
            outputs,
            users,
            pending,
            remaining,
            heights,
            defined: vec![false; nodes.len()],
            ready,
            next_output: 0,
            live: 0,
            schedule: Schedule {
                steps: Vec::new(),
                output_count: outputs.len(),
                peak_live_values: 0,
            },
        }
    }

    fn run(mut self, mode: Mode) -> Schedule {
        match mode {
            Mode::DagOrder => {
                for index in 0..self.dag.nodes().len() {
                    if self.remaining[index] > 0 {
                        self.evaluate(NodeId(index));
                    }
                }
                self.store_ready_outputs(self.outputs.len());
            }
            Mode::Live(options) => self.schedule_live(options.live_budget),
        }
        self.finish()
    }

    fn finish(self) -> Schedule {
        debug_assert_eq!(self.next_output, self.outputs.len());
        debug_assert_eq!(self.live, 0);
        debug_assert!(self.remaining.iter().all(|&count| count == 0));
        self.schedule
    }

    fn schedule_live(&mut self, live_budget: usize) {
        while !self.ready.is_empty() {
            let index = self.select_ready(live_budget);
            let id = self.ready.swap_remove(index);
            self.evaluate(id);
            for &user in &self.users[id.index()] {
                self.pending[user.index()] -= 1;
                if self.pending[user.index()] == 0 {
                    self.ready.push(user);
                }
            }
            self.store_ready_outputs(self.outputs.len());
        }
    }

    fn evaluate(&mut self, id: NodeId) {
        for dependency in self.dag.nodes()[id.index()].dependencies() {
            self.consume(dependency);
        }
        self.defined[id.index()] = true;
        if self.remaining[id.index()] > 0 && self.needs_storage(id) {
            self.live += 1;
        }
        self.schedule.peak_live_values = self.schedule.peak_live_values.max(self.live);
        self.schedule.steps.push(Step::Evaluate(id));
    }

    fn select_ready(&self, live_budget: usize) -> usize {
        let needed = (self.live >= live_budget && self.next_output < self.outputs.len())
            .then(|| self.next_output_dependencies());
        self.ready
            .iter()
            .enumerate()
            .filter(|(_, id)| needed.as_ref().is_none_or(|needed| needed[id.index()]))
            .max_by_key(|(_, id)| {
                let last_uses = self.dag.nodes()[id.index()]
                    .dependencies()
                    .filter(|&dependency| {
                        self.remaining[dependency.index()] == 1 && self.needs_storage(dependency)
                    })
                    .count();
                (last_uses, self.heights[id.index()], Reverse(id.index()))
            })
            .map(|(index, _)| index)
            .expect("an unfinished output in an acyclic DAG has a ready dependency")
    }

    fn next_output_dependencies(&self) -> Vec<bool> {
        let mut needed = vec![false; self.dag.nodes().len()];
        let mut stack = vec![self.outputs[self.next_output]];
        while let Some(id) = stack.pop() {
            if !self.defined[id.index()] && !needed[id.index()] {
                needed[id.index()] = true;
                stack.extend(self.dag.nodes()[id.index()].dependencies());
            }
        }
        needed
    }

    fn store_ready_outputs(&mut self, end: usize) {
        while self.next_output < end {
            let value = self.outputs[self.next_output];
            if !self.defined[value.index()] {
                break;
            }
            self.consume(value);
            self.schedule.steps.push(Step::StoreOutput {
                index: self.next_output,
                value,
            });
            self.next_output += 1;
        }
    }

    fn consume(&mut self, id: NodeId) {
        debug_assert!(self.defined[id.index()]);
        self.remaining[id.index()] -= 1;
        if self.remaining[id.index()] == 0 && self.needs_storage(id) {
            self.live -= 1;
        }
    }

    fn needs_storage(&self, id: NodeId) -> bool {
        !matches!(self.dag.nodes()[id.index()], Node::Constant(_))
    }
}

#[cfg(test)]
mod tests;
