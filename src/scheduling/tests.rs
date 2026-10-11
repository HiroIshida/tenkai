use super::*;

fn evaluate(node: Node, inputs: &[f64], get: impl Fn(NodeId) -> f64) -> f64 {
    match node {
        Node::Constant(value) => value,
        Node::Input(index) => inputs[index],
        Node::Add(a, b) => get(a) + get(b),
        Node::Sub(a, b) => get(a) - get(b),
        Node::Mul(a, b) => get(a) * get(b),
        Node::Sin(a) => get(a).sin(),
        Node::Cos(a) => get(a).cos(),
    }
}

fn check(dag: &Dag, outputs: &[NodeId], inputs: &[f64], live_budget: usize) -> Schedule {
    check_mode(dag, outputs, inputs, Mode::Live(Options { live_budget }))
}

fn check_mode(dag: &Dag, outputs: &[NodeId], inputs: &[f64], mode: Mode) -> Schedule {
    let schedule = Schedule::with_mode(dag, outputs, mode).unwrap();
    check_schedule(dag, outputs, inputs, schedule)
}

fn check_schedule(dag: &Dag, outputs: &[NodeId], inputs: &[f64], schedule: Schedule) -> Schedule {
    let mut reference = Vec::new();
    for &node in dag.nodes() {
        reference.push(evaluate(node, inputs, |id| reference[id.index()]));
    }
    let mut values = vec![None; dag.nodes().len()];
    let mut definitions = vec![None; dag.nodes().len()];
    let mut last_uses = vec![None; dag.nodes().len()];
    let mut stored = 0;
    for (position, &step) in schedule.steps().iter().enumerate() {
        match step {
            Step::Evaluate(id) => {
                assert!(values[id.index()].is_none(), "each node is evaluated once");
                let node = dag.nodes()[id.index()];
                let value = evaluate(node, inputs, |dependency| {
                    values[dependency.index()].expect("dependencies precede their users")
                });
                assert_eq!(value, reference[id.index()]);
                values[id.index()] = Some(value);
                definitions[id.index()] = Some(position);
                for dependency in node.dependencies() {
                    last_uses[dependency.index()] = Some(position);
                }
            }
            Step::StoreOutput { index, value } => {
                assert_eq!(index, stored, "output stores retain their requested order");
                assert_eq!(value, outputs[index]);
                assert_eq!(values[value.index()], Some(reference[value.index()]));
                last_uses[value.index()] = Some(position);
                stored += 1;
            }
        }
    }
    assert_eq!(stored, outputs.len());
    assert_eq!(schedule.output_count(), outputs.len());
    assert_eq!(
        values.iter().map(Option::is_some).collect::<Vec<_>>(),
        dag.required_nodes(outputs),
        "only reachable nodes are evaluated"
    );

    let mut changes = vec![0_isize; schedule.steps().len()];
    for (index, definition) in definitions.into_iter().enumerate() {
        if let Some(start) = definition {
            let end = last_uses[index].expect("each scheduled node has a user or an output store");
            assert!(end > start);
            if !matches!(dag.nodes()[index], Node::Constant(_)) {
                changes[start] += 1;
                changes[end] -= 1;
            }
        }
    }
    let mut live = 0;
    let mut peak = 0;
    for change in changes {
        live += change;
        peak = peak.max(live);
    }
    assert_eq!(live, 0);
    assert_eq!(schedule.peak_live_values(), peak as usize);
    schedule
}

#[test]
fn shared_values_survive_repeated_operands_and_output_stores() {
    let mut dag = Dag::default();
    let x = dag.input(0);
    let square = dag.mul(x, x);
    let doubled = dag.add(square, square);
    let difference = dag.sub(doubled, square);
    let dead = dag.cos(x);
    for budget in [0, 1, 64, usize::MAX] {
        let schedule = check(&dag, &[x, difference, square, x], &[0.75], budget);
        assert!(!schedule.steps().contains(&Step::Evaluate(dead)));
    }
}

#[test]
fn last_uses_take_priority_over_starting_other_operations() {
    let mut dag = Dag::default();
    let a = dag.input(0);
    let b = dag.input(1);
    let a_sin = dag.sin(a);
    let a_cos = dag.cos(a);
    let b_sin = dag.sin(b);
    let schedule = check(&dag, &[a_sin, a_cos, b_sin], &[0.3, -0.7], usize::MAX);
    let position = |id| {
        schedule
            .steps()
            .iter()
            .position(|&step| step == Step::Evaluate(id))
            .unwrap()
    };
    assert!(position(b_sin) < position(a_sin));
    assert!(position(b_sin) < position(a_cos));
}

#[test]
fn a_small_budget_finishes_outputs_early_and_reduces_peak_liveness() {
    let mut dag = Dag::default();
    let inputs: Vec<_> = (0..8).map(|index| dag.input(index)).collect();
    let outputs: Vec<_> = inputs
        .iter()
        .map(|&input| {
            let sin = dag.sin(input);
            let cos = dag.cos(input);
            dag.mul(sin, cos)
        })
        .collect();
    let q = [0.2; 8];
    let unbounded = check(&dag, &outputs, &q, usize::MAX);
    let bounded = check(&dag, &outputs, &q, 1);
    assert_eq!(
        bounded.peak_live_values(),
        2,
        "the budget is a soft threshold"
    );
    assert!(bounded.peak_live_values() < unbounded.peak_live_values());
    let first_store = bounded
        .steps()
        .iter()
        .position(|step| matches!(step, Step::StoreOutput { index: 0, .. }))
        .unwrap();
    let second_input = bounded
        .steps()
        .iter()
        .position(|&step| step == Step::Evaluate(inputs[1]))
        .unwrap();
    assert!(first_store < second_input);
    assert_eq!(
        bounded.steps(),
        check(&dag, &outputs, &q, 1).steps(),
        "scheduling is deterministic"
    );
}

#[test]
fn constants_and_empty_outputs_do_not_keep_runtime_values_live() {
    let mut dag = Dag::default();
    let constant = dag.constant(3.0);
    let input = dag.input(0);
    let result = dag.add(input, constant);
    let schedule = check(&dag, &[constant, result, constant], &[2.0], 0);
    assert_eq!(schedule.peak_live_values(), 1);
    assert!(check(&dag, &[], &[2.0], 0).steps().is_empty());
    assert!(check(&Dag::default(), &[], &[], 64).steps().is_empty());
}

#[test]
fn out_of_range_outputs_are_rejected() {
    let mut dag = Dag::default();
    let value = dag.constant(1.0);
    let error = Schedule::new(&dag, &[value, NodeId(usize::MAX)], Options::default()).unwrap_err();
    assert_eq!(
        error,
        InvalidOutput {
            index: 1,
            node: NodeId(usize::MAX),
            node_count: 1
        }
    );
    assert_eq!(
        Schedule::for_output_groups(
            &dag,
            &[&[], &[value], &[NodeId(usize::MAX)]],
            Mode::DagOrder
        )
        .unwrap_err(),
        error
    );
}

#[test]
fn output_groups_finish_before_unrelated_work_and_reuse_shared_values() {
    let mut dag = Dag::default();
    let inputs: Vec<_> = (0..8).map(|index| dag.input(index)).collect();
    let shared = dag.sin(inputs[0]);
    let mut groups = vec![vec![]];
    for &input in &inputs[1..] {
        let sin = dag.sin(input);
        let sum = dag.add(sin, shared);
        groups.push(vec![sum, sin, sum]);
    }
    groups.push(groups[1].clone());
    groups.push(vec![]);
    let groups: Vec<_> = groups.iter().map(Vec::as_slice).collect();
    let outputs = groups.concat();
    let q = [0.3; 8];
    let original = check_mode(&dag, &outputs, &q, Mode::DagOrder);
    for mode in std::iter::once(Mode::DagOrder)
        .chain([0, 1, 64, usize::MAX].map(|live_budget| Mode::Live(Options { live_budget })))
    {
        let schedule = check_schedule(
            &dag,
            &outputs,
            &q,
            Schedule::for_output_groups(&dag, &groups, mode).unwrap(),
        );
        assert!(schedule.peak_live_values() < original.peak_live_values());
        let mut stored = 0;
        for &step in schedule.steps() {
            match step {
                Step::Evaluate(id) => {
                    let group = &outputs[stored / 3 * 3..stored / 3 * 3 + 3];
                    assert!(dag.required_nodes(group)[id.index()]);
                }
                Step::StoreOutput { .. } => stored += 1,
            }
        }
        assert_eq!(
            schedule.steps(),
            Schedule::for_output_groups(&dag, &groups, mode)
                .unwrap()
                .steps(),
            "grouped scheduling is deterministic"
        );
        assert!(
            Schedule::for_output_groups(&dag, &[&[], &[]], mode)
                .unwrap()
                .steps()
                .is_empty()
        );
    }
}

#[test]
fn all_link_poses_match_dag_order_across_budgets_and_configurations() {
    for urdf in [
        include_str!("../../tests/fixtures/fk_chain.urdf"),
        include_str!("../../assets/fetch/fetch.urdf"),
    ] {
        let fk =
            crate::fk::FkDag::from_robot(&crate::urdf::read_from_string(urdf).unwrap()).unwrap();
        let outputs: Vec<_> = fk
            .link_poses()
            .iter()
            .flat_map(|pose| pose.translation.into_iter().chain(pose.rotation))
            .collect();
        for mode in std::iter::once(Mode::DagOrder).chain(
            [0, 1, 16, 64, usize::MAX].map(|live_budget| Mode::Live(Options { live_budget })),
        ) {
            for configuration in 0..3 {
                let inputs: Vec<_> = (0..fk.joint_names().len())
                    .map(|i| (configuration as f64 * 0.7 + i as f64 * 0.31).sin())
                    .collect();
                check_mode(fk.dag(), &outputs, &inputs, mode);
            }
        }
    }
}

#[test]
fn disabled_scheduling_keeps_dag_order_and_stores_outputs_at_the_end() {
    let mut dag = Dag::default();
    let input = dag.input(0);
    let square = dag.mul(input, input);
    let _dead = dag.cos(input);
    let sum = dag.add(input, square);
    let schedule = check_mode(&dag, &[input, sum, square, input], &[0.3], Mode::DagOrder);
    assert_eq!(
        schedule.steps(),
        &[
            Step::Evaluate(input),
            Step::Evaluate(square),
            Step::Evaluate(sum),
            Step::StoreOutput {
                index: 0,
                value: input
            },
            Step::StoreOutput {
                index: 1,
                value: sum
            },
            Step::StoreOutput {
                index: 2,
                value: square
            },
            Step::StoreOutput {
                index: 3,
                value: input
            },
        ]
    );
    assert!(
        check_mode(&dag, &[], &[0.3], Mode::DagOrder)
            .steps()
            .is_empty()
    );
}
