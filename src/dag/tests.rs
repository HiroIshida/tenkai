use super::*;
use crate::scheduling::{self, Schedule, Step};

const PRECISIONS: [Precision; 2] = [Precision::F32, Precision::F64];
const ENABLED: [Optimization; 2] = [Optimization::Basic, Optimization::Algebraic];

#[test]
fn zero_and_one_identities_reuse_the_operand_or_zero() {
    let mut dag = Dag::default();
    let x = dag.input(0);
    let zero = dag.constant(0.0);
    let one = dag.constant(1.0);
    let outputs = [
        x,
        dag.add(x, zero),
        dag.add(zero, x),
        dag.sub(x, zero),
        dag.mul(x, one),
        dag.mul(one, x),
        zero,
        dag.mul(x, zero),
        dag.mul(zero, x),
    ];
    for precision in PRECISIONS {
        for mode in ENABLED {
            let (optimized, outputs) = dag.optimize(&outputs, precision, mode);
            assert!(outputs[..6].iter().all(|&id| id == outputs[0]));
            assert!(outputs[6..].iter().all(|&id| id == outputs[6]));
            assert_eq!(optimized.nodes()[outputs[0].index()], Node::Input(0));
            assert_eq!(optimized.nodes()[outputs[6].index()], Node::Constant(0.0));
        }
    }
}

#[test]
fn constants_are_folded_at_the_selected_precision() {
    let mut dag = Dag::default();
    let large = dag.constant(16_777_216.0);
    let one = dag.constant(1.0);
    let sum = dag.add(large, one);
    let difference = dag.sub(sum, large);
    let factor = dag.constant(4097.0);
    let product = dag.mul(factor, factor);
    let literal = dag.constant(16_777_217.0);
    for (precision, expected) in [
        (
            Precision::F32,
            [16_777_216.0, 0.0, 16_785_408.0, 16_777_216.0],
        ),
        (
            Precision::F64,
            [16_777_217.0, 1.0, 16_785_409.0, 16_777_217.0],
        ),
    ] {
        for mode in ENABLED {
            let (optimized, outputs) =
                dag.optimize(&[sum, difference, product, literal], precision, mode);
            for (id, value) in outputs.into_iter().zip(expected) {
                assert_eq!(optimized.nodes()[id.index()], Node::Constant(value));
            }
        }
    }
}

#[test]
fn repeated_expressions_share_results_while_distinct_expressions_stay_distinct() {
    let mut dag = Dag::default();
    let expressions = std::array::from_fn::<_, 2, _>(|_| {
        let x = dag.input(0);
        let y = dag.input(1);
        let quarter = dag.constant(0.25);
        let half = dag.constant(0.5);
        let sum = dag.add(x, quarter);
        let difference = dag.sub(sum, y);
        let product = dag.mul(difference, sum);
        let sine = dag.sin(x);
        let cosine = dag.cos(x);
        [x, y, quarter, half, sum, difference, product, sine, cosine]
    });
    for precision in PRECISIONS {
        for mode in ENABLED {
            let (_, outputs) = dag.optimize(expressions.as_flattened(), precision, mode);
            let (first, second) = outputs.split_at(expressions[0].len());
            assert_eq!(first, second);
            for (i, id) in first.iter().enumerate() {
                assert!(first[i + 1..].iter().all(|other| id != other));
            }
        }
    }
}

#[test]
fn algebraic_equivalents_share_results() {
    let mut dag = Dag::default();
    let a = dag.input(0);
    let b = dag.input(1);
    let c = dag.input(2);
    let zero = dag.constant(0.0);
    let minus_one = dag.constant(-1.0);
    let ab = dag.add(a, b);
    let bc = dag.add(b, c);
    let a_times_b = dag.mul(a, b);
    let b_times_c = dag.mul(b, c);
    let a_times_c = dag.mul(a, c);
    let positive = dag.constant(0.25);
    let negative = dag.constant(-0.25);
    let scaled = dag.mul(positive, a);
    let negated = dag.mul(negative, a);
    let pairs = [
        ("addition order", [ab, dag.add(b, a)]),
        ("multiplication order", [a_times_b, dag.mul(b, a)]),
        ("addition grouping", [dag.add(ab, c), dag.add(a, bc)]),
        (
            "multiplication grouping",
            [dag.mul(a_times_b, c), dag.mul(a, b_times_c)],
        ),
        (
            "distribution",
            [dag.mul(ab, c), dag.add(a_times_c, b_times_c)],
        ),
        ("cancellation", [dag.sub(ab, b), a]),
        ("self subtraction", [dag.sub(a, a), zero]),
        ("negation", [dag.sub(zero, a), dag.mul(minus_one, a)]),
        ("opposite coefficients", [dag.add(scaled, negated), zero]),
    ];
    let outputs: Vec<_> = pairs.iter().flat_map(|(_, pair)| *pair).collect();
    for precision in PRECISIONS {
        let (_, outputs) = dag.optimize(&outputs, precision, Optimization::Algebraic);
        assert_eq!(outputs.len(), pairs.len() * 2);
        for ((name, _), pair) in pairs.iter().zip(outputs.as_chunks::<2>().0) {
            assert_eq!(pair[0], pair[1], "{name}, {precision:?}");
        }
    }
}

#[test]
fn disabled_optimization_preserves_nodes_and_output_ids() {
    let mut dag = Dag::default();
    let x = dag.input(0);
    let zero = dag.constant(0.0);
    let sum = dag.add(x, zero);
    dag.cos(x);
    let outputs = [sum, x, sum];
    for precision in PRECISIONS {
        let (optimized, remapped) = dag.optimize(&outputs, precision, Optimization::None);
        assert_eq!(optimized.nodes(), dag.nodes());
        assert_eq!(remapped, outputs);
    }
}

#[test]
fn remapped_outputs_keep_their_values_and_order_without_discarded_work() {
    let mut dag = Dag::default();
    let x = dag.input(0);
    let y = dag.input(1);
    let unused_input = dag.input(2);
    let sine = dag.sin(unused_input);
    let zero = dag.constant(0.0);
    let discarded = dag.mul(sine, zero);
    let xy = dag.add(x, y);
    let yx = dag.add(y, x);
    let product = dag.mul(xy, yx);
    let difference = dag.sub(x, y);
    let outputs = [yx, discarded, product, difference, xy, yx];
    for precision in PRECISIONS {
        for mode in ENABLED {
            let (optimized, outputs) = dag.optimize(&outputs, precision, mode);
            let groups: Vec<_> = outputs.chunks(3).collect();
            for scheduling in [
                scheduling::Mode::DagOrder,
                scheduling::Mode::Live(scheduling::Options::default()),
            ] {
                let schedule =
                    Schedule::for_output_groups(&optimized, &groups, scheduling).unwrap();
                let mut values: Vec<Option<f64>> = vec![None; optimized.nodes().len()];
                let mut actual = Vec::new();
                for &step in schedule.steps() {
                    match step {
                        Step::Evaluate(id) => {
                            let get = |id: NodeId| values[id.index()].unwrap();
                            let value = match optimized.nodes()[id.index()] {
                                Node::Constant(value) => value,
                                Node::Input(index) => [2.0, 3.0][index],
                                Node::Add(a, b) => get(a) + get(b),
                                Node::Sub(a, b) => get(a) - get(b),
                                Node::Mul(a, b) => get(a) * get(b),
                                Node::Sin(_) | Node::Cos(_) => {
                                    panic!("discarded work must not be scheduled")
                                }
                            };
                            assert!(values[id.index()].replace(value).is_none());
                        }
                        Step::StoreOutput { index, value } => {
                            assert_eq!(index, actual.len());
                            actual.push(values[value.index()].unwrap());
                        }
                    }
                }
                assert_eq!(actual, [5.0, 0.0, 25.0, -1.0, 5.0, 5.0]);
            }
        }
    }
}
