import pytest
from helpers import execute

from tenkai.dag import Constant, Dag, Optimization, Precision
from tenkai.scheduling import Evaluate, Store, schedule


@pytest.mark.parametrize("mode", ["basic", "algebraic"])
@pytest.mark.parametrize(
    "precision, expected",
    [
        ("f32", [16777216, 0, 16785408, 16777216]),
        ("f64", [16777217, 1, 16785409, 16777217]),
    ],
)
def test_constant_folding_uses_selected_precision(
    mode: Optimization, precision: Precision, expected: list[float]
) -> None:
    dag = Dag()
    large = dag.constant(16777216)
    one = dag.constant(1)
    factor = dag.constant(4097)
    total = dag.add(large, one)
    outputs = [total, dag.sub(total, large), dag.mul(factor, factor), dag.constant(16777217)]
    result, ids = dag.optimize(outputs, precision, mode)
    assert [result.nodes[i] for i in ids] == [Constant(value) for value in expected]


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_algebraic_equivalents_share_nodes(precision: Precision) -> None:
    dag = Dag()
    a, b, c = [dag.input(i) for i in range(3)]
    ab = dag.add(a, b)
    outputs = [dag.mul(ab, c), dag.add(dag.mul(a, c), dag.mul(b, c)), dag.sub(ab, b), a]
    _, ids = dag.optimize(outputs, precision, "algebraic")
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]


@pytest.mark.parametrize("mode", ["basic", "algebraic"])
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("budget", [None, 0, 1, 64])
def test_schedules_keep_output_order_and_omit_dead_work(
    mode: Optimization, precision: Precision, budget: int | None
) -> None:
    dag = Dag()
    x, y, unused = [dag.input(i) for i in range(3)]
    dead = dag.mul(dag.sin(unused), dag.constant(0))
    xy = dag.add(x, y)
    yx = dag.add(y, x)
    outputs = [yx, dead, dag.mul(xy, yx), dag.sub(x, y), xy, yx]
    optimized, ids = dag.optimize(outputs, precision, mode)
    order = schedule(optimized, ids, budget, [ids[:3], ids[3:]])
    assert execute(optimized, order, [2, 3]) == [5, 0, 25, -1, 5, 5]
    assert order == schedule(optimized, ids, budget, [ids[:3], ids[3:]])


def test_live_budget_finishes_outputs_early() -> None:
    dag = Dag()
    inputs = [dag.input(i) for i in range(8)]
    outputs = [dag.mul(dag.sin(i), dag.cos(i)) for i in inputs]
    bounded = schedule(dag, outputs, 1)
    unbounded = schedule(dag, outputs, 1000)
    assert bounded.peak_live_values == 2
    assert bounded.peak_live_values < unbounded.peak_live_values
    assert bounded.steps.index(Store(0, outputs[0])) < bounded.steps.index(Evaluate(inputs[1]))
    assert execute(dag, bounded, [0.2] * 8) == execute(dag, unbounded, [0.2] * 8)


def test_dag_order_and_duplicate_outputs() -> None:
    dag = Dag()
    x = dag.input(0)
    square = dag.mul(x, x)
    dag.cos(x)
    total = dag.add(x, square)
    outputs = [x, total, square, x]
    order = schedule(dag, outputs)
    assert order.steps == [
        Evaluate(x),
        Evaluate(square),
        Evaluate(total),
        Store(0, x),
        Store(1, total),
        Store(2, square),
        Store(3, x),
    ]
    assert execute(dag, order, [2]) == [2, 6, 4, 2]
    assert schedule(dag, []).steps == []
    with pytest.raises(ValueError):
        schedule(dag, [999])


def test_no_optimization_keeps_original_nodes() -> None:
    dag = Dag()
    x = dag.input(0)
    total = dag.add(x, dag.constant(0))
    dag.cos(x)
    optimized, ids = dag.optimize([total, x, total], "f32", "none")
    assert optimized.nodes == dag.nodes
    assert ids == [total, x, total]
