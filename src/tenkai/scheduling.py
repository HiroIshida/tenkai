"""DAG-order and liveness scheduling, with ordered output stores."""

from collections.abc import Sequence
from dataclasses import dataclass

from tenkai.dag import Constant, Dag, dependencies
from tenkai.ids import NodeId


@dataclass(frozen=True)
class Evaluate:
    node: NodeId


@dataclass(frozen=True)
class Store:
    index: int
    node: NodeId


type Step = Evaluate | Store


@dataclass
class Schedule:
    steps: list[Step]
    output_count: int
    peak_live_values: int


class Scheduler:
    def __init__(self, dag: Dag, outputs: Sequence[NodeId]) -> None:
        self.dag = dag
        self.outputs = list(outputs)

        size = len(dag.nodes)
        needed = dag.required_nodes(outputs)
        self.users: list[list[NodeId]] = [[] for _ in range(size)]
        self.pending = [0] * size
        self.remaining = [0] * size
        self.ready: list[NodeId] = []
        for position, node in enumerate(dag.nodes):
            index = NodeId(position)
            if not needed[index]:
                continue

            for dependency in dependencies(node):
                self.users[dependency].append(index)
                self.pending[index] += 1
                self.remaining[dependency] += 1
            if self.pending[index] == 0:
                self.ready.append(index)
        for output in outputs:
            self.remaining[output] += 1

        self.heights = [0] * size
        for position in range(size - 1, -1, -1):
            index = NodeId(position)
            if needed[index]:
                height = max((self.heights[i] for i in self.users[index]), default=0)
                self.heights[index] = 1 + height

        self.defined = [False] * size
        self.next_output = 0
        self.live = 0
        self.peak_live_values = 0
        self.steps: list[Step] = []

    def needs_storage(self, index: NodeId) -> bool:
        return not isinstance(self.dag.nodes[index], Constant)

    def consume(self, index: NodeId) -> None:
        self.remaining[index] -= 1
        if self.remaining[index] == 0 and self.needs_storage(index):
            self.live -= 1

    def evaluate(self, index: NodeId) -> None:
        for dependency in dependencies(self.dag.nodes[index]):
            self.consume(dependency)

        self.defined[index] = True
        if self.remaining[index] > 0 and self.needs_storage(index):
            self.live += 1
        self.peak_live_values = max(self.peak_live_values, self.live)
        self.steps.append(Evaluate(index))

    def store_ready_outputs(self, end: int) -> None:
        while self.next_output < end:
            value = self.outputs[self.next_output]
            if not self.defined[value]:
                break

            self.consume(value)
            self.steps.append(Store(self.next_output, value))
            self.next_output += 1

    def next_output_dependencies(self) -> set[NodeId]:
        needed: set[NodeId] = set()
        stack = [self.outputs[self.next_output]]
        while stack:
            index = stack.pop()
            if not self.defined[index] and index not in needed:
                needed.add(index)
                stack.extend(dependencies(self.dag.nodes[index]))

        return needed

    def score(self, index: NodeId) -> tuple[int, int, int]:
        last_uses = 0
        for dependency in dependencies(self.dag.nodes[index]):
            if self.remaining[dependency] == 1 and self.needs_storage(dependency):
                last_uses += 1

        return last_uses, self.heights[index], -index

    def select_ready(self, live_budget: int) -> int:
        needed: set[NodeId] | None = None
        if self.live >= live_budget and self.next_output < len(self.outputs):
            needed = self.next_output_dependencies()

        candidate_positions: list[int] = []
        for position, index in enumerate(self.ready):
            if needed is None or index in needed:
                candidate_positions.append(position)

        return max(candidate_positions, key=lambda position: self.score(self.ready[position]))

    def run(self, live_budget: int | None) -> Schedule:
        if live_budget is None:
            for position in range(len(self.dag.nodes)):
                if self.remaining[position] > 0:
                    self.evaluate(NodeId(position))
            self.store_ready_outputs(len(self.outputs))
        else:
            while self.ready:
                position = self.select_ready(live_budget)
                index = self.ready[position]
                self.ready[position] = self.ready[-1]
                self.ready.pop()

                self.evaluate(index)
                for user in self.users[index]:
                    self.pending[user] -= 1
                    if self.pending[user] == 0:
                        self.ready.append(user)

                self.store_ready_outputs(len(self.outputs))

        return self.finish()

    def finish(self) -> Schedule:
        return Schedule(self.steps, len(self.outputs), self.peak_live_values)


def schedule(
    dag: Dag,
    outputs: Sequence[NodeId],
    live_budget: int | None = None,
    groups: Sequence[Sequence[NodeId]] | None = None,
) -> Schedule:
    if live_budget is not None and live_budget < 0:
        raise ValueError("live budget must be nonnegative")

    order = Scheduler(dag, outputs).run(live_budget)
    if groups is None:
        return order

    scheduler = Scheduler(dag, outputs)
    for group in groups:
        needed = dag.required_nodes(group)
        for step in order.steps:
            if not isinstance(step, Evaluate):
                continue
            if needed[step.node] and not scheduler.defined[step.node]:
                scheduler.evaluate(step.node)
        scheduler.store_ready_outputs(scheduler.next_output + len(group))

    return scheduler.finish()
