#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "rrt.cuh"

namespace m = RKERN_MODEL_NAMESPACE;
using Scalar = m::Scalar;
using Status = rkern::PlanStatus;
using Algorithm = rkern::RrtAlgorithm;

template <typename F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
}

bool collision(const Scalar* q, uint32_t environment) {
    if (!environment) return false;
    const double x = std::max(std::abs(double(q[1])) - 0.1, 0.0);
    const double y = std::max(std::abs(double(q[0])) - (environment == 1 ? 0.4 : 2.0), 0.0);
    return x*x + y*y <= 0.05*0.05;
}

void validate_path(const std::vector<Scalar>& path, const m::RrtProblem& problem,
                   const m::RrtSpace& space, const rkern::RrtConfig& config, Algorithm algorithm) {
    assert(!path.empty() && path.size() % 2 == 0);
    for (size_t d = 0; d < 2; ++d) {
        assert(path[d] == problem.start[d]);
        assert(path[path.size() - 2 + d] == problem.goal[d]);
    }
    for (size_t i = 0; i < path.size(); i += 2) {
        for (size_t d = 0; d < 2; ++d)
            assert(std::isfinite(path[i+d]) && path[i+d] >= space.lower[d] && path[i+d] <= space.upper[d]);
        assert(!collision(&path[i], problem.environment_id));
        if (!i) continue;
        double norm = 0, segments = 1;
        for (size_t d = 0; d < 2; ++d) {
            const double delta = double(path[i+d]) - double(path[i-2+d]);
            norm += delta*delta;
            segments = std::max(segments, std::ceil(std::abs(delta) / double(space.resolution[d])));
        }
        if (algorithm == Algorithm::RrtConnect || i + 2 < path.size())
            assert(std::sqrt(norm) <= config.step_size + 1e-6);
        if (algorithm == Algorithm::RrtConnect) assert(norm > 0);
        for (unsigned int j = 0; j <= segments; ++j) {
            Scalar q[2];
            for (size_t d = 0; d < 2; ++d)
                q[d] = Scalar(double(path[i-2+d]) + (j / segments) * (double(path[i+d])-double(path[i-2+d])));
            assert(!collision(q, problem.environment_id));
        }
    }
}

__global__ void check_edge_grid(m::RrtCandidate* candidates, uint32_t count,
                                const Scalar* nodes, m::RrtSpace space, m::Scene scene,
                                m::EdgeDirection direction) {
    m::RrtProblem problem{};
    problem.environment_id = 1;
    m::rrt_check_edges(candidates, count, nodes, problem, space, scene, direction);
}

void check_edge_grids(const m::RrtSpace& space, m::Scene scene) {
    struct Edge { double from, to, y; uint32_t steps; bool blocked; };
    const Edge cases[] = {
        {-0.8, 1.2, 0, 80, true}, // All quarter probes miss the obstacle.
        {-2.6, 0.6, 0, 65, true}, // Collision after the first 32 samples.
        {-0.8, 1.2, 0.8, 80, false},
        {-0.8, 0.8, 0, 1, false}, // Only the endpoint belongs to this grid.
        {-0.8, 0.8, 0, 2, true},
        {0.2, 0, 0, 1, true}, // The new endpoint must be checked in reverse too.
        {0.8, 0.8, 0, 0, false},
    };
    const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
    const uint32_t count = 3 * case_count;
    std::vector<m::RrtCandidate> candidates(count);
    std::vector<Scalar> nodes(count * 2);
    for (uint32_t c = 0; c < count; ++c) {
        const auto& edge = cases[c % case_count];
        candidates[c].parent = c;
        candidates[c].q[0] = Scalar(edge.y);
        candidates[c].q[1] = Scalar(edge.to);
        candidates[c].steps = edge.steps;
        nodes[2*c] = Scalar(edge.y);
        nodes[2*c+1] = Scalar(edge.from);
    }
    m::DeviceBuffer<m::RrtCandidate> device_candidates(count);
    m::DeviceBuffer<Scalar> device_nodes(nodes.size());
    device_nodes.upload(nodes.data(), nodes.size());
    for (auto direction : {m::EdgeDirection::Forward, m::EdgeDirection::Reverse}) {
        for (uint32_t warps : {1u, 3u, 4u, 8u}) {
            device_candidates.upload(candidates.data(), count);
            check_edge_grid<<<1, 32 * warps>>>(device_candidates.data(), count,
                device_nodes.data(), space, scene, direction);
            m::cuda_check(cudaGetLastError(), "check edge grids");
            std::vector<m::RrtCandidate> checked(count);
            device_candidates.download(checked.data(), count);
            for (uint32_t c = 0; c < count; ++c)
                assert(bool(checked[c].blocked) == cases[c % case_count].blocked);
        }
    }
}

__global__ void check_distance_precision(double* output) {
    const Scalar zero[2] = {0, 0};
    const Scalar large[2] = {Scalar(1e20), Scalar(2e20)};
    const Scalar tiny[2] = {Scalar(1e-30), Scalar(2e-30)};
    output[0] = m::rrt_squared_distance(zero, large);
    output[1] = m::rrt_squared_distance(zero, tiny);
    output[2] = m::rrt_squared_distance(large, large);
    m::RrtSpace space{};
    space.resolution[0] = space.resolution[1] = Scalar(1e20);
    const Scalar nodes[] = {Scalar(4e20), 0, Scalar(3e20), 0};
    m::RrtCandidate candidate{};
    m::rrt_extend(candidate, nodes, 2, space, 1e20);
    output[3] = candidate.parent;
    output[4] = candidate.q[0];
    candidate.q[0] = Scalar(1e30);
    candidate.q[1] = 0;
    space.resolution[0] = space.resolution[1] = Scalar(1e-30);
    m::rrt_extend(candidate, zero, 1, space, 1e-30);
    output[5] = candidate.q[0];
}

void check_distance_range() {
    m::DeviceBuffer<double> output(6);
    check_distance_precision<<<1, 1>>>(output.data());
    m::cuda_check(cudaGetLastError(), "check distance precision");
    double values[6];
    output.download(values, 6);
    for (unsigned i = 0; i < 2; ++i) {
        const double a = double(Scalar(i ? 1e-30 : 1e20));
        const double b = double(Scalar(i ? 2e-30 : 2e20));
        assert(std::isfinite(values[i]) && values[i] > 0);
        assert(std::abs(values[i] / (a*a + b*b) - 1) < 1e-12);
    }
    assert(values[2] == 0 && values[3] == 1);
    assert(std::abs(values[4] / 2e20 - 1) < 1e-6);
    assert(std::abs(values[5] / 1e-30 - 1) < 1e-6);
}

__global__ void check_cached_nearest(m::RrtSpace space) {
    using Tree = m::RrtTree;
    const Scalar nodes[] = {-5,0, 5,0, 0,0, -1,0, 1,0, Scalar(0.5),0, Scalar(0.1),0, 0,0};
    const Tree trees[] = {Tree::Start, Tree::Goal, Tree::Start, Tree::Goal,
                          Tree::Goal, Tree::Goal, Tree::Start, Tree::Goal};
    const uint32_t snapshots[] = {2, 2, 3, 5, 5, 6, 7, 8};
    for (Tree tree : {Tree::Start, Tree::Goal}) {
        for (Scalar target : {Scalar(0), Scalar(2.2)}) {
            m::RrtNearestCache cache{0, 0, INFINITY};
            for (uint32_t snapshot : snapshots) {
                m::RrtCandidate cached{}, scanned{};
                cached.q[0] = scanned.q[0] = target;
                cached.reaches_target = scanned.reaches_target = true;
                m::rrt_extend(cached, nodes, snapshot, space, 0.25, trees, tree, &cache);
                m::rrt_extend(scanned, nodes, snapshot, space, 0.25, trees, tree);
                assert(cached.parent == scanned.parent && cache.node == scanned.parent);
                assert(cache.scanned == snapshot);
                assert(cached.steps == scanned.steps && cached.blocked == scanned.blocked);
                assert(cached.reaches_target == scanned.reaches_target);
                for (unsigned d = 0; d < m::rrt_dimension; ++d) assert(cached.q[d] == scanned.q[d]);
                if (tree == Tree::Goal && target == 0 && snapshot == 5) assert(cached.parent == 3);
            }
        }
    }
}

int main() try {
    check_distance_range();
    const m::Box boxes[] = {
        {{{0,0,Scalar(0.25)}, {1,0,0,0,1,0,0,0,1}}, {Scalar(0.2),Scalar(0.8),Scalar(0.2)}},
        {{{0,0,Scalar(0.25)}, {1,0,0,0,1,0,0,0,1}}, {Scalar(0.2),4,4}},
    };
    const m::Environment environments[] = {
        {{0,0},{0,0},{0,0},{0,0}}, {{0,1},{0,0},{0,0},{0,0}}, {{1,1},{0,0},{0,0},{0,0}},
    };
    m::DeviceBuffer<m::Box> device_boxes(2);
    m::DeviceBuffer<m::Environment> device_environments(3);
    device_boxes.upload(boxes, 2);
    device_environments.upload(environments, 3);
    const m::Scene scene = {device_environments.data(), device_boxes.data(), nullptr, nullptr, nullptr};
    m::RrtSpace space{{-1,-1}, {1,1}, {Scalar(0.025),Scalar(0.025)}, {0,0,Scalar(0.25)}};
    check_edge_grids(space, scene);
    check_cached_nearest<<<1, 1>>>(space);
    m::cuda_check(cudaGetLastError(), "check cached nearest nodes");
    m::cuda_check(cudaDeviceSynchronize(), "validate cached nearest nodes");
    rkern::RrtConfig config;
    config.max_nodes = 1024;
    config.samples_per_round = 64;
    config.max_rounds = 128;
    config.step_size = 0.35;
    config.goal_bias = 0.2;
    std::vector<m::RrtProblem> problems(9);
    for (auto& problem : problems) {
        problem.environment_id = 0;
        problem.seed = 317;
        problem.start[0] = 0; problem.start[1] = Scalar(-0.8);
        problem.goal[0] = 0; problem.goal[1] = Scalar(0.8);
    }
    problems[0].goal[1] = problems[0].start[1];
    problems[2].environment_id = 1;
    problems[3].environment_id = 2;
    problems[4].start[0] = 2;
    problems[5].goal[0] = std::numeric_limits<Scalar>::quiet_NaN();
    problems[6].environment_id = 1; problems[6].start[1] = 0;
    problems[7].environment_id = 1; problems[7].goal[1] = 0;
    for (Algorithm algorithm : {Algorithm::Rrt, Algorithm::RrtConnect}) {
        std::vector<std::vector<Scalar>> reference;
        std::vector<rkern::RrtResult> reference_results;
        for (uint32_t warps : {1u, 3u, 4u, 8u}) {
            m::RrtPlanner planner(config, {warps}, space, scene, 3, problems.size(), algorithm);
            rejects([&] { planner.path(0); });
            for (int repetition = 0; repetition < 2; ++repetition) {
                const auto result = planner.solve(problems);
                assert(result[0].status == Status::Success && result[0].path_length == 1 && result[0].rounds == 0);
                assert(result[1].status == Status::Success && result[2].status == Status::Success);
                assert(result[3].status == Status::NodeLimit || result[3].status == Status::RoundLimit);
                assert(result[4].status == Status::InvalidStart && result[5].status == Status::InvalidGoal);
                assert(result[6].status == Status::InvalidStart && result[7].status == Status::InvalidGoal);
                assert(result[8].status == Status::Success);
                if (algorithm == Algorithm::RrtConnect) {
                    assert(result[1].rounds == 1);
                    // This seed connects while the goal tree is active, exercising
                    // the opposite parent-chain order from the empty scene.
                    assert(result[2].rounds % 2 == 0);
                }
                std::vector<std::vector<Scalar>> paths;
                for (size_t p = 0; p < problems.size(); ++p) {
                    paths.push_back(planner.path(p));
                    if (result[p].status == Status::Success) validate_path(paths.back(), problems[p], space, config, algorithm);
                    else assert(paths.back().empty() && result[p].path_length == 0);
                    assert(result[p].node_count <= config.max_nodes);
                }
                assert(paths[1] == paths[8]);
                if (reference.empty()) {
                    reference = paths;
                    reference_results = result;
                } else {
                    assert(paths == reference);
                    for (size_t p = 0; p < result.size(); ++p) {
                        assert(result[p].status == reference_results[p].status);
                        assert(result[p].node_count == reference_results[p].node_count);
                        assert(result[p].rounds == reference_results[p].rounds);
                        assert(result[p].path_length == reference_results[p].path_length);
                    }
                }
            }
            auto invalid = problems;
            invalid[0].environment_id = 3;
            rejects([&] { planner.solve(invalid); });
            invalid.pop_back();
            rejects([&] { planner.solve(invalid); });
        }
    }
    rejects([&] { m::RrtPlanner p(config, {0}, space, scene, 3, 1); });
    rejects([&] { m::RrtPlanner p(config, {33}, space, scene, 3, 1); });
    auto bad_space = space;
    bad_space.resolution[0] = 0;
    rejects([&] { m::RrtPlanner p(config, {1}, bad_space, scene, 3, 1); });
    config.samples_per_round = 1;
    config.max_rounds = 1;
    config.goal_bias = 0;
    config.step_size = 0.01;
    for (Algorithm algorithm : {Algorithm::Rrt, Algorithm::RrtConnect}) {
        m::RrtPlanner round_limit(config, {1}, space, scene, 3, 1, algorithm);
        assert(round_limit.solve({problems[3]})[0].status == Status::RoundLimit);
        auto small = config;
        small.max_nodes = 2;
        m::RrtPlanner node_limit(small, {1}, space, scene, 3, 1, algorithm);
        assert(node_limit.solve({problems[1]})[0].status == Status::NodeLimit);
        const auto equal = node_limit.solve({problems[0]})[0];
        assert(equal.status == Status::Success && equal.path_length == 1);
    }
    config.goal_bias = 1;
    config.step_size = 0.05;
    m::RrtPlanner connect(config, {4}, space, scene, 3, 1, Algorithm::RrtConnect);
    const auto connected = connect.solve({problems[1]})[0];
    assert(connected.status == Status::Success && connected.rounds == 1 && connected.path_length > 30);
    validate_path(connect.path(0), problems[1], space, config, Algorithm::RrtConnect);
    for (uint32_t capacity : {3u, 5u, 10u}) {
        auto small = config;
        small.max_nodes = capacity;
        small.samples_per_round = capacity - 1;
        m::RrtPlanner limit(small, {4}, space, scene, 3, 1, Algorithm::RrtConnect);
        const auto result = limit.solve({problems[1]})[0];
        assert(result.status == Status::NodeLimit && result.node_count == capacity);
    }
    // The active tree can reach an existing opposite root even with no free slot.
    config.max_nodes = 3;
    config.step_size = 2;
    m::RrtPlanner root_join(config, {1}, space, scene, 3, 1, Algorithm::RrtConnect);
    const auto joined = root_join.solve({problems[1]})[0];
    assert(joined.status == Status::Success && joined.path_length == 2 && joined.node_count == 3);
    validate_path(root_join.path(0), problems[1], space, config, Algorithm::RrtConnect);
    std::cout << "RRT / RRT-Connect paths, limits, environment mapping, replay and 1/3/4/8 warps passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
