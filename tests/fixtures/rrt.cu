#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "rrt.cuh"

namespace m = RKERN_MODEL_NAMESPACE;
using Scalar = m::Scalar;
using Status = rkern::PlanStatus;
using Algorithm = rkern::RrtAlgorithm;
namespace sdf = m;
#include "broad_phase_checks.h"

__global__ void check_broad_phase_device(bool* result) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < broad_phase_case_count) result[index] = check_broad_phase_case(index);
}

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
                                m::EdgeDirection direction, bool interleaved) {
    m::RrtProblem problem{};
    problem.environment_id = 1;
    problem.goal[1] = Scalar(0.8);
    m::rrt_check_edges(candidates, count, nodes, problem, space, scene, direction, interleaved);
}

void check_edge_grids(const m::RrtSpace& space, m::Scene scene) {
    struct Edge {
        double from, to, y;
        uint32_t steps;
        bool forward_blocked, reverse_blocked, goal_blocked;
    };
    std::vector<Edge> cases{
        {-0.8, 1.2, 0, 80, true, true, false}, // All quarter probes miss the obstacle.
        {-2.6, 0.6, 0, 65, true, true, false}, // Collision after the first 32 samples.
        {-0.8, 1.2, 0.8, 80, false, false, false},
        {-0.8, 0.8, 0, 1, false, false, false}, // Only the endpoint belongs to this grid.
        {-0.8, 0.8, 0, 2, true, true, false},
        {0.2, 0, 0, 1, true, true, true}, // The new endpoint must be checked in reverse too.
        {0.8, 0.8, 0, 0, false, false, false},
        {-0.9, -0.8, 0, 4, false, false, true},
        {-0.8, 1.2, 0.8, 4097, false, false, false},
    };
    for (uint32_t steps : {1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 65u, 97u}) {
        for (uint32_t hit = 0; hit <= steps + 1; ++hit) {
            // Exactly one grid point can hit; omitting any point must fail.
            cases.push_back({-0.5 * hit, 0.5 * (double(steps) - hit), 0, steps,
                             hit > 0 && hit <= steps, hit <= steps, hit >= steps});
        }
    }
    const uint32_t capacity = static_cast<uint32_t>(cases.size());
    std::vector<m::RrtCandidate> candidates(capacity);
    std::vector<Scalar> nodes(capacity * 2);
    for (uint32_t c = 0; c < capacity; ++c) {
        const auto& edge = cases[c];
        candidates[c].parent = c;
        candidates[c].q[0] = Scalar(edge.y);
        candidates[c].q[1] = Scalar(edge.to);
        candidates[c].steps = edge.steps;
        nodes[2*c] = Scalar(edge.y);
        nodes[2*c+1] = Scalar(edge.from);
    }
    m::DeviceBuffer<m::RrtCandidate> device_candidates(capacity);
    m::DeviceBuffer<Scalar> device_nodes(nodes.size());
    device_nodes.upload(nodes.data(), nodes.size());
    for (auto direction : {m::EdgeDirection::Forward, m::EdgeDirection::Reverse, m::EdgeDirection::Goal}) {
        for (uint32_t warps : {1u, 3u, 4u, 8u}) {
            for (uint32_t count : {0u, 1u, 3u, 4u, 5u, 7u, 8u, 9u, 33u, 65u, capacity}) {
                for (bool interleaved : {false, true}) {
                    device_candidates.upload(candidates.data(), capacity);
                    check_edge_grid<<<1, 32 * warps>>>(device_candidates.data(), count,
                        device_nodes.data(), space, scene, direction, interleaved);
                    m::cuda_check(cudaGetLastError(), "check edge grids");
                    std::vector<m::RrtCandidate> checked(capacity);
                    device_candidates.download(checked.data(), capacity);
                    for (uint32_t c = 0; c < capacity; ++c) {
                        const bool goal = direction == m::EdgeDirection::Goal;
                        const bool blocked = goal ? checked[c].goal_blocked : checked[c].blocked;
                        const auto& edge = cases[c];
                        const bool expected = goal ? edge.goal_blocked :
                            (direction == m::EdgeDirection::Forward ? edge.forward_blocked : edge.reverse_blocked);
                        assert(blocked == (c < count && expected));
                    }
                }
            }
        }
    }
}

struct StepCase {
    Scalar from[2], to[2], resolution[2];
    uint32_t expected;
};

__global__ void check_step_counts(const StepCase* cases, uint32_t count, uint32_t* output) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    m::RrtSpace space{};
    for (uint32_t d = 0; d < 2; ++d) space.resolution[d] = cases[i].resolution[d];
    output[i] = m::rrt_steps(cases[i].from, cases[i].to, space);
}

void check_interpolation_grid_sizes() {
    std::vector<StepCase> cases{
        {{0, 0}, {0, 0}, {1, 1}, 1},
        {{0, 0}, {0, 0}, {1, 2}, 1},
        {{0, 0}, {1, 2}, {Scalar(0.5), Scalar(0.25)}, 8},
        {{0, 0}, {1, 2}, {Scalar(0.25), Scalar(0.5)}, 4},
        {{-1, -2}, {1, 2}, {Scalar(0.5), Scalar(0.5)}, 8},
    };
    const Scalar small = std::ldexp(Scalar(1), sizeof(Scalar) == sizeof(float) ? -120 : -1000);
    const Scalar large = std::ldexp(Scalar(1), sizeof(Scalar) == sizeof(float) ? 100 : 900);
    for (Scalar resolution : {Scalar(0.25), small, large}) {
        const Scalar boundary = 4 * resolution;
        for (uint32_t d = 0; d < 2; ++d) {
            StepCase edge{{0, 0}, {0, 0}, {resolution, resolution}, 4};
            edge.to[d] = boundary;
            cases.push_back(edge);
            edge.to[d] = std::nextafter(boundary, Scalar(0));
            cases.push_back(edge);
            edge.to[d] = std::nextafter(boundary, std::numeric_limits<Scalar>::infinity());
            edge.expected = 5;
            cases.push_back(edge);
        }
    }
    const Scalar subnormal = std::numeric_limits<Scalar>::denorm_min();
    cases.push_back({{0, 0}, {subnormal, 2 * subnormal}, {subnormal, subnormal}, 2});
    const Scalar fine = std::ldexp(Scalar(1), -31);
    cases.push_back({{0, 0}, {1, 1}, {fine, fine}, UINT32_C(2147483648)});
    m::DeviceBuffer<StepCase> device_cases(cases.size());
    m::DeviceBuffer<uint32_t> output(cases.size());
    device_cases.upload(cases.data(), cases.size());
    check_step_counts<<<1, 32>>>(device_cases.data(), cases.size(), output.data());
    m::cuda_check(cudaGetLastError(), "check interpolation grid sizes");
    std::vector<uint32_t> counts(cases.size());
    output.download(counts.data(), counts.size());
    for (size_t i = 0; i < cases.size(); ++i) assert(counts[i] == cases[i].expected);
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

__global__ void check_grouped_nearest(m::RrtSpace space) {
    using Tree = m::RrtTree;
    constexpr uint32_t capacity = 192, max_queries = 129;
    __shared__ Scalar nodes[capacity * m::rrt_dimension];
    __shared__ Tree trees[capacity];
    __shared__ m::RrtCandidate candidates[max_queries];
    for (uint32_t width : {1u, 2u, 4u, 8u, 16u, 32u}) {
        assert(m::rrtc_nearest_width(31, blockDim.x / width) == 1);
        assert(m::rrtc_nearest_width(32, blockDim.x / width) == width);
    }
    assert(m::rrtc_nearest_width(32, blockDim.x + 1) == 1);
    assert(m::rrtc_nearest_width(capacity, 1) == 32);
    for (Scalar scale : {Scalar(1), Scalar(1e-30), Scalar(1e20)}) {
        for (uint32_t d = 0; d < m::rrt_dimension; ++d) space.resolution[d] = Scalar(0.025) * scale;
        for (uint32_t n = threadIdx.x; n < capacity; n += blockDim.x) {
            Scalar x = Scalar(4 + n % 7);
            if (n == 2) x = -1;
            if (n == 31 || n == 32) x = 1;
            if (n == 64 || n == 65 || n == 95) x = 0;
            nodes[n * m::rrt_dimension] = x * scale;
            nodes[n * m::rrt_dimension + 1] = 0;
            trees[n] = n % 2 ? Tree::Goal : Tree::Start;
        }
        __syncthreads();
        for (uint32_t queries : {1u, 3u, 4u, 8u, 17u, 32u, 65u, max_queries}) {
            for (Tree tree : {Tree::Start, Tree::Goal}) {
                for (bool cached : {false, true}) {
                    for (uint32_t c = threadIdx.x; c < queries; c += blockDim.x)
                        candidates[c].nearest = {0, 0, INFINITY};
                    __syncthreads();
                    uint32_t scanned = 0;
                    for (uint32_t snapshot : {2u, 2u, 31u, 32u, 64u, 65u, 96u, 128u, 160u, capacity}) {
                        const uint32_t width = m::rrtc_nearest_width(snapshot - (cached ? scanned : 0), queries);
                        const uint32_t lane = threadIdx.x & (width - 1);
                        for (uint32_t c = threadIdx.x / width; c < queries; c += blockDim.x / width) {
                            auto& candidate = candidates[c];
                            const Scalar target = Scalar(c % 5) * Scalar(0.25) * scale;
                            if (lane == 0) {
                                candidate.q[0] = target;
                                candidate.q[1] = 0;
                                candidate.reaches_target = true;
                            }
                            if (width > 1) __syncwarp(m::rrt_group_mask(width));
                            m::rrt_extend(candidate, nodes, snapshot, space, double(scale) * 0.25,
                                          trees, tree, cached ? &candidate.nearest : nullptr, width);
                            if (lane == 0) {
                                m::RrtCandidate reference{};
                                reference.q[0] = target;
                                reference.reaches_target = true;
                                m::RrtNearestCache nearest{0, 0, INFINITY};
                                m::rrt_extend(reference, nodes, snapshot, space, double(scale) * 0.25,
                                              trees, tree, &nearest);
                                assert(candidate.parent == reference.parent);
                                assert(candidate.steps == reference.steps && candidate.blocked == reference.blocked);
                                assert(candidate.reaches_target == reference.reaches_target);
                                for (uint32_t d = 0; d < m::rrt_dimension; ++d)
                                    assert(candidate.q[d] == reference.q[d]);
                                if (cached) {
                                    assert(candidate.nearest.node == nearest.node);
                                    assert(candidate.nearest.scanned == snapshot);
                                    assert(candidate.nearest.squared_distance == nearest.squared_distance);
                                }
                                if (c == 0 && tree == Tree::Start && snapshot >= 31)
                                    assert(reference.parent == (snapshot <= 64 ? 2u : 64u));
                                if (c == 0 && tree == Tree::Goal && snapshot >= 32)
                                    assert(reference.parent == (snapshot <= 65 ? 31u : 65u));
                            }
                        }
                        scanned = snapshot;
                        __syncthreads();
                    }
                }
            }
        }
    }
}

void check_single_candidate(const m::RrtSpace& space, m::Scene scene,
                            const std::vector<m::RrtProblem>& problems) {
    for (uint32_t capacity : {2u, 3u, 4u, 16u, 1024u}) {
        for (double step : {0.05, 1.0, 2.0}) {
            for (double bias : {0.0, 1.0}) {
                rkern::RrtConfig config;
                config.samples_per_round = 1;
                config.max_nodes = capacity;
                config.max_rounds = 128;
                config.step_size = step;
                config.goal_bias = bias;
                m::RrtPlanner single(config, {1}, space, scene, 3, problems.size(), Algorithm::RrtConnect);
                m::RrtPlanner general(config, {4}, space, scene, 3, problems.size(), Algorithm::RrtConnect);
                const auto actual = single.solve(problems), expected = general.solve(problems);
                for (size_t i = 0; i < problems.size(); ++i) {
                    assert(actual[i].status == expected[i].status);
                    assert(actual[i].node_count == expected[i].node_count);
                    assert(actual[i].rounds == expected[i].rounds);
                    assert(actual[i].path_length == expected[i].path_length);
                    assert(single.path(i) == general.path(i));
                    if (actual[i].status == Status::Success)
                        validate_path(single.path(i), problems[i], space, config, Algorithm::RrtConnect);
                }
            }
        }
    }
}

int main() try {
    m::DeviceBuffer<bool> broad_phase_result(broad_phase_case_count);
    check_broad_phase_device<<<(broad_phase_case_count+127)/128,128>>>(broad_phase_result.data());
    m::cuda_check(cudaGetLastError(), "check broad phase");
    bool broad_phase_passed[broad_phase_case_count];
    broad_phase_result.download(broad_phase_passed, broad_phase_case_count);
    for (unsigned int i=0; i<broad_phase_case_count; ++i) {
        if (!broad_phase_passed[i]) std::cerr << "broad phase case " << i << " failed\n";
        assert(broad_phase_passed[i]);
    }
    check_distance_range();
    check_interpolation_grid_sizes();
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
    for (uint32_t warps : {1u, 3u, 4u, 8u}) {
        check_grouped_nearest<<<1, 32 * warps>>>(space);
        m::cuda_check(cudaGetLastError(), "check grouped nearest nodes");
        m::cuda_check(cudaDeviceSynchronize(), "validate grouped nearest nodes");
    }
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
    check_single_candidate(space, scene, problems);
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
