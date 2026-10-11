#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "rrt.cuh"

namespace m = RKERN_MODEL_NAMESPACE;
using Scalar = m::Scalar;
using Status = rkern::PlanStatus;

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
                   const m::RrtSpace& space, const rkern::RrtConfig& config) {
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
        if (i + 2 < path.size()) assert(std::sqrt(norm) <= config.step_size + 1e-6);
        for (unsigned int j = 0; j <= segments; ++j) {
            Scalar q[2];
            for (size_t d = 0; d < 2; ++d)
                q[d] = Scalar(double(path[i-2+d]) + (j / segments) * (double(path[i+d])-double(path[i-2+d])));
            assert(!collision(q, problem.environment_id));
        }
    }
}

int main() try {
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
    std::vector<std::vector<Scalar>> reference;
    for (uint32_t warps : {1u, 4u, 8u}) {
        m::RrtPlanner planner(config, {warps}, space, scene, 3, problems.size());
        rejects([&] { planner.path(0); });
        for (int repetition = 0; repetition < 2; ++repetition) {
            const auto result = planner.solve(problems);
            assert(result[0].status == Status::Success && result[0].path_length == 1 && result[0].rounds == 0);
            assert(result[1].status == Status::Success && result[2].status == Status::Success);
            assert(result[3].status == Status::NodeLimit || result[3].status == Status::RoundLimit);
            assert(result[4].status == Status::InvalidStart && result[5].status == Status::InvalidGoal);
            assert(result[6].status == Status::InvalidStart && result[7].status == Status::InvalidGoal);
            assert(result[8].status == Status::Success);
            std::vector<std::vector<Scalar>> paths;
            for (size_t p = 0; p < problems.size(); ++p) {
                paths.push_back(planner.path(p));
                if (result[p].status == Status::Success) validate_path(paths.back(), problems[p], space, config);
                else assert(paths.back().empty() && result[p].path_length == 0);
                assert(result[p].node_count <= config.max_nodes);
            }
            assert(paths[1] == paths[8]);
            if (reference.empty()) reference = paths;
            else assert(paths == reference);
        }
        auto invalid = problems;
        invalid[0].environment_id = 3;
        rejects([&] { planner.solve(invalid); });
        invalid.pop_back();
        rejects([&] { planner.solve(invalid); });
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
    m::RrtPlanner round_limit(config, {1}, space, scene, 3, 1);
    assert(round_limit.solve({problems[3]})[0].status == Status::RoundLimit);
    config.max_nodes = 2;
    m::RrtPlanner node_limit(config, {1}, space, scene, 3, 1);
    assert(node_limit.solve({problems[1]})[0].status == Status::NodeLimit);
    std::cout << "RRT paths, failures, environment mapping, replay and 1/4/8 warps passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
