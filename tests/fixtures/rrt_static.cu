#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "rrt/planner.cuh"

namespace m = TENKAI_MODEL_NAMESPACE;
using Scalar = m::Scalar;
using Status = tenkai::PlanStatus;

int main() try {
    m::Sphere obstacles[] = {
        {{-5, 0, 0}, Scalar(0.05)},
        {{5, 0, 0}, Scalar(0.05)},
        {{0, 0, Scalar(0.5)}, Scalar(0.05)},
        {{Scalar(2 * std::cos(0.5)), Scalar(2 * std::sin(0.5)), 0}, Scalar(0.05)},
        {{Scalar(2 * std::cos(0.5)), Scalar(-2 * std::sin(0.5)), 0}, Scalar(0.05)},
        {{2, 0, 0}, Scalar(0.05)},
    };
    const m::Environment environments[] = {
        {{0,0},{0,1},{0,0},{0,0}}, {{0,0},{1,1},{0,0},{0,0}},
        {{0,0},{2,1},{0,0},{0,0}}, {{0,0},{3,1},{0,0},{0,0}},
        {{0,0},{4,1},{0,0},{0,0}}, {}, {{0,0},{5,1},{0,0},{0,0}}
    };
    m::DeviceBuffer<m::Sphere> device_obstacles(6);
    m::DeviceBuffer<m::Environment> device_environments(7);
    device_obstacles.upload(obstacles, 6);
    device_environments.upload(environments, 7);
    const m::Scene scene = {device_environments.data(), nullptr, device_obstacles.data(), nullptr, nullptr};
    m::RrtSpace space{{-1}, {1}, {Scalar(0.025)}, {13, 0, 0}};
    tenkai::RrtConfig config;
    config.max_nodes = 128;
    config.max_rounds = 8;
    config.goal_bias = 1;
    config.step_size = 0.25;
    const uint32_t ids[] = {5, 1, 3, 0, 4, 2, 6};
    std::vector<m::RrtProblem> problems(7);
    for (size_t i = 0; i < problems.size(); ++i) {
        problems[i].environment_id = ids[i];
        problems[i].seed = 317;
        problems[i].start[0] = Scalar(-0.5);
        problems[i].goal[0] = Scalar(0.5);
    }
    m::RrtPlanner planner(config, space, scene, 7, problems.size());
    const auto results = planner.solve(problems);
    for (size_t i = 0; i < results.size(); ++i) {
        Status expected = Status::Success;
        if (TEST_SPHERE_COUNT >= 3 && ids[i] < 3) expected = Status::InvalidStart;
        if (TEST_SPHERE_COUNT == 4 && ids[i] == 3) expected = Status::InvalidGoal;
        if (TEST_SPHERE_COUNT == 4 && ids[i] == 4) expected = Status::InvalidStart;
        if (TEST_SPHERE_COUNT == 4 && ids[i] == 6) expected = Status::RoundLimit;
        assert(results[i].status == expected);
        const auto path = planner.path(i);
        if (expected == Status::Success) {
            assert(path.front() == problems[i].start[0] && path.back() == problems[i].goal[0]);
        } else {
            assert(path.empty());
        }
    }
    // A later solve must observe scene changes even when only fixed spheres can hit.
    obstacles[1].position.x = 6;
    device_obstacles.upload(obstacles, 6);
    assert(planner.solve(problems)[1].status == Status::Success);
    obstacles[1].position.x = 5;
    device_obstacles.upload(obstacles, 6);
    assert(planner.solve(problems)[1].status ==
        (TEST_SPHERE_COUNT ? Status::InvalidStart : Status::Success));
    auto shifted = space;
    shifted.reference[1] = 1;
    shifted.reference[2] = 1;
    m::RrtPlanner shifted_planner(config, shifted, scene, 7, problems.size());
    const auto shifted_results = shifted_planner.solve(problems);
    assert(shifted_results[1].status == Status::Success);
    assert(shifted_results[2].status == Status::Success && shifted_results[4].status == Status::Success);
    assert(shifted_results[6].status == Status::Success);
    obstacles[1].position.x = 6;
    device_obstacles.upload(obstacles, 6);
    assert(shifted_planner.solve(problems)[1].status ==
        (TEST_SPHERE_COUNT ? Status::InvalidStart : Status::Success));
    obstacles[1].position.x = 5;
    device_obstacles.upload(obstacles, 6);
    // Standalone queries retain all spheres, independently of planning controls.
    Scalar configurations[21]{};
    for (size_t i = 0; i < 7; ++i) configurations[3*i] = Scalar(-0.5);
    m::DeviceBuffer<Scalar> device_configurations(21);
    m::DeviceBuffer<uint32_t> device_ids(7);
    m::DeviceBuffer<unsigned char> device_flags(7);
    device_configurations.upload(configurations, 21);
    device_ids.upload(ids, 7);
    tenkai_collision<<<1, 32>>>(device_configurations.data(), device_ids.data(), scene, device_flags.data(), 7);
    m::cuda_check(cudaGetLastError(), "check standalone fixed spheres");
    unsigned char flags[7];
    device_flags.download(flags, 7);
    for (size_t i = 0; i < 7; ++i)
        assert(bool(flags[i]) == ((TEST_SPHERE_COUNT >= 3 && ids[i] < 3) ||
                                 (TEST_SPHERE_COUNT == 4 && ids[i] == 4)));
    std::cout << "Fixed sphere initialization, reference values, scene updates and empty models passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
