#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "rrt.cuh"

namespace m = TENKAI_MODEL_NAMESPACE;
using Scalar = m::Scalar;

void compare(m::RrtPlanner& actual, m::RrtPlanner& reference,
             const std::vector<m::RrtProblem>& problems) {
    const auto a = actual.solve(problems), b = reference.solve(problems);
    for (size_t i = 0; i < problems.size(); ++i) {
        assert(a[i].status == b[i].status);
        assert(a[i].node_count == b[i].node_count);
        assert(a[i].rounds == b[i].rounds);
        assert(a[i].path_length == b[i].path_length);
        assert(actual.path(i) == reference.path(i));
    }
}

#ifdef TEST_STANDARD_FP32
__global__ void check_streaming_votes(m::Scene scene) {
    const unsigned lane = threadIdx.x;
    for (unsigned mask : {1u, 0x80000000u, 15u, 0xaaaaaaaau, 0xffffffffu}) {
        for (unsigned hit_lane : {0u, 15u, 31u, 32u}) {
            const float q[3] = {lane == hit_lane || !(mask & (1u << lane)) ? 0.0f : -0.75f,
                                0.0f, 0.35f};
            bool hit = false;
            if (mask & (1u << lane)) {
                const auto result = tenkai_collision_streaming(q, scene, 0, mask);
                assert(result != m::StreamingCollision::Retry);
                hit = result == m::StreamingCollision::Blocked;
            }
            const bool expected = hit_lane < 32 && (mask & (1u << hit_lane));
            assert(bool(__any_sync(0xffffffffu, hit)) == expected);
        }
    }
    for (float angle : {0.0f, -0.0f, 3.1415927f, -10.0f, 100000.0f, -100000.0f,
                         1.0e10f, FLT_MAX}) {
        const float q[3] = {float(lane) * 0.02f - 0.3f, 0.0f, angle};
        const bool expected = __any_sync(0xffffffffu, tenkai_collision_edge(q, scene, 0));
        const auto result = tenkai_collision_streaming(q, scene, 0, 0xffffffffu);
        assert(result != m::StreamingCollision::Retry);
        assert(bool(__any_sync(0xffffffffu, result == m::StreamingCollision::Blocked)) == expected);
    }
    m::Box box = scene.boxes[0];
    for (float x : {0.0f, 0.18f, nextafterf(0.18f, 0.0f), nextafterf(0.18f, 1.0f),
                    FLT_MIN / 4, FLT_MAX}) {
        const m::Vec3 point{x, 0, 0.35f};
        bool retry = false;
        const bool hit = m::streaming_intersects(box, point, 0.03f, retry);
        assert(!retry && hit == m::intersects(box, point, 0.03f));
    }
    box.pose.position.x = -FLT_MAX;
    bool retry = false;
    m::streaming_intersects(box, {FLT_MAX, 0, 0}, 0.03f, retry);
    assert(retry);
}
#endif

int main() try {
    const Scalar c = Scalar(std::sqrt(0.5));
    m::Box boxes[] = {
        {{{0,0,Scalar(0.3)}, {1,0,0,0,1,0,0,0,1}}, {Scalar(0.3),Scalar(0.8),Scalar(0.6)}},
        {{{0,0,Scalar(0.3)}, {c,c,0,-c,c,0,0,0,1}}, {Scalar(0.3),Scalar(0.8),Scalar(0.6)}},
    };
    const m::Ground ground{0};
    const m::Sphere sphere{{0, Scalar(0.65), Scalar(0.35)}, Scalar(0.1)};
    const m::Cylinder cylinder{{{0,Scalar(-0.65),Scalar(0.3)}, {1,0,0,0,1,0,0,0,1}},
                               Scalar(0.1), Scalar(0.6)};
    m::Environment environments[] = {
        {{0,1},{},{},{0,1}}, {{1,1},{},{},{0,1}}, {{0,2},{},{},{0,1}},
        {{0,1},{0,1},{},{0,1}}, {{0,1},{},{0,1},{0,1}}, {},
        {{0,1},{},{},{}}, {{},{},{},{0,1}},
    };
    m::DeviceBuffer<m::Box> device_boxes(2);
    m::DeviceBuffer<m::Ground> device_ground(1);
    m::DeviceBuffer<m::Sphere> device_sphere(1);
    m::DeviceBuffer<m::Cylinder> device_cylinder(1);
    m::DeviceBuffer<m::Environment> device_environments(8);
    device_boxes.upload(boxes, 2);
    device_ground.upload(&ground, 1);
    device_sphere.upload(&sphere, 1);
    device_cylinder.upload(&cylinder, 1);
    device_environments.upload(environments, 8);
    const m::Scene scene{device_environments.data(), device_boxes.data(), device_sphere.data(),
                         device_cylinder.data(), device_ground.data()};
#ifdef TEST_STANDARD_FP32
    check_streaming_votes<<<1,32>>>(scene);
    m::cuda_check(cudaGetLastError(), "check streaming collision masks and contact");
    m::cuda_check(cudaDeviceSynchronize(), "validate streaming collision masks and contact");
#endif
    const m::RrtSpace space{{Scalar(0.1),-1,-1}, {1,1,1},
                            {Scalar(0.025),Scalar(0.025),Scalar(0.025)}, {0,0,0}};
    tenkai::RrtConfig config;
    config.samples_per_round = 1;
    config.max_nodes = 1024;
    config.max_rounds = 512;
    for (size_t count : {1u, 3u, 4u, 5u, 33u}) {
        std::vector<m::RrtProblem> problems(count);
        for (size_t i = 0; i < count; ++i) {
            auto& p = problems[i];
            p.environment_id = i % 8;
            p.seed = 713 + i;
            p.start[0] = p.goal[0] = Scalar(0.35);
            p.start[1] = Scalar(-0.75); p.goal[1] = Scalar(0.75);
            if (i % 8 == 0 && i >= 8) {
                switch (i / 8) {
                case 1: p.goal[1] = p.start[1]; break;
                case 2: p.start[1] = 0; break;
                case 3: p.goal[1] = 0; break;
                default: p.goal[0] = std::numeric_limits<Scalar>::quiet_NaN();
                }
            }
        }
        for (double step : {0.2, 1.0}) {
            config.step_size = step;
            config.goal_bias = step == 1.0 ? 0.0 : 0.2;
            m::RrtPlanner actual(config, {1}, space, scene, 8, count, tenkai::RrtAlgorithm::RrtConnect);
            m::RrtPlanner reference(config, {4}, space, scene, 8, count, tenkai::RrtAlgorithm::RrtConnect);
            compare(actual, reference, problems);
            boxes[0].pose.world_to_local[0] = c;
            boxes[0].pose.world_to_local[1] = c;
            boxes[0].pose.world_to_local[3] = -c;
            boxes[0].pose.world_to_local[4] = c;
            device_boxes.upload(boxes, 2);
            compare(actual, reference, problems);
            boxes[0].pose.world_to_local[0] = boxes[0].pose.world_to_local[4] = 1;
            boxes[0].pose.world_to_local[1] = boxes[0].pose.world_to_local[3] = 0;
            device_boxes.upload(boxes, 2);
            environments[0].spheres = {0,1};
            device_environments.upload(environments, 8);
            compare(actual, reference, problems);
            environments[0].spheres = {};
            device_environments.upload(environments, 8);
            compare(actual, reference, problems);
        }
    }
    std::cout << "Streaming and fallback paths, partial blocks, scene updates and contact passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
