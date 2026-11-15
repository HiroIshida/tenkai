#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "rrt/planner.cuh"
#else
#define __device__
#define __host__
#define __global__
#define __sinf sinf
#define __cosf cosf
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{}, threadIdx{};
#include "collision.cuh"
#endif

namespace m = tenkai_f32;

__host__ __device__ bool expected_self(const float* q) {
    for (int a = 0; a < TEST_MEMBERS; ++a) {
        for (int b = 0; b < TEST_MEMBERS; ++b) {
            const float dx = q[0];
            const float dy = (q[1] + b * 0.5f) - a * 0.5f;
            const float dz = q[2];
            if (dx * dx + dy * dy + dz * dz <= 0.0625f) return true;
        }
    }
    return false;
}

__device__ void check_predicates() {
    const m::Environment env[] = {{}, {{0,1},{},{},{0,1}}, {{},{0,1},{},{}}};
    const m::Box box = {{{100,100,100}, {1,0,0,0,1,0,0,0,1}}, {1,1,1}};
    const m::Ground ground{10};
    m::Sphere obstacle{{0,0,0}, 0.125f};
    const m::Scene scene{env, &box, &obstacle, nullptr, &ground};
    for (float x : {-1.0f, -0.25f, 0.0f, 0.2f, 0.25f, nextafterf(0.25f, 0.0f),
                    nextafterf(0.25f, 1.0f), 1.0f}) {
        for (float y : {0.0f, 0.25f, 0.5f}) {
            const float q[] = {x, y, 0};
            const bool expected = expected_self(q);
            for (unsigned id : {0u, 1u}) {
                assert(tenkai_collision_one(q, scene, id) == expected);
#if defined(__CUDACC__) || defined(__HIPCC__)
                assert(tenkai_collision_moving(q, scene, id) == expected);
#endif
            }
        }
    }
    const float separated[] = {2, 0, 0};
    assert(!tenkai_collision_one(separated, scene, 2)); // Root is self-only.
    obstacle.position.x = 2;
    assert(tenkai_collision_one(separated, scene, 2));

#if defined(__CUDACC__) || defined(__HIPCC__)
    for (unsigned mask : {1u, 0x80000000u, 15u, 0xaaaaaaaau, 0xffffffffu}) {
        for (unsigned hit_lane : {0u, 15u, 31u, 32u}) {
            const unsigned lane = threadIdx.x;
            const float q[] = {lane == hit_lane ? 0.0f : 2.0f, 0, 0};
            bool blocked = false;
            if (mask & (1u << lane)) {
                blocked = tenkai_collision_moving<true>(q, scene, 1, mask);
            }
            const bool expected = hit_lane < 32 && (mask & (1u << hit_lane));
            assert(bool(__any_sync(0xffffffffu, blocked)) == expected);
        }
    }
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
__global__ void checks() { check_predicates(); }
#endif

int main() {
#if defined(__CUDACC__) || defined(__HIPCC__)
    checks<<<1,32>>>();
    m::cuda_check(cudaGetLastError(), "self collision predicates");
    m::cuda_check(cudaDeviceSynchronize(), "self collision predicates");
    m::Environment env{};
    m::DeviceBuffer<m::Environment> device_env(1);
    device_env.upload(&env, 1);
    const m::Scene scene{device_env.data(), nullptr, nullptr, nullptr, nullptr};
    m::RrtSpace space{};
    for (unsigned i = 0; i < m::rrt_dimension; ++i) {
        space.lower[i] = -2;
        space.upper[i] = 2;
        space.resolution[i] = 0.02f;
    }
    tenkai::RrtConfig config;
    config.max_rounds = 256;
    m::RrtPlanner planner(config, space, scene, 1, 3);
    std::vector<m::RrtProblem> problems(3);
    for (auto& problem : problems) {
        problem.start[0] = -1;
        problem.goal[0] = 1;
        problem.seed = 42;
    }
    problems[1].start[0] = 0;
    problems[2].goal[0] = 0;
    const auto result = planner.solve(problems);
    assert(result[1].status == tenkai::PlanStatus::InvalidStart);
    assert(result[2].status == tenkai::PlanStatus::InvalidGoal);
    if (m::rrt_dimension == 1) {
        assert(result[0].status != tenkai::PlanStatus::Success);
    } else {
        assert(result[0].status == tenkai::PlanStatus::Success);
        const auto path = planner.path(0);
        for (size_t i = 0; i < path.size(); i += m::rrt_dimension)
            assert(!expected_self(path.data() + i));
    }
#else
    check_predicates();
#endif
}
