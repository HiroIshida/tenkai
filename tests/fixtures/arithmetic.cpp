#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#include <stdexcept>
#else
#define __device__
#define __global__
// Host tests check generated arithmetic; intrinsic accuracy is tested on GPUs.
#define __sinf sinf
#define __cosf cosf
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{}, threadIdx{};
#endif
#include "fk.cuh"
#include "centers.cuh"
#include "collision.cuh"
#include "config.h"
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include "sdf_checks.h"
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T> class Buffer {
public:
    T* data = nullptr;
    Buffer(const T* input, size_t count) : bytes_(count * sizeof(T)) {
        check(cudaMalloc(&data, bytes_));
        check(cudaMemcpy(data, input, bytes_, cudaMemcpyHostToDevice));
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void download(T* output) {
        check(cudaMemcpy(output, data, bytes_, cudaMemcpyDeviceToHost));
    }
private:
    size_t bytes_;
};
#endif

__global__ void check_grouped_collisions(const Scalar* configurations, const Scalar* centers,
                                       unsigned char* results, size_t count) {
    const size_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= count) return;
    const Scalar* q = configurations + batch * 4;
    const Scalar* points = centers + batch * sphere_count * 3;
    const sdf::Environment environment = {{0,0},{0,1},{0,0},{0,0}};

    bool valid = true;
    // Isolated probes prevent an earlier hit from hiding a skipped sphere.
    for (size_t i = 0; i < sphere_count; ++i) {
        for (Scalar factor : {0.5f, 1.5f}) {
            const sdf::Sphere obstacle = {
                {points[3*i], points[3*i+1], points[3*i+2] + factor * radii[i]}, 0};
            const sdf::Scene scene = {&environment, nullptr, &obstacle, nullptr, nullptr};
            valid &= tenkai_collision_one(q, scene, 0) == (factor < 1);
        }
    }
    results[batch] = valid;
}

int main() {
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    if (!check_sdfs()) return 1;
#endif
    const Scalar q[] = {0,0,0,0.5, Scalar(1.5707963267948966),0,0,0.5,
                        0,Scalar(1.5707963267948966),0,0.5};
    const double expected[][7] = {
        {6.5,1,0, 0,0, std::sqrt(0.5),std::sqrt(0.5)},
        {0,5.5,0, 0,0,1,0}, {3,1,-3.5, 0.5,0.5,0.5,0.5}
    };
    const sdf::Sphere obstacle = {{6.5,1,0},0.25};
    const sdf::Ground ground = {1};
    const sdf::Environment environment = {{0,0},{0,1},{0,0},{0,1}};
    const sdf::Scene scene = {&environment, nullptr, &obstacle, nullptr, &ground};
    const uint32_t ids[] = {0,0,0};
    const double directions[][3] = {{0,1,0}, {-1,0,0}, {0,1,0}};
    constexpr size_t center_count = 4 * sphere_count * 3;
    Scalar poses[28], centers[center_count];
    unsigned char flags[4], grouped[4];
    std::fill_n(poses, 28, Scalar(123));
    std::fill_n(centers, center_count, Scalar(123));
    std::fill_n(flags, 4, 123);
    std::fill_n(grouped, 4, 123);
#if defined(__CUDACC__) || defined(__HIPCC__)
    Buffer<Scalar> device_q(q, 12), device_poses(poses, 28), device_centers(centers, center_count);
    Buffer<uint32_t> device_ids(ids, 3);
    Buffer<unsigned char> device_flags(flags, 4), device_grouped(grouped, 4);
    Buffer<sdf::Sphere> device_obstacle(&obstacle, 1);
    Buffer<sdf::Ground> device_ground(&ground, 1);
    Buffer<sdf::Environment> device_environment(&environment, 1);
    const sdf::Scene device_scene = {device_environment.data, nullptr, device_obstacle.data,
                                     nullptr, device_ground.data};
    tenkai_fk<<<1, 32>>>(device_q.data, device_poses.data, 3);
    check(cudaGetLastError());
    tenkai_centers<<<1, 32>>>(device_q.data, device_centers.data, 3);
    check(cudaGetLastError());
    tenkai_collision<<<1, 32>>>(device_q.data, device_ids.data, device_scene, device_flags.data, 3);
    check(cudaGetLastError());
    check_grouped_collisions<<<1, 32>>>(device_q.data, device_centers.data, device_grouped.data, 3);
    check(cudaGetLastError());
    device_poses.download(poses);
    device_centers.download(centers);
    device_flags.download(flags);
    device_grouped.download(grouped);
#else
    blockDim.x = 4;
    for (threadIdx.x = 0; threadIdx.x < 4; ++threadIdx.x) {
        tenkai_fk(q, poses, 3);
        tenkai_centers(q, centers, 3);
        tenkai_collision(q, ids, scene, flags, 3);
        check_grouped_collisions(q, centers, grouped, 3);
    }
#endif
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 7; ++j)
            if (std::abs(poses[i*7+j] - expected[i][j]) > tolerance) return 2;
        for (size_t sphere = 0; sphere < sphere_count; ++sphere) {
            const double offset = sphere == 0 ? 0 : sphere - 1;
            for (size_t axis = 0; axis < 3; ++axis) {
                double position = expected[i][axis] + offset * directions[i][axis];
                if (sphere == 1) position = axis == 2 ? 2 : 0;
                if (std::abs(centers[(i*sphere_count+sphere)*3+axis] - position) > tolerance)
                    return 3;
            }
        }
        if (flags[i] != (i == 1 ? 0 : 1)) return 4;
        if (grouped[i] != 1) return 8;
    }
    for (size_t i = 21; i < 28; ++i) if (poses[i] != 123) return 5;
    for (size_t i = 3*sphere_count*3; i < center_count; ++i) if (centers[i] != 123) return 6;
    return flags[3] == 123 && grouped[3] == 123 ? 0 : 7;
}
