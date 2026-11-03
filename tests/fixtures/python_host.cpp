#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#define __device__
#define __global__
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{}, threadIdx{};
#include "fk.cuh"
#include "centers.cuh"
#include "collision.cuh"
#include "config.h"
#include "sdf_checks.h"

int main() {
    if (!check_sdfs()) return 1;
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
    Scalar poses[28], centers[12];
    unsigned char flags[4];
    std::fill_n(poses, 28, Scalar(123));
    std::fill_n(centers, 12, Scalar(123));
    std::fill_n(flags, 4, 123);
    blockDim.x = 4;
    for (threadIdx.x = 0; threadIdx.x < 4; ++threadIdx.x) {
        rkern_fk(q, poses, 3);
        rkern_centers(q, centers, 3);
        rkern_collision(q, ids, scene, flags, 3);
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 7; ++j)
            if (std::abs(poses[i*7+j] - expected[i][j]) > tolerance) return 2;
        for (size_t j = 0; j < 3; ++j)
            if (std::abs(centers[i*3+j] - expected[i][j]) > tolerance) return 3;
        if (flags[i] != (i == 1 ? 0 : 1)) return 4;
    }
    for (size_t i = 21; i < 28; ++i) if (poses[i] != 123) return 5;
    for (size_t i = 9; i < 12; ++i) if (centers[i] != 123) return 6;
    return flags[3] == 123 ? 0 : 7;
}
