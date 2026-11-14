#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#define __device__
#define __global__
#define __sinf sinf
#define __cosf cosf
struct Dim { unsigned int x; };
Dim blockIdx{}, blockDim{1}, threadIdx{};
#include "centers.cuh"
#include "collision.cuh"
#include "reference.h"

int main() {
    namespace m = tenkai_f32;
    const m::Environment env[] = {{}, {{0,1},{0,1},{0,1},{0,1}}};
    const m::Box box = {{{0.95f,0,0.8f}, {1,0,0,0,1,0,0,0,1}}, {1,2,0.05f}};
    const m::Sphere obstacle{{0.1f,0.5f,0.8f}, 0.1f};
    const m::Cylinder cylinder{{{0.5f,-0.5f,0.8f}, {1,0,0,0,1,0,0,0,1}}, 0.1f, 0.3f};
    const m::Ground ground{0};
    const m::Scene scene{env, &box, &obstacle, &cylinder, &ground};
    uint32_t state = 42;
    unsigned hits = 0;
    for (unsigned sample = 0; sample < 4096; ++sample) {
        float q[15];
        for (float& value : q) {
            state = state * 1664525u + 1013904223u;
            value = float(state >> 8) * (1.0f / 16777216.0f) * 4 - 2;
        }
        float centers[sphere_count * 3];
        tenkai_centers(q, centers, 1);
        for (unsigned id : {0u, 1u}) {
            bool expected = false;
            for (unsigned i = 0; i < sphere_count; ++i) {
                const m::Vec3 center{centers[i*3], centers[i*3+1], centers[i*3+2]};
                if (id == 1 && external[i]) {
                    expected |= m::intersects(box, center, radii[i]);
                    expected |= m::intersects(obstacle, center, radii[i]);
                    expected |= m::intersects(cylinder, center, radii[i]);
                    expected |= m::intersects(ground, center, radii[i]);
                }
                for (unsigned j = i + 1; j < sphere_count; ++j) {
                    if (!self_pairs[i][j]) continue;
                    const m::Sphere sphere{center, radii[i]};
                    expected |= m::intersects(sphere,
                        {centers[j*3], centers[j*3+1], centers[j*3+2]}, radii[j]);
                }
            }
            hits += id == 0 && expected;
            assert(tenkai_collision_one(q, scene, id) == expected);
        }
    }
    assert(hits > 0 && hits < 4096);
}
