#include "broad_phase_checks.h"

bool check_sdfs() {
    const sdf::Pose identity = {{0,0,0}, {1,0,0,0,1,0,0,0,1}};
    const sdf::Box box = {identity, {2,4,6}};
    const sdf::Cylinder cylinder = {identity, 1,4};
    const sdf::Sphere sphere = {{1,2,3},2};
    const sdf::Ground ground = {2};
    const sdf::Box rotated_box = {{{1,2,3}, {0,1,0,-1,0,0,0,0,1}}, {2,4,6}};
    const sdf::Cylinder rotated_cylinder = {{{1,2,3}, {0,0,-1,0,1,0,1,0,0}}, 1,4};
    const double checks[][2] = {
        {sdf::distance(box, {0,0,0}), -1},
        {sdf::distance(box, {1,2,3}), 0},
        {sdf::distance(box, {4,6,3}), 5},
        {sdf::distance(box, {-4,-6,-3}), 5},
        {sdf::distance(cylinder, {0,0,0}), -1},
        {sdf::distance(cylinder, {1,0,2}), 0},
        {sdf::distance(cylinder, {4,0,6}), 5},
        {sdf::distance(cylinder, {0,0,-3}), 1},
        {sdf::distance(sphere, {1,2,3}), -2},
        {sdf::distance(sphere, {1,4,3}), 0},
        {sdf::distance(sphere, {4,6,3}), 3},
        {sdf::distance(ground, {10,20,-2}), 0},
        {sdf::distance(ground, {10,20,-3}), -1},
        {sdf::distance(ground, {10,20,0}), 2},
        {sdf::distance(rotated_box, {7,6,6}), 5},
        {sdf::distance(rotated_cylinder, {7,2,7}), 5},
    };
    for (size_t i = 0; i < sizeof(checks)/sizeof(checks[0]); ++i) {
        if (!std::isfinite(checks[i][0]) || std::abs(checks[i][0] - checks[i][1]) > tolerance) {
            std::fprintf(stderr, "SDF check %zu: %.17g != %.17g\n", i, checks[i][0], checks[i][1]);
            return false;
        }
    }
    const sdf::Environment no_obstacles = {};
    const sdf::Scene empty = {&no_obstacles, nullptr, nullptr, nullptr, nullptr};
    if (!sdf::has_ground && sdf::distance(empty, 0, {0,0,0}) != INFINITY) return false;
    const Scalar large = Scalar(1e6);
    const sdf::Sphere large_sphere = {{0,0,0},large};
    const bool intersections[][2] = {
        {sdf::intersects(box, {0,0,0}, 0), true},
        {sdf::intersects(box, {1,2,3}, 0), true},
        {sdf::intersects(box, {Scalar(1.25),2,3}, 0), false},
        {sdf::intersects(box, {4,6,3}, 5), true},
        {sdf::intersects(box, {4,6,3}, Scalar(4.75)), false},
        {sdf::intersects(rotated_box, {7,6,6}, 5), true},
        {sdf::intersects(rotated_box, {7,6,6}, Scalar(4.75)), false},
        {sdf::intersects(sphere, {1,2,3}, 0), true},
        {sdf::intersects(sphere, {1,4,3}, 0), true},
        {sdf::intersects(sphere, {4,6,3}, 3), true},
        {sdf::intersects(sphere, {4,6,3}, Scalar(2.75)), false},
        {sdf::intersects(rotated_cylinder, {7,2,7}, 5), true},
        {sdf::intersects(rotated_cylinder, {7,2,7}, Scalar(4.75)), false},
        {sdf::intersects(ground, {10,20,-2}, 0), true},
        {sdf::intersects(ground, {10,20,0}, 2), true},
        {sdf::intersects(ground, {10,20,0}, Scalar(1.75)), false},
        {sdf::intersects(box, {4*large,0,0}, large), false},
        {sdf::intersects(large_sphere, {4*large,0,0}, large), false},
    };
    for (size_t i = 0; i < sizeof(intersections)/sizeof(intersections[0]); ++i) {
        if (intersections[i][0] != intersections[i][1]) {
            std::fprintf(stderr, "intersection check %zu: %d != %d\n", i,
                         int(intersections[i][0]), int(intersections[i][1]));
            return false;
        }
    }
    for (unsigned int i=0; i<broad_phase_case_count; ++i) {
        if (!check_broad_phase_case(i)) {
            std::fprintf(stderr, "broad phase case %u failed\n", i);
            return false;
        }
    }
    return true;
}
