// Copyright (C) 2024 Hirokazu Ishida. SPDX-License-Identifier: MPL-2.0
// Adapted from plainmp/collision/primitive_sdf.hpp; see licenses/tenkai2-MPL-2.0.txt.
// License text: https://mozilla.org/MPL/2.0/.
#ifndef @NAMESPACE@_SDF_INCLUDED
#define @NAMESPACE@_SDF_INCLUDED
#include <stddef.h>
#include <stdint.h>
#include <math.h>

namespace @NAMESPACE@ {
using Scalar = @SCALAR@;

struct Vec3 { Scalar x, y, z; };
struct Pose {
    Vec3 position;
    Scalar world_to_local[9]; // Row-major inverse rigid rotation.
};
struct Box { Pose pose; Vec3 extents; }; // Full side lengths.
struct Sphere { Vec3 position; Scalar radius; };
struct Cylinder { Pose pose; Scalar radius, height; }; // Local Z axis, full height.
struct Ground { Scalar height; }; // Surface z = -height; negative side is occupied.
struct Range { size_t offset, count; };
struct Environment { Range boxes, spheres, cylinders, grounds; };
struct Scene {
    const Environment* environments;
    const Box* boxes;
    const Sphere* spheres;
    const Cylinder* cylinders;
    const Ground* grounds;
};

__device__ inline Vec3 local_point(const Pose& pose, Vec3 point) {
    const Scalar x = point.x - pose.position.x;
    const Scalar y = point.y - pose.position.y;
    const Scalar z = point.z - pose.position.z;
    const Scalar* r = pose.world_to_local;
    return {r[0]*x + r[1]*y + r[2]*z,
            r[3]*x + r[4]*y + r[5]*z,
            r[6]*x + r[7]*y + r[8]*z};
}

__device__ inline Vec3 box_face_distances(const Box& box, Vec3 point) {
    const Vec3 p = local_point(box.pose, point);
    return {fabs@SUFFIX@(p.x) - box.extents.x * Scalar(0.5),
            fabs@SUFFIX@(p.y) - box.extents.y * Scalar(0.5),
            fabs@SUFFIX@(p.z) - box.extents.z * Scalar(0.5)};
}

__device__ inline Scalar squared_outside_distance(Vec3 face_distances) {
    const Scalar x = fmax@SUFFIX@(face_distances.x, Scalar(0));
    const Scalar y = fmax@SUFFIX@(face_distances.y, Scalar(0));
    const Scalar z = fmax@SUFFIX@(face_distances.z, Scalar(0));
    return x*x + y*y + z*z;
}

__device__ inline Scalar squared_distance(Vec3 a, Vec3 b) {
    const Scalar x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return x*x + y*y + z*z;
}

__device__ inline Scalar distance(const Box& box, Vec3 point) {
    const Vec3 d = box_face_distances(box, point);
    return sqrt@SUFFIX@(squared_outside_distance(d))
        + fmin@SUFFIX@(fmax@SUFFIX@(d.x, fmax@SUFFIX@(d.y, d.z)), Scalar(0));
}

__device__ inline Scalar distance(const Sphere& sphere, Vec3 point) {
    return sqrt@SUFFIX@(squared_distance(point, sphere.position)) - sphere.radius;
}

__device__ inline Scalar distance(const Cylinder& cylinder, Vec3 point) {
    const Vec3 p = local_point(cylinder.pose, point);
    const Scalar radial = sqrt@SUFFIX@(p.x*p.x + p.y*p.y) - cylinder.radius;
    const Scalar axial = fabs@SUFFIX@(p.z) - cylinder.height * Scalar(0.5);
    const Scalar r = fmax@SUFFIX@(radial, Scalar(0)), z = fmax@SUFFIX@(axial, Scalar(0));
    return sqrt@SUFFIX@(r*r + z*z) + fmin@SUFFIX@(fmax@SUFFIX@(radial, axial), Scalar(0));
}

__device__ inline Scalar distance(const Ground& ground, Vec3 point) {
    return point.z + ground.height;
}

__device__ inline bool intersects(const Box& box, Vec3 point, Scalar radius) {
    const Scalar squared_radius = radius * radius;
    // Finite radii can overflow when squared; infinity <= infinity would report a hit.
    if (isinf(squared_radius)) return distance(box, point) <= radius;
    return squared_outside_distance(box_face_distances(box, point)) <= squared_radius;
}

__device__ inline bool intersects(const Sphere& sphere, Vec3 point, Scalar radius) {
    const Scalar sum = sphere.radius + radius;
    const Scalar squared_radius = sum * sum;
    if (isinf(squared_radius)) return distance(sphere, point) <= radius;
    return squared_distance(point, sphere.position) <= squared_radius;
}

__device__ inline bool intersects(const Cylinder& cylinder, Vec3 point, Scalar radius) {
    return distance(cylinder, point) <= radius;
}

__device__ inline bool intersects(const Ground& ground, Vec3 point, Scalar radius) {
    return distance(ground, point) <= radius;
}

__device__ inline Scalar distance(const Scene& scene, const Environment& env, Vec3 point) {
    Scalar result = INFINITY;
    // Keep reductions unconditional; short-circuit tests would skip remaining obstacles.
    for (size_t i = 0; i < env.boxes.count; ++i)
        result = fmin@SUFFIX@(result, distance(scene.boxes[env.boxes.offset + i], point));
    for (size_t i = 0; i < env.spheres.count; ++i)
        result = fmin@SUFFIX@(result, distance(scene.spheres[env.spheres.offset + i], point));
    for (size_t i = 0; i < env.cylinders.count; ++i)
        result = fmin@SUFFIX@(result, distance(scene.cylinders[env.cylinders.offset + i], point));
    for (size_t i = 0; i < env.grounds.count; ++i)
        result = fmin@SUFFIX@(result, distance(scene.grounds[env.grounds.offset + i], point));
    return result;
}
} // namespace @NAMESPACE@
#endif
