// Copyright (C) 2024 Hirokazu Ishida. SPDX-License-Identifier: MPL-2.0
// Adapted from plainmp/collision/primitive_sdf.hpp; see
// licenses/tenkai2-MPL-2.0.txt. License text: https://mozilla.org/MPL/2.0/.
#ifndef TPL_NAMESPACE_SDF_INCLUDED
#define TPL_NAMESPACE_SDF_INCLUDED
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#endif
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace TPL_NAMESPACE {
using Scalar = TPL_SCALAR;

struct Vec3 {
  Scalar x, y, z;
};

struct Pose {
  Vec3 position;
  Scalar world_to_local[9];  // Row-major inverse rigid rotation.
};

struct Box {
  Pose pose;
  Vec3 extents;
};  // Full side lengths.

struct Sphere {
  Vec3 position;
  Scalar radius;
};

struct Cylinder {
  Pose pose;
  Scalar radius;
  Scalar height;
};  // Local Z axis, full height.

struct Ground {
  Scalar height;
};  // Surface z = -height; negative side is occupied.

struct Range {
  size_t offset;
  size_t count;
};

struct Environment {
  Range boxes;
  Range spheres;
  Range cylinders;
  Range grounds;
};

struct Scene {
  const Environment* environments;
  const Box* boxes;
  const Sphere* spheres;
  const Cylinder* cylinders;
  const Ground* grounds;
};

// Bounds of the computed centers, expanded by the group's largest sphere
// radius.
struct SphereBounds {
  Vec3 lower;
  Vec3 upper;
  Scalar radius;
};

__device__ inline bool overlaps(Vec3 position,
                                Vec3 half_extents,
                                const SphereBounds& bounds) {
  const Scalar epsilon = sizeof(Scalar) == sizeof(float) ? Scalar(FLT_EPSILON)
                                                         : Scalar(DBL_EPSILON);
  const Scalar minimum =
      sizeof(Scalar) == sizeof(float) ? Scalar(FLT_MIN) : Scalar(DBL_MIN);

  const Scalar scale =
      TPL_FABS(position.x) + TPL_FABS(position.y) + TPL_FABS(position.z) +
      half_extents.x + half_extents.y + half_extents.z + bounds.radius +
      TPL_FMAX(TPL_FABS(bounds.lower.x), TPL_FABS(bounds.upper.x)) +
      TPL_FMAX(TPL_FABS(bounds.lower.y), TPL_FABS(bounds.upper.y)) +
      TPL_FMAX(TPL_FABS(bounds.lower.z), TPL_FABS(bounds.upper.z));

  // Broad-phase rounding must not discard contact accepted by the narrow phase.
  // Its squared distances can also underflow near zero; uncertain bounds fall
  // through.
  const Scalar radius = bounds.radius + Scalar(64) * epsilon * scale +
                        Scalar(8) * TPL_SQRT(minimum);

  return !(bounds.lower.x > position.x + half_extents.x + radius ||
           bounds.upper.x < position.x - half_extents.x - radius ||
           bounds.lower.y > position.y + half_extents.y + radius ||
           bounds.upper.y < position.y - half_extents.y - radius ||
           bounds.lower.z > position.z + half_extents.z + radius ||
           bounds.upper.z < position.z - half_extents.z - radius);
}

__device__ inline bool overlaps(const Pose& pose,
                                Vec3 half_extents,
                                const SphereBounds& bounds) {
  const Scalar* r = pose.world_to_local;
  const Vec3 world_half_extents = {
      TPL_FABS(r[0]) * half_extents.x + TPL_FABS(r[3]) * half_extents.y +
          TPL_FABS(r[6]) * half_extents.z,
      TPL_FABS(r[1]) * half_extents.x + TPL_FABS(r[4]) * half_extents.y +
          TPL_FABS(r[7]) * half_extents.z,
      TPL_FABS(r[2]) * half_extents.x + TPL_FABS(r[5]) * half_extents.y +
          TPL_FABS(r[8]) * half_extents.z};

  return overlaps(pose.position, world_half_extents, bounds);
}

__device__ inline bool overlaps(const Box& box, const SphereBounds& bounds) {
  return overlaps(box.pose,
                  {box.extents.x * Scalar(0.5), box.extents.y * Scalar(0.5),
                   box.extents.z * Scalar(0.5)},
                  bounds);
}

__device__ inline bool overlaps(const Sphere& sphere,
                                const SphereBounds& bounds) {
  return overlaps(sphere.position,
                  {sphere.radius, sphere.radius, sphere.radius}, bounds);
}

__device__ inline bool overlaps(const Cylinder& cylinder,
                                const SphereBounds& bounds) {
  return overlaps(
      cylinder.pose,
      {cylinder.radius, cylinder.radius, cylinder.height * Scalar(0.5)},
      bounds);
}

__device__ inline Vec3 local_point(const Pose& pose, Vec3 point) {
  const Scalar x = point.x - pose.position.x;
  const Scalar y = point.y - pose.position.y;
  const Scalar z = point.z - pose.position.z;

  const Scalar* r = pose.world_to_local;
  return {r[0] * x + r[1] * y + r[2] * z, r[3] * x + r[4] * y + r[5] * z,
          r[6] * x + r[7] * y + r[8] * z};
}

__device__ inline Vec3 box_face_distances(const Box& box, Vec3 point) {
  const Vec3 p = local_point(box.pose, point);
  return {TPL_FABS(p.x) - box.extents.x * Scalar(0.5),
          TPL_FABS(p.y) - box.extents.y * Scalar(0.5),
          TPL_FABS(p.z) - box.extents.z * Scalar(0.5)};
}

__device__ inline Scalar squared_outside_distance(Vec3 face_distances) {
  const Scalar x = TPL_FMAX(face_distances.x, Scalar(0));
  const Scalar y = TPL_FMAX(face_distances.y, Scalar(0));
  const Scalar z = TPL_FMAX(face_distances.z, Scalar(0));

  return x * x + y * y + z * z;
}

__device__ inline Scalar squared_distance(Vec3 a, Vec3 b) {
  const Scalar x = a.x - b.x;
  const Scalar y = a.y - b.y;
  const Scalar z = a.z - b.z;

  return x * x + y * y + z * z;
}

__device__ inline Scalar distance(const Box& box, Vec3 point) {
  const Vec3 d = box_face_distances(box, point);
  return TPL_SQRT(squared_outside_distance(d)) +
         TPL_FMIN(TPL_FMAX(d.x, TPL_FMAX(d.y, d.z)), Scalar(0));
}

__device__ inline Scalar distance(const Sphere& sphere, Vec3 point) {
  return TPL_SQRT(squared_distance(point, sphere.position)) - sphere.radius;
}

__device__ inline Scalar distance(const Cylinder& cylinder, Vec3 point) {
  const Vec3 p = local_point(cylinder.pose, point);
  const Scalar radial = TPL_SQRT(p.x * p.x + p.y * p.y) - cylinder.radius;
  const Scalar axial = TPL_FABS(p.z) - cylinder.height * Scalar(0.5);
  const Scalar r = TPL_FMAX(radial, Scalar(0));
  const Scalar z = TPL_FMAX(axial, Scalar(0));

  return TPL_SQRT(r * r + z * z) + TPL_FMIN(TPL_FMAX(radial, axial), Scalar(0));
}

__device__ inline Scalar distance(const Ground& ground, Vec3 point) {
  return point.z + ground.height;
}

__device__ inline bool intersects(const Box& box, Vec3 point, Scalar radius) {
  const Scalar squared_radius = radius * radius;
  // Finite radii can overflow when squared; infinity <= infinity would report a
  // hit.
  if (isinf(squared_radius)) {
    return distance(box, point) <= radius;
  }

  return squared_outside_distance(box_face_distances(box, point)) <=
         squared_radius;
}

__device__ inline bool intersects(const Sphere& sphere,
                                  Vec3 point,
                                  Scalar radius) {
  const Scalar sum = sphere.radius + radius;
  const Scalar squared_radius = sum * sum;
  if (isinf(squared_radius)) {
    return distance(sphere, point) <= radius;
  }

  return squared_distance(point, sphere.position) <= squared_radius;
}

__device__ inline bool intersects(const Cylinder& cylinder,
                                  Vec3 point,
                                  Scalar radius) {
  return distance(cylinder, point) <= radius;
}

__device__ inline bool intersects(const Ground& ground,
                                  Vec3 point,
                                  Scalar radius) {
  return distance(ground, point) <= radius;
}

__device__ inline Scalar distance(const Scene& scene,
                                  const Environment& env,
                                  Vec3 point) {
  Scalar result = INFINITY;

  // Keep reductions unconditional; short-circuit tests would skip remaining
  // obstacles.
  for (size_t i = 0; i < env.boxes.count; ++i) {
    result =
        TPL_FMIN(result, distance(scene.boxes[env.boxes.offset + i], point));
  }

  for (size_t i = 0; i < env.spheres.count; ++i) {
    result = TPL_FMIN(result,
                      distance(scene.spheres[env.spheres.offset + i], point));
  }

  for (size_t i = 0; i < env.cylinders.count; ++i) {
    result = TPL_FMIN(
        result, distance(scene.cylinders[env.cylinders.offset + i], point));
  }

  for (size_t i = 0; i < env.grounds.count; ++i) {
    result = TPL_FMIN(result,
                      distance(scene.grounds[env.grounds.offset + i], point));
  }

  return result;
}

}  // namespace TPL_NAMESPACE
#endif
