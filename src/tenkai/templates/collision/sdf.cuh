#ifndef tenkai_f32_SDF_INCLUDED
#define tenkai_f32_SDF_INCLUDED
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_runtime.h>
#endif
#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace tenkai_f32 {
using Scalar = float;
constexpr bool has_ground = TPL_HAS_GROUND;

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

enum GeometryType : unsigned {
  BoxType = 1,
  SphereType = 2,
  CylinderType = 4,
  AllTypes = BoxType | SphereType | CylinderType
};

struct Environment {
  Range boxes;
  Range spheres;
  Range cylinders;
};

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
  return {r[0] * x + r[1] * y + r[2] * z, r[3] * x + r[4] * y + r[5] * z,
          r[6] * x + r[7] * y + r[8] * z};
}

__device__ inline Vec3 box_face_distances(const Box& box, Vec3 point) {
  const Vec3 p = local_point(box.pose, point);
  return {fabsf(p.x) - box.extents.x * Scalar(0.5),
          fabsf(p.y) - box.extents.y * Scalar(0.5),
          fabsf(p.z) - box.extents.z * Scalar(0.5)};
}

__device__ inline Scalar squared_outside_distance(Vec3 face_distances) {
  const Scalar x = fmaxf(face_distances.x, Scalar(0));
  const Scalar y = fmaxf(face_distances.y, Scalar(0));
  const Scalar z = fmaxf(face_distances.z, Scalar(0));

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
  return sqrtf(squared_outside_distance(d)) +
         fminf(fmaxf(d.x, fmaxf(d.y, d.z)), Scalar(0));
}

__device__ inline Scalar distance(const Sphere& sphere, Vec3 point) {
  return sqrtf(squared_distance(point, sphere.position)) - sphere.radius;
}

__device__ inline Scalar distance(const Cylinder& cylinder, Vec3 point) {
  const Vec3 p = local_point(cylinder.pose, point);
  const Scalar radial = sqrtf(p.x * p.x + p.y * p.y) - cylinder.radius;
  const Scalar axial = fabsf(p.z) - cylinder.height * Scalar(0.5);
  const Scalar r = fmaxf(radial, Scalar(0));
  const Scalar z = fmaxf(axial, Scalar(0));

  return sqrtf(r * r + z * z) + fminf(fmaxf(radial, axial), Scalar(0));
}

__device__ inline Scalar distance(const Ground& ground, Vec3 point) {
  return point.z + ground.height;
}

__device__ inline bool intersects(const Box& box, Vec3 point, Scalar radius) {
  const Scalar squared_radius = radius * radius;
  return squared_outside_distance(box_face_distances(box, point)) <=
         squared_radius;
}

__device__ inline bool intersects(const Sphere& sphere,
                                  Vec3 point,
                                  Scalar radius) {
  const Scalar sum = sphere.radius + radius;
  const Scalar squared_radius = sum * sum;
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
                                  size_t environment_id,
                                  Vec3 point) {
  const Environment& env = scene.environments[environment_id];
  Scalar result = INFINITY;

  // Keep reductions unconditional; short-circuit tests would skip remaining
  // obstacles.
  for (size_t i = 0; i < env.boxes.count; ++i) {
    result = fminf(result, distance(scene.boxes[env.boxes.offset + i], point));
  }

  for (size_t i = 0; i < env.spheres.count; ++i) {
    result =
        fminf(result, distance(scene.spheres[env.spheres.offset + i], point));
  }

  for (size_t i = 0; i < env.cylinders.count; ++i) {
    result = fminf(result,
                   distance(scene.cylinders[env.cylinders.offset + i], point));
  }

  if (has_ground) {
    result = fminf(result, distance(scene.grounds[environment_id], point));
  }

  return result;
}

}  // namespace tenkai_f32
#endif
