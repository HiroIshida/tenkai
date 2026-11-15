#ifndef tenkai_f32_BROAD_INCLUDED
#define tenkai_f32_BROAD_INCLUDED
#include <float.h>
#include <math.h>

#include "sdf.cuh"

namespace tenkai_f32 {
__device__ inline Scalar coordinate_scale(Vec3 point) {
  return fabsf(point.x) + fabsf(point.y) + fabsf(point.z);
}

__device__ inline Sphere link_sphere(Vec3 center,
                                     Vec3 translation,
                                     Scalar qx,
                                     Scalar qy,
                                     Scalar qz,
                                     Scalar qw,
                                     Scalar radius,
                                     Scalar local_scale) {
  const Scalar norm_squared = qx * qx + qy * qy + qz * qz + qw * qw;
  // SFU/composition rounding need not leave a unit quaternion. The linear map
  // p + 2*q.xyz cross (qw*p + q.xyz cross p) has norm <= max(1, 2*|q|^2-1).
  const Scalar stretch = fmaxf(1, 2 * norm_squared - 1);
  const Scalar scale =
      coordinate_scale(translation) + local_scale * (1 + norm_squared);
  // Cover both center transforms, rounded local constants and matrix expansion.
  const Scalar margin = 256 * FLT_EPSILON * scale + 8 * sqrtf(FLT_MIN);
  return {center, radius * stretch + margin};
}

__device__ inline Scalar obstacle_scale(const Box& box) {
  return coordinate_scale(box.pose.position) + coordinate_scale(box.extents);
}

__device__ inline Scalar obstacle_scale(const Sphere& sphere) {
  return coordinate_scale(sphere.position) + sphere.radius;
}

__device__ inline Scalar obstacle_scale(const Cylinder& cylinder) {
  return coordinate_scale(cylinder.pose.position) + cylinder.radius +
         cylinder.height;
}

__device__ inline Scalar obstacle_scale(const Ground& ground) {
  return fabsf(ground.height);
}

template <typename Obstacle>
__device__ inline Scalar enclosing_radius(const Obstacle& obstacle,
                                          const Sphere& enclosing) {
  const Scalar scale = coordinate_scale(enclosing.position) + enclosing.radius +
                       obstacle_scale(obstacle);
  // Overflow, NaN and squared-distance underflow must not certify separation.
  if (!(scale < sqrtf(FLT_MAX) / 16)) {
    return INFINITY;
  }
  return enclosing.radius + 256 * FLT_EPSILON * scale + 8 * sqrtf(FLT_MIN);
}

template <typename Obstacle>
__device__ inline bool may_intersect(const Obstacle& obstacle,
                                     const Sphere& enclosing) {
  const Scalar radius = enclosing_radius(obstacle, enclosing);
  return !isfinite(radius) || intersects(obstacle, enclosing.position, radius);
}

template <unsigned Types = AllTypes>
__device__ inline bool may_intersect(const Scene& scene,
                                     const Environment& env,
                                     const Sphere& enclosing) {
  for (size_t i = 0; i < (Types & BoxType ? env.boxes.count : 0); ++i) {
    if (may_intersect(scene.boxes[env.boxes.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & SphereType ? env.spheres.count : 0); ++i) {
    if (may_intersect(scene.spheres[env.spheres.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & CylinderType ? env.cylinders.count : 0);
       ++i) {
    if (may_intersect(scene.cylinders[env.cylinders.offset + i], enclosing))
      return true;
  }
  for (size_t i = 0; i < (Types & GroundType ? env.grounds.count : 0); ++i) {
    if (may_intersect(scene.grounds[env.grounds.offset + i], enclosing))
      return true;
  }
  return false;
}

}  // namespace tenkai_f32
#endif
