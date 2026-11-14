#ifndef tenkai_f32_STREAMING_INCLUDED
#define tenkai_f32_STREAMING_INCLUDED
#include <cuda_runtime.h>
#include <math.h>

#include "broad.cuh"

namespace tenkai_f32 {
enum class StreamingCollision { Clear, Blocked, Retry };

__device__ inline bool streaming_intersects(const Box& box,
                                            Vec3 point,
                                            float radius,
                                            bool& retry) {
  const Vec3 p = {point.x - box.pose.position.x, point.y - box.pose.position.y,
                  point.z - box.pose.position.z};
  // Identity matrix products involving infinity/NaN need the general predicate.
  retry |= !isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z);

  const Vec3 d = {fabsf(p.x) - box.extents.x * 0.5f,
                  fabsf(p.y) - box.extents.y * 0.5f,
                  fabsf(p.z) - box.extents.z * 0.5f};
  return squared_outside_distance(d) <= radius * radius;
}

__device__ inline bool streaming_may_intersect(const Box& box,
                                               const Sphere& enclosing) {
  const Scalar radius = enclosing_radius(box, enclosing);
  bool retry = false;
  return !isfinite(radius) ||
         streaming_intersects(box, enclosing.position, radius, retry) || retry;
}

}  // namespace tenkai_f32
#endif
