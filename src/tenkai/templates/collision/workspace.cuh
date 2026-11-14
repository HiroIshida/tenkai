#ifndef TPL_NAMESPACE_WORKSPACE_INCLUDED
#define TPL_NAMESPACE_WORKSPACE_INCLUDED
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stddef.h>

#include "sdf.cuh"

#if TPL_FP32
namespace tenkai_f32 {
struct WorkspaceInterval {
  float lower;
  float upper;
};

__device__ inline WorkspaceInterval workspace_unknown() {
  return {-INFINITY, INFINITY};
}

__device__ inline bool workspace_finite(WorkspaceInterval a) {
  return isfinite(a.lower) && isfinite(a.upper) && a.lower <= a.upper;
}

__device__ inline WorkspaceInterval workspace_input(float lower, float upper) {
  // Endpoint checks certify the joint interval once per edge. Enclose rounding
  // of from + (step / steps) * (to - from), including subnormal intermediates.
  const float scale = __fadd_ru(fabsf(lower), fabsf(upper));
  const float margin =
      __fadd_ru(__fmul_ru(8 * FLT_EPSILON, scale), 8 * FLT_MIN);

  return {__fsub_rd(lower, margin), __fadd_ru(upper, margin)};
}

__device__ inline WorkspaceInterval workspace_add(WorkspaceInterval a,
                                                  WorkspaceInterval b) {
  if (!workspace_finite(a) || !workspace_finite(b)) {
    return workspace_unknown();
  }

  return {__fadd_rd(a.lower, b.lower), __fadd_ru(a.upper, b.upper)};
}

__device__ inline WorkspaceInterval workspace_sub(WorkspaceInterval a,
                                                  WorkspaceInterval b) {
  if (!workspace_finite(a) || !workspace_finite(b)) {
    return workspace_unknown();
  }

  return {__fsub_rd(a.lower, b.upper), __fsub_ru(a.upper, b.lower)};
}

__device__ inline WorkspaceInterval workspace_mul(WorkspaceInterval a,
                                                  WorkspaceInterval b) {
  if (!workspace_finite(a) || !workspace_finite(b)) {
    return workspace_unknown();
  }

  return {
      fminf(fminf(__fmul_rd(a.lower, b.lower), __fmul_rd(a.lower, b.upper)),
            fminf(__fmul_rd(a.upper, b.lower), __fmul_rd(a.upper, b.upper))),
      fmaxf(fmaxf(__fmul_ru(a.lower, b.lower), __fmul_ru(a.lower, b.upper)),
            fmaxf(__fmul_ru(a.upper, b.lower), __fmul_ru(a.upper, b.upper)))};
}

__device__ inline WorkspaceInterval workspace_trig(WorkspaceInterval a,
                                                   bool cosine) {
  if (!workspace_finite(a)) {
    return workspace_unknown();
  }
  if (a.lower == 0 && a.upper == 0) {
    return cosine ? WorkspaceInterval{1, 1} : WorkspaceInterval{0, 0};
  }

  // Standard CUDA sinf/cosf have a 2-ULP error bound. Do not use this for SFU
  // mode. Keeping the whole range avoids trusting rounded extrema/argument
  // reduction.
  return {-(1 + 4 * FLT_EPSILON), 1 + 4 * FLT_EPSILON};
}

__device__ inline bool workspace_clear(WorkspaceInterval x,
                                       WorkspaceInterval y,
                                       WorkspaceInterval z,
                                       float radius,
                                       Scene scene,
                                       const Environment& environment) {
  if (!workspace_finite(x) || !workspace_finite(y) || !workspace_finite(z)) {
    return false;
  }

  const SphereBounds bounds = {
      {x.lower, y.lower, z.lower}, {x.upper, y.upper, z.upper}, radius};

  for (size_t i = 0; i < environment.boxes.count; ++i) {
    if (overlaps(scene.boxes[environment.boxes.offset + i], bounds)) {
      return false;
    }
  }

  for (size_t i = 0; i < environment.spheres.count; ++i) {
    if (overlaps(scene.spheres[environment.spheres.offset + i], bounds)) {
      return false;
    }
  }

  for (size_t i = 0; i < environment.cylinders.count; ++i) {
    if (overlaps(scene.cylinders[environment.cylinders.offset + i], bounds)) {
      return false;
    }
  }

  for (size_t i = 0; i < environment.grounds.count; ++i) {
    const float ground_distance_lower = __fadd_rd(
        z.lower, scene.grounds[environment.grounds.offset + i].height);
    if (!(ground_distance_lower > radius)) {
      return false;
    }
  }

  return true;
}

}  // namespace tenkai_f32
#endif
#endif
