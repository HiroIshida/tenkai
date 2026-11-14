#ifndef TPL_NAMESPACE_RRT_MODEL_INCLUDED
#define TPL_NAMESPACE_RRT_MODEL_INCLUDED
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "../collision/sdf.cuh"
#if TPL_FP32
#include "../collision/streaming.cuh"
#include "../collision/workspace.cuh"
#endif

namespace TPL_NAMESPACE {
constexpr uint32_t rrt_dimension = TPL_DIMENSION;
constexpr uint32_t rrt_fk_dof = TPL_FK_DOF;
}  // namespace TPL_NAMESPACE

/* TPL_COLLISION_FUNCTIONS */

#if TPL_FP32
/* TPL_EDGE_FUNCTION */

__device__ inline bool tenkai_collision_streamable(tenkai_f32::Scene scene,
                                                   uint32_t environment_id) {
  /* TPL_STREAMABLE_BODY */
}

__device__ __forceinline__ tenkai_f32::StreamingCollision
tenkai_collision_streaming(const float* q,
                           tenkai_f32::Scene scene,
                           uint32_t environment_id,
                           unsigned mask) {
  using namespace tenkai_f32;

  const Environment environment = scene.environments[environment_id];
  const Box& box = scene.boxes[environment.boxes.offset];
  const Ground& ground = scene.grounds[environment.grounds.offset];

  bool collision = false;
  bool retry = false;
  Vec3 center;
  /* TPL_STREAMING_BODY */

  StreamingCollision result;
  if (retry) {
    result = StreamingCollision::Retry;
  } else if (collision) {
    result = StreamingCollision::Blocked;
  } else {
    result = StreamingCollision::Clear;
  }
  return result;
}

__device__ inline bool tenkai_collision_in_workspace(const float* q,
                                                     const float* lower,
                                                     const float* upper) {
  /* TPL_BOUNDS_CHECKS */

  return true;
}

__device__ __noinline__ bool tenkai_collision_workspace_clear(
    const float* lower,
    const float* upper,
    const float* reference,
    tenkai_f32::Scene scene,
    uint32_t environment_id) {
  using namespace tenkai_f32;

  const Environment environment = scene.environments[environment_id];

  /* TPL_WORKSPACE_BODY */

  return true;
}
#endif

/* TPL_COLLISION_KERNEL */
#endif
