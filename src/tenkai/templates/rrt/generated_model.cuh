#ifndef tenkai_f32_RRT_MODEL_INCLUDED
#define tenkai_f32_RRT_MODEL_INCLUDED
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "../collision/broad.cuh"

namespace tenkai_f32 {
constexpr uint32_t rrt_dimension = TPL_DIMENSION;
constexpr uint32_t rrt_fk_dof = TPL_FK_DOF;
}  // namespace tenkai_f32

/* TPL_COLLISION_FUNCTIONS */

/* TPL_COLLISION_KERNEL */
#endif
