#ifndef tenkai_f32_RRT_COMMON_INCLUDED
#define tenkai_f32_RRT_COMMON_INCLUDED
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "generated_model.cuh"
#include "types.hpp"

namespace tenkai_f32 {
struct RrtSpace {
  Scalar lower[rrt_dimension];
  Scalar upper[rrt_dimension];
  Scalar resolution[rrt_dimension];
  Scalar reference[rrt_fk_dof];
};

struct RrtProblem {
  uint32_t environment_id;
  uint64_t seed;
  Scalar start[rrt_dimension];
  Scalar goal[rrt_dimension];
};

enum class RrtTree : uint8_t { Start, Goal };

struct RrtNearestCache {
  uint32_t node;
  uint32_t scanned;
  double squared_distance;
};

struct RrtCandidate {
  Scalar q[rrt_dimension];
  uint32_t parent;
  uint32_t steps;
  unsigned int blocked;
  bool reaches_target;
  RrtNearestCache nearest;
};

inline void validate_rrt(const RrtSpace& space) {
  double diameter_squared = 0;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    const double span = double(space.upper[d]) - double(space.lower[d]);
    diameter_squared += span * span;

    if (!std::isfinite(space.lower[d]) || !std::isfinite(space.upper[d]) ||
        !(span > 0) || !std::isfinite(span) ||
        span > double(std::numeric_limits<Scalar>::max()) ||
        !std::isfinite(space.resolution[d]) || space.resolution[d] <= 0 ||
        std::ceil(span / double(space.resolution[d])) >
            double(UINT32_MAX - 1)) {
      throw std::invalid_argument("invalid RRT bounds or resolution");
    }
  }
  if (!std::isfinite(diameter_squared)) {
    throw std::invalid_argument("RRT distance overflows double precision");
  }

  for (uint32_t d = 0; d < rrt_fk_dof; ++d) {
    if (!std::isfinite(space.reference[d])) {
      throw std::invalid_argument("nonfinite RRT reference configuration");
    }
  }
}

__device__ inline uint64_t rrt_random(uint64_t& state) {
  uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

__device__ inline double rrt_uniform(uint64_t& state) {
  return double(rrt_random(state) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline bool rrt_in_bounds(const Scalar* q, const RrtSpace& space) {
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    if (!(q[d] >= space.lower[d] && q[d] <= space.upper[d])) {
      return false;
    }
  }

  return true;
}

__device__ inline bool rrt_collision(const Scalar* q,
                                     const RrtSpace& space,
                                     Scene scene,
                                     uint32_t environment_id,
                                     bool moving_only = false) {
  const uint32_t controls[rrt_dimension] = {TPL_CONTROLS};
  Scalar full[rrt_fk_dof];
  for (uint32_t d = 0; d < rrt_fk_dof; ++d) {
    full[d] = space.reference[d];
  }
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    full[controls[d]] = q[d];
  }

  return moving_only ? tenkai_collision_moving(full, scene, environment_id)
                     : tenkai_collision_one(full, scene, environment_id);
}

__device__ inline uint32_t rrt_steps(const Scalar* a,
                                     const Scalar* b,
                                     const RrtSpace& space) {
  const Scalar resolution = space.resolution[0];
  bool uniform = true;
  Scalar displacement = 0;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    uniform &= space.resolution[d] == resolution;
    displacement = fmaxf(displacement, fabsf(b[d] - a[d]));
  }
  if (uniform) {
    return static_cast<uint32_t>(fmaxf(1.0f, ceilf(displacement / resolution)));
  }

  Scalar count = 1;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    count = fmaxf(count, ceilf(fabsf(b[d] - a[d]) / space.resolution[d]));
  }
  return static_cast<uint32_t>(count);
}

__device__ inline double rrt_squared_distance(const Scalar* a,
                                              const Scalar* b) {
  Scalar distance = 0;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    const Scalar delta = a[d] - b[d];
    distance += delta * delta;
  }
  return distance;
}

__device__ inline unsigned int rrt_group_mask(uint32_t width) {
  return (0xffffffffu >> (32 - width)) << ((threadIdx.x % 32) & ~(width - 1));
}

// Width is a power of two in [1, 32]; every lane in the aligned group
// participates. Each lane owns its cache value; only lane zero receives the
// reduced answer.
__device__ inline void rrt_nearest(const Scalar* q,
                                   const Scalar* nodes,
                                   uint32_t snapshot_size,
                                   const RrtTree* trees,
                                   RrtTree tree,
                                   RrtNearestCache& nearest,
                                   uint32_t width = 1) {
  const uint32_t lane = threadIdx.x & (width - 1);
  double best_distance = nearest.squared_distance;
  uint32_t best_node = nearest.node;

  // With an unchanged target and tree, only appended nodes can improve the
  // answer.
  for (uint64_t n = uint64_t(nearest.scanned) + lane; n < snapshot_size;
       n += width) {
    if (trees && trees[n] != tree) {
      continue;
    }

    const double distance =
        rrt_squared_distance(nodes + size_t(n) * rrt_dimension, q);
    if (distance < best_distance) {
      best_distance = distance;
      best_node = uint32_t(n);
    }
  }

  if (width > 1) {
    const unsigned int mask = rrt_group_mask(width);
    for (uint32_t offset = width / 2; offset; offset /= 2) {
      const double distance =
          __shfl_down_sync(mask, best_distance, offset, width);
      const uint32_t node = __shfl_down_sync(mask, best_node, offset, width);

      // Equal distances retain the earliest node, matching the serial scan.
      if (lane + offset < width &&
          (distance < best_distance ||
           (distance == best_distance && node < best_node))) {
        best_distance = distance;
        best_node = node;
      }
    }
  }

  nearest = {best_node, snapshot_size, best_distance};
}

__device__ inline void rrt_steer(RrtCandidate& candidate,
                                 const Scalar* parent,
                                 double best_distance,
                                 const RrtSpace& space,
                                 double step_size) {
  if (step_size == 1.0) {
    candidate.reaches_target &= best_distance <= 1.0;
    if (best_distance > 1.0) {
      const Scalar ratio = __frsqrt_rn(Scalar(best_distance));
      for (uint32_t d = 0; d < rrt_dimension; ++d) {
        candidate.q[d] = parent[d] + ratio * (candidate.q[d] - parent[d]);
      }
    }
  } else {
    const Scalar distance = sqrtf(Scalar(best_distance));
    candidate.reaches_target &= distance <= Scalar(step_size);
    if (distance > Scalar(step_size)) {
      const Scalar ratio = Scalar(step_size) / distance;
      for (uint32_t d = 0; d < rrt_dimension; ++d) {
        candidate.q[d] = parent[d] + ratio * (candidate.q[d] - parent[d]);
      }
    }
  }

  bool moved = false;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    moved |= candidate.q[d] != parent[d];
  }
  candidate.blocked = !moved;
  candidate.steps = moved ? rrt_steps(parent, candidate.q, space) : 0;
}

// Parallel callers share candidate/cache and initialize them before the group
// enters.
__device__ inline void rrt_extend(RrtCandidate& candidate,
                                  const Scalar* nodes,
                                  uint32_t snapshot_size,
                                  const RrtSpace& space,
                                  double step_size,
                                  const RrtTree* trees = nullptr,
                                  RrtTree tree = RrtTree::Start,
                                  RrtNearestCache* cache = nullptr,
                                  uint32_t width = 1) {
  RrtNearestCache nearest = cache ? *cache : RrtNearestCache{0, 0, INFINITY};
  rrt_nearest(candidate.q, nodes, snapshot_size, trees, tree, nearest, width);

  // Shuffles do not fence the other lanes' reads of the query and shared cache.
  if (width > 1) {
    __syncwarp(rrt_group_mask(width));
  }

  if ((threadIdx.x & (width - 1)) == 0) {
    if (cache) {
      *cache = nearest;
    }
    candidate.parent = nearest.node;
    rrt_steer(candidate, nodes + size_t(nearest.node) * rrt_dimension,
              nearest.squared_distance, space, step_size);
  }
  if (width > 1) {
    __syncwarp(rrt_group_mask(width));
  }
}

__device__ inline uint32_t rrt_append(const Scalar* q,
                                      uint32_t parent,
                                      Scalar* nodes,
                                      uint32_t* parents,
                                      tenkai::RrtResult& result) {
  const uint32_t node = result.node_count++;
  for (uint32_t d = 0; d < rrt_dimension; ++d) {
    nodes[size_t(node) * rrt_dimension + d] = q[d];
  }
  parents[node] = parent;

  return node;
}

__device__ inline void rrt_start_path(uint32_t node,
                                      const uint32_t* parents,
                                      uint32_t* path,
                                      tenkai::RrtResult& result) {
  result.status = tenkai::PlanStatus::Success;
  for (uint32_t n = node;; n = parents[n]) {
    path[result.path_length++] = n;
    if (n == parents[n]) {
      break;
    }
  }

  for (uint32_t i = 0; i < result.path_length / 2; ++i) {
    const uint32_t other = result.path_length - 1 - i;
    const uint32_t value = path[i];
    path[i] = path[other];
    path[other] = value;
  }
}

}  // namespace tenkai_f32
#endif
