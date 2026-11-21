#ifndef tenkai_f32_RRT_RRTC_INCLUDED
#define tenkai_f32_RRT_RRTC_INCLUDED
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "common.cuh"
#include "generated_model.cuh"
#include "types.hpp"

namespace tenkai_f32 {
enum class RrtPreparation : unsigned char { Search, InvalidStart, InvalidGoal };

#ifdef __HIPCC__
constexpr int rrtc_min_blocks = 6;
#else
// A higher residency target spills the sphere batches on CUDA.
constexpr int rrtc_min_blocks = 3;
#endif

// Every interpolation point is required; reverse edges include their new
// source.
template <unsigned Types = AllTypes>
__device__ __forceinline__ void rrtc_check_edge(RrtCandidate& candidate,
                                                const Scalar* nodes,
                                                const RrtProblem& problem,
                                                const RrtSpace& space,
                                                Scene scene,
                                                bool reverse) {
  const uint32_t lane = threadIdx.x % 32;
  const uint64_t checks =
      uint64_t(candidate.steps) + (reverse && candidate.steps ? 1 : 0);
  const uint64_t waves = (checks + 31) / 32;

  for (uint64_t wave = 0; wave < waves; ++wave) {
    const uint64_t sample = uint64_t(lane) * waves + wave;
    bool hit = false;
    const unsigned mask = __ballot_sync(0xffffffffu, sample < checks);

    if (sample < checks) {
      const uint32_t step = uint32_t(sample) + (reverse ? 0 : 1);
      const Scalar ratio = Scalar(step) / Scalar(candidate.steps);
      const Scalar* parent = nodes + size_t(candidate.parent) * rrt_dimension;
      const Scalar* from = reverse ? candidate.q : parent;
      const Scalar* to = reverse ? parent : candidate.q;

      const uint32_t controls[rrt_dimension] = {TPL_CONTROLS};
      Scalar full[rrt_fk_dof];
      for (uint32_t d = 0; d < rrt_fk_dof; ++d) {
        full[d] = space.reference[d];
      }
      for (uint32_t d = 0; d < rrt_dimension; ++d) {
        Scalar q;
        if (step == candidate.steps) {
          q = to[d];
        } else if (step == 0) {
          q = from[d];
        } else {
          q = from[d] + ratio * (to[d] - from[d]);
        }
        full[controls[d]] = q;
      }

      hit = tenkai_collision_moving<true, Types>(full, scene,
                                                 problem.environment_id, mask);
    }

    if (__any_sync(0xffffffffu, hit)) {
      if (lane == 0) {
        candidate.blocked = 1;
      }
      break;
    }
  }
}

struct RrtcState {
  RrtCandidate candidate;
  tenkai::RrtResult result;
  bool connecting;
  volatile bool finished;
  uint32_t round;
  uint32_t target;
};

template <unsigned Types = AllTypes>
__device__ __forceinline__ void rrtc_search(const RrtProblem* problems,
                                            size_t id,
                                            const RrtSpace& space,
                                            Scene scene,
                                            tenkai::RrtConfig config,
                                            Scalar* all_nodes,
                                            uint32_t* all_parents,
                                            RrtTree* all_trees,
                                            uint32_t* all_paths,
                                            tenkai::RrtResult* results,
                                            RrtcState& state,
                                            const RrtPreparation* preparation) {
  using namespace tenkai_f32;

  const uint32_t lane = threadIdx.x % 32;
  const RrtProblem& problem = problems[id];
  Scalar* nodes = all_nodes + id * config.max_nodes * rrt_dimension;
  uint32_t* parents = all_parents + id * config.max_nodes;
  RrtTree* trees = all_trees + id * config.max_nodes;
  uint32_t* path = all_paths + id * config.max_nodes;

  RrtCandidate& candidate = state.candidate;
  tenkai::RrtResult& result = state.result;
  bool& connecting = state.connecting;
  volatile bool& finished = state.finished;
  uint32_t& round = state.round;
  uint32_t& target = state.target;

  if (lane == 0) {
    round = 0;
    connecting = false;
    result = {tenkai::PlanStatus::RoundLimit, 0, 0, 0};
    finished = false;

    rrt_append(problem.start, 0, nodes, parents, result);
    rrt_append(problem.goal, 1, nodes, parents, result);
    trees[0] = RrtTree::Start;
    trees[1] = RrtTree::Goal;

    const RrtPreparation initial = preparation[id];
    if (initial == RrtPreparation::InvalidStart ||
        initial == RrtPreparation::InvalidGoal) {
      result.status = initial == RrtPreparation::InvalidStart
                          ? tenkai::PlanStatus::InvalidStart
                          : tenkai::PlanStatus::InvalidGoal;
      result.node_count = 0;
      finished = true;
    } else {
      bool equal = true;
      for (uint32_t d = 0; d < rrt_dimension; ++d) {
        equal &= problem.start[d] == problem.goal[d];
      }

      if (equal) {
        result.status = tenkai::PlanStatus::Success;
        result.path_length = 1;
        path[0] = 0;
        finished = true;
      }
    }
  }
  __syncwarp();

  if (lane == 0 && !finished) {
    if (result.node_count == config.max_nodes) {
      result.status = tenkai::PlanStatus::NodeLimit;
      finished = true;
    }
  }
  __syncwarp();

  while (round < config.max_rounds && !finished) {
    const RrtTree active = round % 2 == 0 ? RrtTree::Start : RrtTree::Goal;
    RrtTree tree = active;
    if (connecting) {
      tree = active == RrtTree::Start ? RrtTree::Goal : RrtTree::Start;
    }

    if (connecting) {
      for (uint32_t d = lane; d < rrt_dimension; d += 32) {
        candidate.q[d] = nodes[size_t(target) * rrt_dimension + d];
      }
      if (lane == 0) {
        candidate.reaches_target = true;
      }
    } else {
      uint64_t base =
          problem.seed + uint64_t(round) * UINT64_C(0xd1342543de82ef95);
      const bool biased = rrt_uniform(base) < config.goal_bias;
      const Scalar* bias_target =
          active == RrtTree::Start ? problem.goal : problem.start;

      for (uint32_t d = lane; d < rrt_dimension; d += 32) {
        uint64_t rng = base + uint64_t(d) * UINT64_C(0x9e3779b97f4a7c15);
        if (biased) {
          candidate.q[d] = bias_target[d];
        } else {
          candidate.q[d] = Scalar(double(space.lower[d]) +
                                  rrt_uniform(rng) * (double(space.upper[d]) -
                                                      double(space.lower[d])));
        }
      }
      if (lane == 0) {
        candidate.reaches_target = biased;
      }
    }
    __syncwarp();

    const uint32_t unscanned =
        result.node_count - (connecting ? candidate.nearest.scanned : 0);
    const uint32_t width = unscanned < 32 ? 1 : 32;
    if (lane < width) {
      rrt_extend(candidate, nodes, result.node_count, space, config.step_size,
                 trees, tree, connecting ? &candidate.nearest : nullptr, width);
    }
    // Lane-zero predicates can still compile to shared loads in every lane.
    __syncwarp();

    if (lane == 0 && connecting && !candidate.steps &&
        candidate.reaches_target) {
      candidate.blocked = 0;
    }

    rrtc_check_edge<Types>(candidate, nodes, problem, space, scene,
                           tree == RrtTree::Goal);
    __syncwarp();

    if (lane == 0) {
      if (!connecting) {
        result.rounds = round + 1;
      }

      if (candidate.blocked) {
        connecting = false;
        ++round;
        if (result.node_count == config.max_nodes) {
          result.status = tenkai::PlanStatus::NodeLimit;
          finished = true;
        }
      } else if (!connecting) {
        target =
            rrt_append(candidate.q, candidate.parent, nodes, parents, result);
        trees[target] = active;
        candidate.nearest = {0, 0, INFINITY};
        connecting = true;
      } else {
        uint32_t node = candidate.parent;
        if (candidate.steps) {
          if (result.node_count == config.max_nodes) {
            result.status = tenkai::PlanStatus::NodeLimit;
            finished = true;
          } else {
            node = rrt_append(candidate.q, candidate.parent, nodes, parents,
                              result);
            trees[node] = tree;
          }
        }

        if (!finished && candidate.reaches_target) {
          const uint32_t start = active == RrtTree::Start ? target : node;
          const uint32_t goal = active == RrtTree::Goal ? target : node;
          rrt_start_path(start, parents, path, result);
          for (uint32_t n = goal; n != parents[n];) {
            n = parents[n];
            path[result.path_length++] = n;
          }
          finished = true;
        }

        if (!finished && result.node_count == config.max_nodes) {
          result.status = tenkai::PlanStatus::NodeLimit;
          finished = true;
        }
      }
    }
    __syncwarp();
  }

  if (lane == 0) {
    results[id] = result;
  }
}

}  // namespace tenkai_f32

extern "C" __global__ void tenkai_rrtc_prepare(
    const tenkai_f32::RrtProblem* problems,
    size_t problem_count,
    tenkai_f32::RrtSpace space,
    tenkai_f32::Scene scene,
    tenkai_f32::RrtPreparation* preparation,
    uint32_t* next_problem) {
  using namespace tenkai_f32;

  const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id == 0) {
    *next_problem = 0;
  }
  if (id >= problem_count) {
    return;
  }

  const RrtProblem& problem = problems[id];
  RrtPreparation initial = RrtPreparation::Search;
  if (!rrt_in_bounds(problem.start, space) ||
      rrt_collision(problem.start, space, scene, problem.environment_id)) {
    initial = RrtPreparation::InvalidStart;
  } else if (!rrt_in_bounds(problem.goal, space) ||
             rrt_collision(problem.goal, space, scene, problem.environment_id,
                           true)) {
    initial = RrtPreparation::InvalidGoal;
  }

  preparation[id] = initial;
}

template <unsigned Types = tenkai_f32::AllTypes>
__global__ __launch_bounds__(128, tenkai_f32::rrtc_min_blocks) void tenkai_rrtc(
    const tenkai_f32::RrtProblem* problems,
    size_t problem_count,
    tenkai_f32::RrtSpace space,
    tenkai_f32::Scene scene,
    tenkai::RrtConfig config,
    tenkai_f32::Scalar* all_nodes,
    uint32_t* all_parents,
    tenkai_f32::RrtTree* all_trees,
    uint32_t* all_paths,
    tenkai::RrtResult* results,
    const tenkai_f32::RrtPreparation* preparation,
    uint32_t* next_problem) {
  __shared__ tenkai_f32::RrtcState states[4];
  auto& state = states[threadIdx.x / 32];

  // A finished plan must not strand its warp behind another plan in the block.
  for (;;) {
    uint32_t id = 0;
    if (threadIdx.x % 32 == 0) {
      id = atomicAdd(next_problem, 1);
    }
    id = __shfl_sync(0xffffffffu, id, 0);
    if (id >= problem_count) {
      return;
    }

    tenkai_f32::rrtc_search<Types>(problems, id, space, scene, config,
                                   all_nodes, all_parents, all_trees, all_paths,
                                   results, state, preparation);
    __syncwarp();
  }
}

#endif
