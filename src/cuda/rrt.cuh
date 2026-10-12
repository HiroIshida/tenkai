#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace @NAMESPACE@ {
constexpr uint32_t rrt_dimension = @DIMENSION@;
constexpr uint32_t rrt_fk_dof = @FK_DOF@;

struct RrtSpace {
    Scalar lower[rrt_dimension], upper[rrt_dimension], resolution[rrt_dimension];
    Scalar reference[rrt_fk_dof];
};

struct RrtProblem {
    uint32_t environment_id;
    uint64_t seed;
    Scalar start[rrt_dimension], goal[rrt_dimension];
};

enum class RrtTree : uint8_t { Start, Goal };
enum class GrowState : uint8_t { Trapped, Advanced, Reached };
enum class EdgeDirection { Forward, Reverse, Goal };

struct RrtNearestCache {
    uint32_t node, scanned;
    double squared_distance;
};

struct RrtCandidate {
    Scalar q[rrt_dimension];
    uint32_t parent, steps, target;
    unsigned int blocked, goal_blocked;
    bool reaches_target;
    GrowState state;
    RrtNearestCache nearest;
};

inline void validate_rrt(const rkern::RrtConfig& config, const RrtSpace& space) {
    if (config.max_nodes < 2 || !config.max_rounds || !config.samples_per_round ||
        config.samples_per_round > config.max_nodes - 1 ||
        !std::isfinite(config.step_size) || config.step_size <= 0 ||
        !std::isfinite(config.goal_bias) || config.goal_bias < 0 || config.goal_bias > 1)
        throw std::invalid_argument("invalid RRT config");
    double diameter_squared = 0;
    for (uint32_t d = 0; d < rrt_dimension; ++d) {
        const double span = double(space.upper[d]) - double(space.lower[d]);
        diameter_squared += span * span;
        if (!std::isfinite(space.lower[d]) || !std::isfinite(space.upper[d]) ||
            !(span > 0) || !std::isfinite(span) ||
            span > double(std::numeric_limits<Scalar>::max()) ||
            !std::isfinite(space.resolution[d]) || space.resolution[d] <= 0 ||
            std::ceil(span / double(space.resolution[d])) > double(UINT32_MAX - 1))
            throw std::invalid_argument("invalid RRT bounds or resolution");
    }
    if (!std::isfinite(diameter_squared))
        throw std::invalid_argument("RRT distance overflows double precision");
    for (uint32_t d = 0; d < rrt_fk_dof; ++d)
        if (!std::isfinite(space.reference[d]))
            throw std::invalid_argument("nonfinite RRT reference configuration");
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
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        if (!isfinite(q[d]) || q[d] < space.lower[d] || q[d] > space.upper[d])
            return false;
    return true;
}

__device__ inline bool rrt_collision(const Scalar* q, const RrtSpace& space,
                                     Scene scene, uint32_t environment_id) {
    const uint32_t controls[rrt_dimension] = {@CONTROLS@};
    Scalar full[rrt_fk_dof];
    for (uint32_t d = 0; d < rrt_fk_dof; ++d) full[d] = space.reference[d];
    for (uint32_t d = 0; d < rrt_dimension; ++d) full[controls[d]] = q[d];
    return rkern_collision_one(full, scene, environment_id);
}

__device__ inline uint32_t rrt_steps(const Scalar* a, const Scalar* b,
                                     const RrtSpace& space) {
    const Scalar resolution = space.resolution[0];
    bool uniform = true;
    double displacement = 0;
    for (uint32_t d = 0; d < rrt_dimension; ++d) {
        uniform &= space.resolution[d] == resolution;
        displacement = fmax(displacement, fabs(double(b[d]) - double(a[d])));
    }
    if (uniform)
        return static_cast<uint32_t>(fmax(1.0, ceil(displacement / double(resolution))));
    double count = 1;
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        count = fmax(count, ceil(fabs(double(b[d]) - double(a[d])) / double(space.resolution[d])));
    return static_cast<uint32_t>(count);
}

__device__ inline void rrt_sample(RrtCandidate& candidate, const RrtProblem& problem,
                                  const RrtSpace& space, rkern::RrtConfig config,
                                  uint32_t round, uint32_t c, const Scalar* bias_target) {
    // Each candidate owns a stream independent of warp count and execution order.
    uint64_t rng = problem.seed +
        (uint64_t(round) * config.samples_per_round + c) * UINT64_C(0xd1342543de82ef95);
    const bool biased = rrt_uniform(rng) < config.goal_bias;
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        candidate.q[d] = biased ? bias_target[d] : Scalar(
            double(space.lower[d]) + rrt_uniform(rng) * (double(space.upper[d]) - double(space.lower[d])));
    candidate.reaches_target = biased;
}

__device__ inline double rrt_squared_distance(const Scalar* a, const Scalar* b) {
    Scalar distance = 0;
    for (uint32_t d = 0; d < rrt_dimension; ++d) {
        const Scalar delta = a[d] - b[d];
        distance += delta * delta;
    }
    if (sizeof(Scalar) == sizeof(double) ||
        (isfinite(distance) && distance >= Scalar(FLT_MIN)))
        return distance;
    // Valid FP32 bounds can still overflow or underflow a squared distance.
    double wide_distance = 0;
    for (uint32_t d = 0; d < rrt_dimension; ++d) {
        const double delta = double(a[d]) - double(b[d]);
        wide_distance += delta * delta;
    }
    return wide_distance;
}

__device__ inline unsigned int rrt_group_mask(uint32_t width) {
    return (0xffffffffu >> (32 - width)) << ((threadIdx.x % 32) & ~(width - 1));
}

// Width is a power of two in [1, 32]; every lane in the aligned group participates.
// Each lane owns its cache value; only lane zero receives the reduced answer.
__device__ inline void rrt_nearest(const Scalar* q, const Scalar* nodes, uint32_t snapshot_size,
                                   const RrtTree* trees, RrtTree tree, RrtNearestCache& nearest,
                                   uint32_t width = 1) {
    const uint32_t lane = threadIdx.x & (width - 1);
    double best_distance = nearest.squared_distance;
    uint32_t best_node = nearest.node;
    // With an unchanged target and tree, only appended nodes can improve the answer.
    for (uint64_t n = uint64_t(nearest.scanned) + lane; n < snapshot_size; n += width) {
        if (trees && trees[n] != tree) continue;
        const double distance = rrt_squared_distance(nodes + size_t(n) * rrt_dimension, q);
        if (distance < best_distance) { best_distance = distance; best_node = uint32_t(n); }
    }
    if (width > 1) {
        const unsigned int mask = rrt_group_mask(width);
        for (uint32_t offset = width / 2; offset; offset /= 2) {
            const double distance = __shfl_down_sync(mask, best_distance, offset, width);
            const uint32_t node = __shfl_down_sync(mask, best_node, offset, width);
            // Equal distances retain the earliest node, matching the serial scan.
            if (lane + offset < width &&
                (distance < best_distance || (distance == best_distance && node < best_node))) {
                best_distance = distance;
                best_node = node;
            }
        }
    }
    nearest = {best_node, snapshot_size, best_distance};
}

__device__ inline void rrt_steer(RrtCandidate& candidate, const Scalar* parent,
                                 double best_distance, const RrtSpace& space, double step_size) {
    const double distance = sqrt(best_distance);
    candidate.reaches_target &= distance <= step_size;
    if (distance > step_size) {
        const double wide_ratio = step_size / distance;
        const Scalar ratio = Scalar(wide_ratio);
        for (uint32_t d = 0; d < rrt_dimension; ++d)
            candidate.q[d] = ratio > 0 ? parent[d] + ratio * (candidate.q[d] - parent[d]) :
                Scalar(double(parent[d]) + wide_ratio * (double(candidate.q[d]) - double(parent[d])));
    }
    bool moved = false;
    for (uint32_t d = 0; d < rrt_dimension; ++d) moved |= candidate.q[d] != parent[d];
    candidate.blocked = !moved;
    candidate.steps = moved ? rrt_steps(parent, candidate.q, space) : 0;
}

// Parallel callers share candidate/cache and initialize them before the group enters.
__device__ inline void rrt_extend(RrtCandidate& candidate, const Scalar* nodes,
                                  uint32_t snapshot_size, const RrtSpace& space, double step_size,
                                  const RrtTree* trees = nullptr, RrtTree tree = RrtTree::Start,
                                  RrtNearestCache* cache = nullptr, uint32_t width = 1) {
    RrtNearestCache nearest = cache ? *cache : RrtNearestCache{0, 0, INFINITY};
    rrt_nearest(candidate.q, nodes, snapshot_size, trees, tree, nearest, width);
    // Shuffles do not fence the other lanes' reads of the query and shared cache.
    if (width > 1) __syncwarp(rrt_group_mask(width));
    if ((threadIdx.x & (width - 1)) == 0) {
        if (cache) *cache = nearest;
        candidate.parent = nearest.node;
        rrt_steer(candidate, nodes + size_t(nearest.node) * rrt_dimension,
                  nearest.squared_distance, space, step_size);
    }
    if (width > 1) __syncwarp(rrt_group_mask(width));
}

__device__ inline bool rrt_edge_collision(const RrtCandidate& candidate, uint32_t step,
                                          const Scalar* nodes, const RrtProblem& problem,
                                          const RrtSpace& space, Scene scene, EdgeDirection direction) {
    const Scalar* parent = nodes + size_t(candidate.parent) * rrt_dimension;
    const Scalar* from = direction == EdgeDirection::Forward ? parent : candidate.q;
    const Scalar* to = direction == EdgeDirection::Forward ? candidate.q :
        (direction == EdgeDirection::Reverse ? parent : problem.goal);
    const Scalar ratio = Scalar(step) / Scalar(candidate.steps);
    Scalar q[rrt_dimension];
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        q[d] = step == candidate.steps ? to[d] : (step == 0 ? from[d] :
            from[d] + ratio * (to[d] - from[d]));
    return rrt_collision(q, space, scene, problem.environment_id);
}

// All threads in the block must participate, including those with no edge work.
__device__ inline void rrt_check_edges(RrtCandidate* candidates, uint32_t count,
                                       const Scalar* nodes, const RrtProblem& problem,
                                       const RrtSpace& space, Scene scene, EdgeDirection direction,
                                       bool interleaved = false) {
    __shared__ unsigned int next_edge;
    const uint32_t warps = blockDim.x / 32;
    const bool dynamic = warps > 1 && count > warps;
    // Initial warp assignments need no claim, leaving room for every terminal claim.
    if (threadIdx.x == 0) next_edge = 0;
    // Interleaved checks skip the probe barrier that publishes the work counter.
    if (interleaved && dynamic) __syncthreads();
    if (!interleaved && direction != EdgeDirection::Goal) {
        // Probe existing grid points; accepting an edge still requires the full grid.
        for (uint64_t task = threadIdx.x; task < uint64_t(count) * 4; task += blockDim.x) {
            RrtCandidate& candidate = candidates[task / 4];
            if (candidate.steps <= 4) continue;
            uint32_t step = uint64_t(candidate.steps) * (task % 4 + 1) / 4;
            if (direction == EdgeDirection::Reverse) step = candidate.steps - step;
            if (rrt_edge_collision(candidate, step, nodes, problem, space, scene, direction))
                atomicOr(&candidate.blocked, 1u);
        }
        __syncthreads();
        for (uint32_t c = threadIdx.x; c < count; c += blockDim.x)
            if (candidates[c].blocked) candidates[c].steps = 0;
        __syncthreads();
    }
    if (direction == EdgeDirection::Goal) {
        for (uint32_t c = threadIdx.x; c < count; c += blockDim.x) {
            RrtCandidate& candidate = candidates[c];
            candidate.goal_blocked = candidate.blocked;
            candidate.steps = candidate.blocked || candidate.reaches_target ? 0 :
                rrt_steps(candidate.q, problem.goal, space);
        }
        __syncthreads();
    }
    const unsigned int lane = threadIdx.x % 32;
    for (uint64_t c = threadIdx.x / 32; c < count;) {
        RrtCandidate& candidate = candidates[c];
        // Reverse edges include the new endpoint, which is their source.
        const uint64_t checks = uint64_t(candidate.steps) +
            (direction == EdgeDirection::Reverse && candidate.steps ? 1 : 0);
        // Spreading each warp's checks across the edge avoids separate probes.
        // Every original grid point is still required before accepting an edge.
        const uint64_t waves = (checks + 31) / 32;
        for (uint64_t base = 0; base < checks; base += 32) {
            const uint64_t sample = interleaved ? uint64_t(lane) * waves + base / 32 : base + lane;
            bool hit = false;
            if (sample < checks) {
                const uint32_t step = static_cast<uint32_t>(sample) +
                    (direction == EdgeDirection::Reverse ? 0 : 1);
                hit = rrt_edge_collision(candidate, step, nodes, problem, space, scene, direction);
            }
            if (__any_sync(0xffffffffu, hit)) {
                if (lane == 0)
                    (direction == EdgeDirection::Goal ? candidate.goal_blocked : candidate.blocked) = 1;
                break;
            }
        }
        if (dynamic) {
            unsigned int next = 0;
            if (lane == 0) next = atomicAdd(&next_edge, 1u);
            c = uint64_t(__shfl_sync(0xffffffffu, next, 0)) + warps;
        } else {
            c += warps;
        }
    }
    __syncthreads();
}

__device__ inline uint32_t rrt_append(const Scalar* q, uint32_t parent, Scalar* nodes,
                                      uint32_t* parents, rkern::RrtResult& result) {
    const uint32_t node = result.node_count++;
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        nodes[size_t(node) * rrt_dimension + d] = q[d];
    parents[node] = parent;
    return node;
}

__device__ inline void rrt_start_path(uint32_t node, const uint32_t* parents,
                                      uint32_t* path, rkern::RrtResult& result) {
    result.status = rkern::PlanStatus::Success;
    for (uint32_t n = node;; n = parents[n]) {
        path[result.path_length++] = n;
        if (n == parents[n]) break;
    }
    for (uint32_t i = 0; i < result.path_length / 2; ++i) {
        const uint32_t other = result.path_length - 1 - i;
        const uint32_t value = path[i]; path[i] = path[other]; path[other] = value;
    }
}

__device__ inline void rrt_initialize(const RrtProblem& problem, const RrtSpace& space, Scene scene,
                                      Scalar* nodes, uint32_t* parents, uint32_t* path,
                                      rkern::RrtResult& result, bool& start_valid,
                                      bool& goal_valid, volatile bool& finished, RrtTree* trees = nullptr) {
    if (threadIdx.x == 0) {
        result = {rkern::PlanStatus::RoundLimit, 0, 0, 0};
        finished = false;
        rrt_append(problem.start, 0, nodes, parents, result);
        if (trees) {
            rrt_append(problem.goal, 1, nodes, parents, result);
            trees[0] = RrtTree::Start;
            trees[1] = RrtTree::Goal;
        }
        start_valid = rrt_in_bounds(problem.start, space) &&
            !rrt_collision(problem.start, space, scene, problem.environment_id);
    }
    if (threadIdx.x == 1)
        goal_valid = rrt_in_bounds(problem.goal, space) &&
            !rrt_collision(problem.goal, space, scene, problem.environment_id);
    __syncthreads();
    if (threadIdx.x == 0) {
        if (!start_valid || !goal_valid) {
            result.status = !start_valid ? rkern::PlanStatus::InvalidStart : rkern::PlanStatus::InvalidGoal;
            result.node_count = 0;
            finished = true;
        } else {
            bool equal = true;
            for (uint32_t d = 0; d < rrt_dimension; ++d)
                equal &= problem.start[d] == problem.goal[d];
            if (equal) {
                result.status = rkern::PlanStatus::Success;
                result.path_length = 1;
                path[0] = 0;
                finished = true;
            }
        }
    }
    __syncthreads();
}
}

// Per-problem buffers: nodes[max_nodes][dimension], parents[max_nodes],
// candidates[samples_per_round], path[max_nodes], and one result. Scene/space
// are shared read-only; problem_id and environment_id are independent.
extern "C" __global__ void rkern_rrt(
    const @NAMESPACE@::RrtProblem* problems, size_t problem_count,
    @NAMESPACE@::RrtSpace space, @NAMESPACE@::Scene scene, rkern::RrtConfig config,
    @NAMESPACE@::Scalar* all_nodes, uint32_t* all_parents,
    @NAMESPACE@::RrtCandidate* all_candidates, uint32_t* all_paths,
    rkern::RrtResult* results) {
    using namespace @NAMESPACE@;
    const size_t problem_id = blockIdx.x;
    if (problem_id >= problem_count) return;
    const unsigned int thread = threadIdx.x;
    const RrtProblem& problem = problems[problem_id];
    Scalar* nodes = all_nodes + problem_id * config.max_nodes * rrt_dimension;
    uint32_t* parents = all_parents + problem_id * config.max_nodes;
    uint32_t* path = all_paths + problem_id * config.max_nodes;
    RrtCandidate* candidates = all_candidates + problem_id * config.samples_per_round;
    __shared__ rkern::RrtResult result;
    __shared__ bool start_valid, goal_valid, finished;
    rrt_initialize(problem, space, scene, nodes, parents, path, result,
                   start_valid, goal_valid, finished);
    for (uint32_t round = 0; round < config.max_rounds && !finished; ++round) {
        const uint32_t snapshot_size = result.node_count;
        const uint32_t remaining = config.max_nodes - snapshot_size;
        const uint32_t count = remaining < config.samples_per_round ? remaining : config.samples_per_round;
        for (uint32_t c = thread; c < count; c += blockDim.x) {
            RrtCandidate& candidate = candidates[c];
            rrt_sample(candidate, problem, space, config, round, c, problem.goal);
            rrt_extend(candidate, nodes, snapshot_size, space, config.step_size);
        }
        __syncthreads();
        // Trying only the nearest-to-goal node can repeatedly hit the same obstacle.
        rrt_check_edges(candidates, count, nodes, problem, space, scene, EdgeDirection::Forward);
        rrt_check_edges(candidates, count, nodes, problem, space, scene, EdgeDirection::Goal);
        if (thread == 0) {
            result.rounds = round + 1;
            for (uint32_t c = 0; c < count; ++c) {
                const RrtCandidate& candidate = candidates[c];
                if (candidate.blocked) continue;
                uint32_t node = rrt_append(candidate.q, candidate.parent, nodes, parents, result);
                if (!candidate.goal_blocked && (candidate.reaches_target || result.node_count < config.max_nodes)) {
                    if (!candidate.reaches_target)
                        node = rrt_append(problem.goal, node, nodes, parents, result);
                    rrt_start_path(node, parents, path, result);
                    finished = true;
                    break;
                }
            }
            if (!finished && result.node_count == config.max_nodes) {
                result.status = rkern::PlanStatus::NodeLimit;
                finished = true;
            }
        }
        __syncthreads();
    }
    if (thread == 0) results[problem_id] = result;
}
