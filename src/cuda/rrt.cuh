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

struct RrtCandidate {
    Scalar q[rrt_dimension];
    uint32_t parent, steps;
    uint64_t offset;
    unsigned int blocked, goal_blocked;
    bool reaches_goal;
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
    double count = 1;
    for (uint32_t d = 0; d < rrt_dimension; ++d)
        count = fmax(count, ceil(fabs(double(b[d]) - double(a[d])) / double(space.resolution[d])));
    return static_cast<uint32_t>(count);
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
    __shared__ uint64_t total_checks;
    if (thread == 0) {
        result = {rkern::PlanStatus::RoundLimit, 1, 0, 0};
        finished = false;
        parents[0] = 0; // The root points to itself; every other parent is an earlier node.
        for (uint32_t d = 0; d < rrt_dimension; ++d) nodes[d] = problem.start[d];
        start_valid = rrt_in_bounds(problem.start, space) &&
            !rrt_collision(problem.start, space, scene, problem.environment_id);
    }
    if (thread == 1)
        goal_valid = rrt_in_bounds(problem.goal, space) &&
            !rrt_collision(problem.goal, space, scene, problem.environment_id);
    __syncthreads();
    if (thread == 0) {
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
    for (uint32_t round = 0; round < config.max_rounds && !finished; ++round) {
        const uint32_t snapshot_size = result.node_count;
        const uint32_t remaining = config.max_nodes - snapshot_size;
        const uint32_t count = remaining < config.samples_per_round ? remaining : config.samples_per_round;
        for (uint32_t c = thread; c < count; c += blockDim.x) {
            RrtCandidate& candidate = candidates[c];
            // Each candidate owns a stream independent of warp count and execution order.
            uint64_t rng = problem.seed +
                (uint64_t(round) * config.samples_per_round + c) * UINT64_C(0xd1342543de82ef95);
            const bool goal_sample = rrt_uniform(rng) < config.goal_bias;
            for (uint32_t d = 0; d < rrt_dimension; ++d)
                candidate.q[d] = goal_sample ? problem.goal[d] : Scalar(
                    double(space.lower[d]) + rrt_uniform(rng) * (double(space.upper[d]) - double(space.lower[d])));
            double best_distance = INFINITY;
            uint32_t nearest = 0;
            for (uint32_t n = 0; n < snapshot_size; ++n) {
                double distance = 0;
                for (uint32_t d = 0; d < rrt_dimension; ++d) {
                    const double delta = double(nodes[size_t(n) * rrt_dimension + d]) - double(candidate.q[d]);
                    distance += delta * delta;
                }
                if (distance < best_distance) { best_distance = distance; nearest = n; }
            }
            const Scalar* parent = nodes + size_t(nearest) * rrt_dimension;
            const double distance = sqrt(best_distance);
            candidate.parent = nearest;
            candidate.reaches_goal = goal_sample && distance <= config.step_size;
            if (distance > config.step_size) {
                const double ratio = config.step_size / distance;
                for (uint32_t d = 0; d < rrt_dimension; ++d)
                    candidate.q[d] = Scalar(double(parent[d]) + ratio * (double(candidate.q[d]) - double(parent[d])));
            }
            bool moved = false;
            for (uint32_t d = 0; d < rrt_dimension; ++d) moved |= candidate.q[d] != parent[d];
            candidate.blocked = !moved;
            candidate.steps = moved ? rrt_steps(parent, candidate.q, space) : 0;
        }
        __syncthreads();
        // Validate extensions first, then connect each valid endpoint to the goal.
        // Trying only the nearest-to-goal node can repeatedly hit the same obstacle.
        for (unsigned int phase = 0; phase < 2; ++phase) {
            if (thread == 0) {
                total_checks = 0;
                for (uint32_t c = 0; c < count; ++c) {
                    RrtCandidate& candidate = candidates[c];
                    if (phase == 1) {
                        candidate.goal_blocked = candidate.blocked;
                        candidate.steps = candidate.blocked || candidate.reaches_goal ? 0 :
                            rrt_steps(candidate.q, problem.goal, space);
                    }
                    candidate.offset = total_checks;
                    total_checks += candidate.steps;
                }
            }
            __syncthreads();
            for (uint64_t task = thread; task < total_checks; task += blockDim.x) {
                uint32_t low = 0, high = count;
                while (low + 1 < high) {
                    const uint32_t middle = low + (high - low) / 2;
                    if (candidates[middle].offset <= task) low = middle;
                    else high = middle;
                }
                RrtCandidate& candidate = candidates[low];
                const Scalar* from = phase == 0 ? nodes + size_t(candidate.parent) * rrt_dimension : candidate.q;
                const Scalar* to = phase == 0 ? candidate.q : problem.goal;
                const uint32_t step = static_cast<uint32_t>(task - candidate.offset) + 1;
                const double ratio = double(step) / candidate.steps;
                Scalar q[rrt_dimension];
                for (uint32_t d = 0; d < rrt_dimension; ++d)
                    q[d] = step == candidate.steps ? to[d] : Scalar(
                        double(from[d]) + ratio * (double(to[d]) - double(from[d])));
                if (rrt_collision(q, space, scene, problem.environment_id))
                    atomicOr(phase == 0 ? &candidate.blocked : &candidate.goal_blocked, 1u);
            }
            __syncthreads();
        }
        if (thread == 0) {
            result.rounds = round + 1;
            for (uint32_t c = 0; c < count; ++c) {
                const RrtCandidate& candidate = candidates[c];
                if (candidate.blocked) continue;
                uint32_t node = result.node_count++;
                for (uint32_t d = 0; d < rrt_dimension; ++d)
                    nodes[size_t(node) * rrt_dimension + d] = candidate.q[d];
                parents[node] = candidate.parent;
                if (!candidate.goal_blocked && (candidate.reaches_goal || result.node_count < config.max_nodes)) {
                    if (!candidate.reaches_goal) {
                        const uint32_t goal = result.node_count++;
                        for (uint32_t d = 0; d < rrt_dimension; ++d)
                            nodes[size_t(goal) * rrt_dimension + d] = problem.goal[d];
                        parents[goal] = node;
                        node = goal;
                    }
                    result.status = rkern::PlanStatus::Success;
                    for (uint32_t n = node;; n = parents[n]) {
                        path[result.path_length++] = n;
                        if (n == 0) break;
                    }
                    for (uint32_t i = 0; i < result.path_length / 2; ++i) {
                        const uint32_t other = result.path_length - 1 - i;
                        const uint32_t value = path[i]; path[i] = path[other]; path[other] = value;
                    }
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
