
// Per-problem buffers: nodes[max_nodes][dimension], parents[max_nodes],
// candidates[samples_per_round], path[max_nodes], and one result. Scene/space
// are shared read-only; problem_id and environment_id are independent.
extern "C" __global__ void tenkai_rrt(
    const @NAMESPACE@::RrtProblem* problems, size_t problem_count,
    @NAMESPACE@::RrtSpace space, @NAMESPACE@::Scene scene, tenkai::RrtConfig config,
    @NAMESPACE@::Scalar* all_nodes, uint32_t* all_parents,
    @NAMESPACE@::RrtCandidate* all_candidates, uint32_t* all_paths,
    tenkai::RrtResult* results) {
    using namespace @NAMESPACE@;
    const size_t problem_id = blockIdx.x;
    if (problem_id >= problem_count) return;
    const unsigned int thread = threadIdx.x;
    const RrtProblem& problem = problems[problem_id];
    Scalar* nodes = all_nodes + problem_id * config.max_nodes * rrt_dimension;
    uint32_t* parents = all_parents + problem_id * config.max_nodes;
    uint32_t* path = all_paths + problem_id * config.max_nodes;
    RrtCandidate* candidates = all_candidates + problem_id * config.samples_per_round;
    __shared__ tenkai::RrtResult result;
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
                result.status = tenkai::PlanStatus::NodeLimit;
                finished = true;
            }
        }
        __syncthreads();
    }
    if (thread == 0) results[problem_id] = result;
}
