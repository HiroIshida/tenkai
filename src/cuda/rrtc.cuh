// Both trees share max_nodes slots. Roots point to themselves; other parents
// precede their children. Tree tags keep nearest-neighbor searches separate.
extern "C" __global__ void rkern_rrtc(
    const @NAMESPACE@::RrtProblem* problems, size_t problem_count,
    @NAMESPACE@::RrtSpace space, @NAMESPACE@::Scene scene, rkern::RrtConfig config,
    @NAMESPACE@::Scalar* all_nodes, uint32_t* all_parents, @NAMESPACE@::RrtTree* all_trees,
    @NAMESPACE@::RrtCandidate* all_candidates, uint32_t* all_paths,
    rkern::RrtResult* results) {
    using namespace @NAMESPACE@;
    const size_t problem_id = blockIdx.x;
    if (problem_id >= problem_count) return;
    const unsigned int thread = threadIdx.x;
    const RrtProblem& problem = problems[problem_id];
    Scalar* nodes = all_nodes + problem_id * config.max_nodes * rrt_dimension;
    uint32_t* parents = all_parents + problem_id * config.max_nodes;
    RrtTree* trees = all_trees + problem_id * config.max_nodes;
    uint32_t* path = all_paths + problem_id * config.max_nodes;
    RrtCandidate* candidates = all_candidates + problem_id * config.samples_per_round;
    __shared__ rkern::RrtResult result;
    __shared__ bool start_valid, goal_valid, finished, connecting;
    rrt_initialize(problem, space, scene, nodes, parents, path, result,
                   start_valid, goal_valid, finished, trees);
    if (thread == 0 && !finished && result.node_count == config.max_nodes) {
        result.status = rkern::PlanStatus::NodeLimit;
        finished = true;
    }
    __syncthreads();
    for (uint32_t round = 0; round < config.max_rounds && !finished; ++round) {
        const RrtTree active = round % 2 == 0 ? RrtTree::Start : RrtTree::Goal;
        const RrtTree other = active == RrtTree::Start ? RrtTree::Goal : RrtTree::Start;
        const uint32_t snapshot_size = result.node_count;
        const uint32_t remaining = config.max_nodes - snapshot_size;
        const uint32_t count = remaining < config.samples_per_round ? remaining : config.samples_per_round;
        for (uint32_t c = thread; c < count; c += blockDim.x) {
            RrtCandidate& candidate = candidates[c];
            rrt_sample(candidate, problem, space, config, round, c,
                       active == RrtTree::Start ? problem.goal : problem.start);
            rrt_extend(candidate, nodes, snapshot_size, space, config.step_size, trees, active);
        }
        __syncthreads();
        rrt_check_edges(candidates, count, nodes, problem, space, scene,
                        active == RrtTree::Start ? EdgeDirection::Forward : EdgeDirection::Reverse);
        if (thread == 0) {
            result.rounds = round + 1;
            connecting = false;
            for (uint32_t c = 0; c < count; ++c) {
                RrtCandidate& candidate = candidates[c];
                candidate.state = GrowState::Trapped;
                if (candidate.blocked) continue;
                candidate.target = rrt_append(candidate.q, candidate.parent, nodes, parents, result);
                trees[candidate.target] = active;
                candidate.state = GrowState::Advanced;
                connecting = true;
            }
        }
        __syncthreads();
        // Every Advanced connection adds a node, bounding CONNECT by max_nodes.
        // Each wave reads a snapshot; insertion order is independent of warps.
        while (connecting && !finished) {
            const uint32_t connect_snapshot = result.node_count;
            for (uint32_t c = thread; c < count; c += blockDim.x) {
                RrtCandidate& candidate = candidates[c];
                candidate.steps = 0;
                if (candidate.state != GrowState::Advanced) continue;
                for (uint32_t d = 0; d < rrt_dimension; ++d)
                    candidate.q[d] = nodes[size_t(candidate.target) * rrt_dimension + d];
                candidate.reaches_target = true;
                rrt_extend(candidate, nodes, connect_snapshot, space, config.step_size, trees, other);
                // An existing node at the target is already a valid connection.
                if (!candidate.steps && candidate.reaches_target) candidate.blocked = 0;
            }
            __syncthreads();
            rrt_check_edges(candidates, count, nodes, problem, space, scene,
                            other == RrtTree::Start ? EdgeDirection::Forward : EdgeDirection::Reverse);
            if (thread == 0) {
                connecting = false;
                for (uint32_t c = 0; c < count; ++c) {
                    RrtCandidate& candidate = candidates[c];
                    if (candidate.state != GrowState::Advanced) continue;
                    if (candidate.blocked) {
                        candidate.state = GrowState::Trapped;
                        continue;
                    }
                    uint32_t node = candidate.parent;
                    if (candidate.steps) {
                        if (result.node_count == config.max_nodes) continue;
                        node = rrt_append(candidate.q, candidate.parent, nodes, parents, result);
                        trees[node] = other;
                    }
                    candidate.state = candidate.reaches_target ? GrowState::Reached : GrowState::Advanced;
                    if (candidate.state == GrowState::Reached) {
                        const uint32_t start = active == RrtTree::Start ? candidate.target : node;
                        const uint32_t goal = active == RrtTree::Goal ? candidate.target : node;
                        rrt_start_path(start, parents, path, result);
                        // The connection coordinates occur in both trees. Keep
                        // them once so the path has no zero-length join edge.
                        for (uint32_t n = goal; n != parents[n];) {
                            n = parents[n];
                            path[result.path_length++] = n;
                        }
                        finished = true;
                        break;
                    }
                    connecting = true;
                }
            }
            __syncthreads();
            if (thread == 0 && !finished && result.node_count == config.max_nodes) {
                result.status = rkern::PlanStatus::NodeLimit;
                finished = true;
            }
            __syncthreads();
        }
        if (thread == 0 && !finished && result.node_count == config.max_nodes) {
            result.status = rkern::PlanStatus::NodeLimit;
            finished = true;
        }
        __syncthreads();
    }
    if (thread == 0) results[problem_id] = result;
}
