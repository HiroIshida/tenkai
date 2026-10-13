// One warp owns the single candidate. EXTEND and CONNECT share the collision
// call site so its FK values can stay in registers without duplicating the DAG.
extern "C" __global__ void rkern_rrtc_single(
    const @NAMESPACE@::RrtProblem* problems, size_t problem_count,
    @NAMESPACE@::RrtSpace space, @NAMESPACE@::Scene scene, rkern::RrtConfig config,
    @NAMESPACE@::Scalar* all_nodes, uint32_t* all_parents, @NAMESPACE@::RrtTree* all_trees,
    @NAMESPACE@::RrtCandidate* all_candidates, uint32_t* all_paths,
    rkern::RrtResult* results, const unsigned char* clear) {
    using namespace @NAMESPACE@;
    const size_t id = blockIdx.x;
    if (id >= problem_count) return;
    const uint32_t lane = threadIdx.x;
    const RrtProblem& problem = problems[id];
    Scalar* nodes = all_nodes + id * config.max_nodes * rrt_dimension;
    uint32_t* parents = all_parents + id * config.max_nodes;
    RrtTree* trees = all_trees + id * config.max_nodes;
    uint32_t* path = all_paths + id * config.max_nodes;
    __shared__ RrtCandidate candidate;
    __shared__ rkern::RrtResult result;
    __shared__ bool start_valid, goal_valid, connecting, in_workspace;
    __shared__ volatile bool finished;
    __shared__ uint32_t round, target;
    if (lane == 0) { round = 0; connecting = false; }
    rrt_initialize(problem, space, scene, nodes, parents, path, result,
                   start_valid, goal_valid, finished, trees);
    if (lane == 0 && !finished) {
        if (result.node_count == config.max_nodes) {
            result.status = rkern::PlanStatus::NodeLimit;
            finished = true;
        }
    }
    __syncwarp();
    while (round < config.max_rounds && !finished) {
        const RrtTree active = round % 2 == 0 ? RrtTree::Start : RrtTree::Goal;
        const RrtTree tree = connecting ?
            (active == RrtTree::Start ? RrtTree::Goal : RrtTree::Start) : active;
        if (lane == 0) {
            if (connecting) {
                for (uint32_t d = 0; d < rrt_dimension; ++d)
                    candidate.q[d] = nodes[size_t(target) * rrt_dimension + d];
                candidate.reaches_target = true;
            } else {
                rrt_sample(candidate, problem, space, config, round, 0,
                           active == RrtTree::Start ? problem.goal : problem.start);
            }
        }
        __syncwarp();
        const uint32_t unscanned = result.node_count - (connecting ? candidate.nearest.scanned : 0);
        const uint32_t width = unscanned < 32 ? 1 : 32;
        if (lane < width)
            rrt_extend(candidate, nodes, result.node_count, space, config.step_size,
                       trees, tree, connecting ? &candidate.nearest : nullptr, width, true);
        if (lane == 0 && connecting && !candidate.steps && candidate.reaches_target)
            candidate.blocked = 0;
        if (lane == 0)
            in_workspace = clear[problem.environment_id] &&
                rkern_collision_in_workspace(candidate.q, space.lower, space.upper) &&
                rkern_collision_in_workspace(nodes + size_t(candidate.parent) * rrt_dimension,
                                             space.lower, space.upper);
        __syncwarp();
        const bool reverse = tree == RrtTree::Goal;
        const uint64_t checks = uint64_t(candidate.steps) + (reverse && candidate.steps ? 1 : 0);
        const uint64_t waves = (checks + 31) / 32;
        for (uint64_t wave = 0; wave < waves; ++wave) {
            const uint64_t sample = uint64_t(lane) * waves + wave;
            bool hit = false;
            if (sample < checks) {
                const uint32_t step = uint32_t(sample) + (reverse ? 0 : 1);
                const Scalar ratio = Scalar(step) / Scalar(candidate.steps);
                const Scalar* parent = nodes + size_t(candidate.parent) * rrt_dimension;
                const Scalar* from = reverse ? candidate.q : parent;
                const Scalar* to = reverse ? parent : candidate.q;
                const uint32_t controls[rrt_dimension] = {@CONTROLS@};
                Scalar full[rrt_fk_dof];
                for (uint32_t d = 0; d < rrt_fk_dof; ++d) full[d] = space.reference[d];
                for (uint32_t d = 0; d < rrt_dimension; ++d) {
                    const Scalar q = step == candidate.steps ? to[d] :
                        (step == 0 ? from[d] : from[d] + ratio * (to[d] - from[d]));
                    full[controls[d]] = q;
                }
                hit = in_workspace ? rkern_collision_edge(full, scene, problem.environment_id) :
                    rkern_collision_moving(full, scene, problem.environment_id);
            }
            if (__any_sync(0xffffffffu, hit)) {
                if (lane == 0) candidate.blocked = 1;
                break;
            }
        }
        __syncwarp();
        if (lane == 0) {
            if (!connecting) result.rounds = round + 1;
            if (candidate.blocked) {
                connecting = false;
                ++round;
                if (result.node_count == config.max_nodes) {
                    result.status = rkern::PlanStatus::NodeLimit;
                    finished = true;
                }
            } else if (!connecting) {
                target = rrt_append(candidate.q, candidate.parent, nodes, parents, result);
                trees[target] = active;
                candidate.nearest = {0, 0, INFINITY};
                connecting = true;
            } else {
                uint32_t node = candidate.parent;
                if (candidate.steps) {
                    if (result.node_count == config.max_nodes) {
                        result.status = rkern::PlanStatus::NodeLimit;
                        finished = true;
                    } else {
                        node = rrt_append(candidate.q, candidate.parent, nodes, parents, result);
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
                    result.status = rkern::PlanStatus::NodeLimit;
                    finished = true;
                }
            }
        }
        __syncwarp();
    }
    if (lane == 0) results[id] = result;
}
extern "C" __global__ void rkern_rrtc_workspace(
    @NAMESPACE@::RrtSpace space, @NAMESPACE@::Scene scene,
    size_t environment_count, unsigned char* clear) {
    for (size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         id < environment_count; id += size_t(blockDim.x) * gridDim.x)
        clear[id] = rkern_collision_workspace_clear(
            space.lower, space.upper, space.reference, scene, uint32_t(id));
}
