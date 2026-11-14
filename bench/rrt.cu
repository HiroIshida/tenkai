#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "rrt.cuh"

namespace model = TENKAI_MODEL_NAMESPACE;
using Scalar = model::Scalar;

namespace {
thread_local std::string last_error;

template <typename F> int boundary(F operation) {
    try {
        operation();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

struct Context {
    size_t problem_count, environment_count;
    model::DeviceBuffer<model::Environment> environments;
    model::DeviceBuffer<model::Box> boxes;
    model::DeviceBuffer<model::Ground> grounds;
    model::Scene scene;
    std::unique_ptr<model::RrtPlanner> planner;

    Context(const tenkai::RrtConfig& config, uint32_t warps, size_t problems, size_t envs,
            const double* geometry, const double* lower, const double* upper, const double* resolution,
            tenkai::RrtAlgorithm algorithm)
        : problem_count(problems), environment_count(envs), environments(envs), boxes(envs), grounds(envs) {
        model::RrtSpace space{};
        for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
            space.lower[d] = Scalar(lower[d]);
            space.upper[d] = Scalar(upper[d]);
            space.resolution[d] = Scalar(resolution[d]);
        }
        model::validate_rrt(config, space);
        std::vector<model::Environment> host_environments(envs);
        std::vector<model::Box> host_boxes(envs);
        std::vector<model::Ground> host_grounds(envs);
        for (size_t i = 0; i < envs; ++i) {
            const double* g = geometry + 7 * i;
            for (size_t j = 0; j < 7; ++j)
                if (!std::isfinite(g[j]) || !std::isfinite(Scalar(g[j])))
                    throw std::invalid_argument("nonfinite scene geometry");
            if (g[3] < 0 || g[4] < 0 || g[5] < 0)
                throw std::invalid_argument("negative box extent");
            host_boxes[i] = {{{Scalar(g[0]), Scalar(g[1]), Scalar(g[2])}, {1,0,0,0,1,0,0,0,1}},
                             {Scalar(g[3]), Scalar(g[4]), Scalar(g[5])}};
            host_grounds[i] = {Scalar(g[6])};
            host_environments[i] = {{i,1}, {0,0}, {0,0}, {i,1}};
        }
        environments.upload(host_environments.data(), envs);
        boxes.upload(host_boxes.data(), envs);
        grounds.upload(host_grounds.data(), envs);
        scene = {environments.data(), boxes.data(), nullptr, nullptr, grounds.data()};
        planner.reset(new model::RrtPlanner(config, {warps}, space, scene, envs, problems, algorithm));
    }
};

Context& context(void* handle) {
    if (!handle) throw std::invalid_argument("null RRT handle");
    return *static_cast<Context*>(handle);
}

int create(const tenkai::RrtConfig* config, uint32_t warps, uint32_t problems,
           uint32_t environments, const double* geometry, const double* lower,
           const double* upper, const double* resolution, void** output, tenkai::RrtAlgorithm algorithm) {
    return boundary([&] {
        if (!output) throw std::invalid_argument("null RRT output handle");
        *output = nullptr;
        if (!config || !problems || !environments || !geometry || !lower || !upper || !resolution)
            throw std::invalid_argument("invalid RRT creation arguments");
        *output = new Context(*config, warps, problems, environments, geometry, lower, upper, resolution, algorithm);
    });
}
}

extern "C" const char* rrt_error() { return last_error.c_str(); }

extern "C" int rrt_create(const tenkai::RrtConfig* config, uint32_t warps,
                          uint32_t problems, uint32_t environments, const double* geometry,
                          const double* lower, const double* upper, const double* resolution,
                          void** output) {
    return create(config, warps, problems, environments, geometry, lower, upper, resolution,
                  output, tenkai::RrtAlgorithm::Rrt);
}

extern "C" int rrtc_create(const tenkai::RrtConfig* config, uint32_t warps,
                           uint32_t problems, uint32_t environments, const double* geometry,
                           const double* lower, const double* upper, const double* resolution,
                           void** output) {
    return create(config, warps, problems, environments, geometry, lower, upper, resolution,
                  output, tenkai::RrtAlgorithm::RrtConnect);
}

extern "C" void rrt_destroy(void* handle) { delete static_cast<Context*>(handle); }

extern "C" int rrt_solve(void* handle, const double* starts, const double* goals,
                         const uint32_t* environment_ids, const uint64_t* seeds,
                         tenkai::RrtResult* output) {
    return boundary([&] {
        auto& ctx = context(handle);
        if (!starts || !goals || !environment_ids || !seeds || !output)
            throw std::invalid_argument("null RRT solve input/output");
        std::vector<model::RrtProblem> problems(ctx.problem_count);
        for (size_t p = 0; p < problems.size(); ++p) {
            problems[p].environment_id = environment_ids[p];
            problems[p].seed = seeds[p];
            for (uint32_t d = 0; d < model::rrt_dimension; ++d) {
                problems[p].start[d] = Scalar(starts[p * model::rrt_dimension + d]);
                problems[p].goal[d] = Scalar(goals[p * model::rrt_dimension + d]);
            }
        }
        const auto results = ctx.planner->solve(problems);
        std::copy(results.begin(), results.end(), output);
    });
}

extern "C" int rrt_path(void* handle, uint32_t problem, double* output, uint32_t capacity,
                        uint32_t* length) {
    return boundary([&] {
        const auto path = context(handle).planner->path(problem);
        if (!length || (!path.empty() && !output)) throw std::invalid_argument("null RRT path output");
        *length = static_cast<uint32_t>(path.size() / model::rrt_dimension);
        if (*length > capacity) throw std::length_error("path output buffer too small");
        std::copy(path.begin(), path.end(), output);
    });
}

extern "C" int rrt_check(void* handle, const double* configurations,
                         const uint32_t* environment_ids, uint32_t count, unsigned char* output) {
    return boundary([&] {
        const auto& ctx = context(handle);
        if (!count) return;
        if (!configurations || !environment_ids || !output)
            throw std::invalid_argument("null collision input/output");
        std::vector<Scalar> q(size_t(count) * model::rrt_fk_dof);
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = Scalar(configurations[i]);
            if (!std::isfinite(q[i])) throw std::invalid_argument("nonfinite collision input");
        }
        for (size_t i = 0; i < count; ++i)
            if (environment_ids[i] >= ctx.environment_count) throw std::invalid_argument("invalid environment ID");
        model::DeviceBuffer<Scalar> device_q(q.size());
        model::DeviceBuffer<uint32_t> ids(count);
        model::DeviceBuffer<unsigned char> flags(count);
        device_q.upload(q.data(), q.size());
        ids.upload(environment_ids, count);
        tenkai_collision<<<(size_t(count) + 127) / 128, 128>>>(device_q.data(), ids.data(), ctx.scene, flags.data(), count);
        model::cuda_check(cudaGetLastError(), "launch path validation");
        flags.download(output, count);
    });
}
