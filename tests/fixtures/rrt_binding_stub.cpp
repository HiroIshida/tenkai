// A CPU-only C ABI fixture for array marshalling and planner lifetime tests.
#include <algorithm>
#include <cstdint>
#include <vector>

struct Config {
    uint32_t max_nodes, max_rounds;
    double step_size, goal_bias;
};
struct Result { uint32_t status, node_count, rounds, path_length; };
struct Context {
    Config config;
    uint32_t problems, environments;
    std::vector<double> starts, goals;
};
static int alive = 0;
static const char* error = "";

extern "C" const char* rrt_error() { return error; }
extern "C" int live_contexts() { return alive; }

extern "C" int rrtc_create(const Config* config, uint32_t problems,
                          uint32_t environments, const double*, const double*,
                          const double*, const double*, void** output) {
    *output = new Context{*config, problems, environments, {}, {}};
    ++alive;
    return 0;
}
extern "C" void rrt_destroy(void* handle) {
    delete static_cast<Context*>(handle);
    --alive;
}
extern "C" int rrt_solve(void* handle, const double* starts, const double* goals,
                        const uint32_t* ids, const uint64_t* seeds, Result* output) {
    auto& context = *static_cast<Context*>(handle);
    context.starts.assign(starts, starts + context.problems * 2);
    context.goals.assign(goals, goals + context.problems * 2);
    for (uint32_t i = 0; i < context.problems; ++i) {
        if (ids[i] >= context.environments) return -1;
        output[i] = {0, uint32_t(seeds[i] ^ (seeds[i] >> 32)),
                     context.config.max_rounds, 3};
    }
    return 0;
}
extern "C" int rrt_path(void* handle, uint32_t problem, double* output,
                       uint32_t capacity, uint32_t* length) {
    auto& context = *static_cast<Context*>(handle);
    if (context.starts.empty() || problem >= context.problems || capacity < 3) {
        error = "path unavailable";
        return -1;
    }
    const double ratio = context.config.step_size * context.config.goal_bias;
    for (uint32_t d = 0; d < 2; ++d) {
        const double start = context.starts[problem * 2 + d];
        const double goal = context.goals[problem * 2 + d];
        output[d] = start;
        output[2 + d] = start + ratio * (goal - start);
        output[4 + d] = goal;
    }
    *length = 3;
    return 0;
}
extern "C" int rrt_check(void*, const double* configurations, const uint32_t* ids,
                        uint32_t count, unsigned char* output) {
    for (uint32_t i = 0; i < count; ++i)
        output[i] = configurations[i * 3 + 2] < double(ids[i]);
    return 0;
}
