#ifndef RKERN_RRT_TYPES_HPP
#define RKERN_RRT_TYPES_HPP
#include <stdint.h>

namespace rkern {
enum class RrtAlgorithm : uint32_t { Rrt, RrtConnect };

struct RrtConfig {
    uint32_t max_nodes = 8192;
    uint32_t max_rounds = 1024;
    uint32_t samples_per_round = 32;
    double step_size = 0.5;
    double goal_bias = 0.1;
};

enum class PlanStatus : uint32_t {
    Success,
    RoundLimit,
    NodeLimit,
    InvalidStart,
    InvalidGoal,
};

struct RrtResult {
    PlanStatus status;
    uint32_t node_count;
    uint32_t rounds;
    uint32_t path_length;
};
}
#endif
