#include "operation_scheduler.hpp"
#include <stack>
#include <unordered_set>

namespace tenkai {

namespace compiler {

std::vector<Operation::Ptr> DepthFirstScheduler::flatten(
    const std::vector<Operation::Ptr>& inputs,
    const std::vector<Operation::Ptr>& outputs) {
  std::vector<Operation::Ptr> operations;

  // Order the operation using depth-first search
  // DFS is better than BFS because the operation is likely to be used immediately
  // after it is calculated, and will consume less xmm register
  std::stack<Operation::Ptr> opstack;
  for (auto& output : outputs) {
    opstack.push(output);
  }
  while (!opstack.empty()) {
    auto op = opstack.top();
    opstack.pop();
    operations.push_back(op);
    for (auto& arg : op->args) {
      opstack.push(arg);
    }
  }

  // Do common subexpression elimination
  std::unordered_set<int32_t> visited;
  std::vector<Operation::Ptr> result;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    if (visited.find((*it)->hash_id) != visited.end()) {
      continue;
    }
    visited.insert((*it)->hash_id);
    result.push_back(*it);
  }
  return result;
}

}  // namespace compiler
}  // namespace tenkai
