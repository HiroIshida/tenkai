#include "operation_scheduler.hpp"
#include <stack>
#include <unordered_set>
#include "cg.hpp"

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

  std::unordered_set<int32_t> visited;
  std::vector<Operation::Ptr> result;

  // extcall-first optimization
  for (const auto& inp : inputs) {
    for (const auto& caller_wptr : inp->callers) {
      auto caller = caller_wptr.lock();
      if (caller->kind == OpKind::SIN || caller->kind == OpKind::COS) {
        result.push_back(caller->args[0]);
        result.push_back(caller);
        visited.insert(caller->hash_id);
        visited.insert(caller->args[0]->hash_id);
      }
    }
  }

  // Do common subexpression elimination
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    if (visited.find((*it)->hash_id) != visited.end()) {
      continue;
    }
    visited.insert((*it)->hash_id);
    result.push_back(*it);
  }
  return result;
}

std::vector<Operation::Ptr> ExtCallFirstScheduler::flatten(
    const std::vector<Operation::Ptr>& inputs,
    const std::vector<Operation::Ptr>& outputs) {
  std::unordered_set<int32_t> visited;
  std::vector<Operation::Ptr> ext_evals;

  for (const auto& inp : inputs) {
    for (const auto& caller_wptr : inp->callers) {
      auto caller = caller_wptr.lock();
      if (caller->kind == OpKind::SIN || caller->kind == OpKind::COS) {
        ext_evals.push_back(caller);
        ext_evals.push_back(caller->args[0]);
        visited.insert(caller->hash_id);
        visited.insert(caller->args[0]->hash_id);
      }
    }
  }
}

}  // namespace compiler
}  // namespace tenkai
