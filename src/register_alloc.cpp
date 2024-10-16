#include "register_alloc.hpp"
#include <algorithm>
#include "cg.hpp"

namespace tenkai {

namespace register_alloc {

AllocState::AllocState(const std::vector<Operation::Ptr>& opseq,
                       const std::vector<Operation::Ptr>& inputs,
                       size_t n_xmm)
    : xmm_usage(n_xmm, std::nullopt), stack_usage(opseq.size(), std::nullopt) {
  for (auto& op : opseq) {
    if (op->kind == OpKind::VALIABLE) {
      auto it = std::find_if(inputs.begin(), inputs.end(), [&](const Operation::Ptr& input) {
        return input->hash_id == op->hash_id;
      });
      size_t idx = std::distance(inputs.begin(), it);
      location[op->hash_id] = Location{LocationType::INPUT, idx};
    }
  }
}

std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq) {
  std::vector<std::unordered_set<HashType>> table;
  std::unordered_set<int32_t> visited;
  for (int i = opseq.size() - 1; i >= 0; --i) {
    for (auto& arg_op : opseq[i]->args) {
      bool unappended = visited.find(arg_op->hash_id) == visited.end();
      if (unappended) {
        table[i].insert(arg_op->hash_id);
        visited.insert(arg_op->hash_id);
      }
    }
  }
  return table;
}

std::vector<TransitionSet> AdhocRegisterAllocator::allocate(
    const std::vector<Operation::Ptr>& opseq) {
  auto disappear_table = compute_disappear_hashid_table(opseq);
}

}  // namespace register_alloc
}  // namespace tenkai
