#include "register_alloc.hpp"
#include <algorithm>
#include "cg.hpp"

namespace tenkai {

namespace register_alloc {

AllocState::AllocState(const std::vector<Operation::Ptr>& opseq,
                       const std::vector<Operation::Ptr>& inputs,
                       size_t n_xmm)
    : xmm_usage(n_xmm, std::nullopt),
      stack_usage(opseq.size(), std::nullopt),
      xmm_age(n_xmm, std::nullopt) {
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

void AllocState::tell_xmm_assigned(HashType hash_id, size_t xmm_idx) {
  xmm_usage[xmm_idx] = hash_id;
  xmm_age[xmm_idx] = 0;
  location[hash_id] = Location{LocationType::REGISTER, xmm_idx};
}

void AllocState::spill_away_register(size_t xmm_idx, std::optional<size_t> stack_idx) {
  if (xmm_usage[xmm_idx] == std::nullopt) {
    throw std::runtime_error("xmm_idx is not used");
  }
  if (stack_idx == std::nullopt) {
    auto it = std::find(stack_usage.begin(), stack_usage.end(), std::nullopt);
    if (it == stack_usage.end()) {
      throw std::runtime_error("no stack is available");
    }
    stack_idx = std::distance(stack_usage.begin(), it);
  }

  stack_usage[*stack_idx] = xmm_usage[xmm_idx];
  xmm_usage[xmm_idx] = std::nullopt;
  location[*xmm_usage[xmm_idx]] = Location{LocationType::STACK, *stack_idx};
  xmm_age[xmm_idx] = std::nullopt;
}

void AllocState::load_to_register(size_t stack_idx, size_t xmm_idx) {
  if (stack_usage[stack_idx] == std::nullopt) {
    throw std::runtime_error("anyting is not stored in the stack");
  }
  xmm_usage[xmm_idx] = stack_usage[stack_idx];
  stack_usage[stack_idx] = std::nullopt;
  location[*xmm_usage[xmm_idx]] = Location{LocationType::REGISTER, xmm_idx};
  xmm_age[xmm_idx] = 0;
}

size_t AllocState::most_unused_xmm() const {
  size_t max_age = 0;
  std::optional<size_t> max_age_idx = std::nullopt;
  for (size_t i = 0; i < xmm_age.size(); ++i) {
    if (xmm_age[i] == std::nullopt) {
      return i;
    }
    if (*xmm_age[i] > max_age) {
      max_age = *xmm_age[i];
      max_age_idx = i;
    }
  }
  return *max_age_idx;
}

std::optional<size_t> AllocState::get_available_xmm() const {
  auto it = std::find(xmm_usage.begin(), xmm_usage.end(), std::nullopt);
  if (it == xmm_usage.end()) {
    return std::nullopt;
  }
  return std::distance(xmm_usage.begin(), it);
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

std::vector<TransitionSet> RegisterAllocator::allocate(const std::vector<Operation::Ptr>& opseq,
                                                       const std::vector<Operation::Ptr>& inputs,
                                                       const std::vector<Operation::Ptr>& outputs) {
  auto disappear_table = compute_disappear_hashid_table(opseq);
  auto alloc_state = AllocState(opseq, inputs, 16);

  for (size_t t = 0; t < opseq.size(); ++t) {
    auto& op = opseq[t];
    // operand register preparation
    std::vector<size_t> operand_xmm_indices(op->args.size());
    for (auto& operand : op->args) {
      operand_xmm_indices.push_back(load_to_xmm(alloc_state, operand->hash_id));
    }

    // result register preparation
    std::optional<size_t> opt_result_xmm_idx = std::nullopt;
    auto& disappear_hashids = disappear_table[t];
    for (auto& operand : op->args) {
      auto it = disappear_hashids.find(operand->hash_id);
      if (it != disappear_hashids.end()) {  // if the operand is going to disappear
        opt_result_xmm_idx = load_to_xmm(alloc_state, operand->hash_id);
        break;
      }
    }
    if (opt_result_xmm_idx == std::nullopt) {
      opt_result_xmm_idx = get_available_xmm(alloc_state);
    }
    size_t result_xmm_idx = *opt_result_xmm_idx;
  }
}

size_t RegisterAllocator::load_to_xmm(AllocState& as, HashType hash_id) {
  const auto& locations = as.get_location();
  auto it = locations.find(hash_id);
  bool location_not_found = it == locations.end();
  if (location_not_found) {
    throw std::runtime_error("hash_id is not found");
  }
  const Location& loc = it->second;
  bool already_on_register = loc.type == LocationType::REGISTER;
  if (already_on_register) {
    return loc.idx;
  }
  size_t xmm_idx = as.most_unused_xmm();
  if (as.get_xmm_usage()[xmm_idx] != std::nullopt) {
    as.spill_away_register(xmm_idx, std::nullopt);
    as.load_to_register(loc.idx, xmm_idx);
  }
  return xmm_idx;
}

size_t RegisterAllocator::get_available_xmm(AllocState& as) {
  auto xmm_idx_cand = as.get_available_xmm();
  if (xmm_idx_cand == std::nullopt) {
    size_t xmm_idx = as.most_unused_xmm();
    as.spill_away_register(xmm_idx, std::nullopt);
    return xmm_idx;
  }
  return *xmm_idx_cand;
}

}  // namespace register_alloc
}  // namespace tenkai
