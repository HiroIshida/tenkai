#include "register_alloc.hpp"
#include <algorithm>
#include <format>
#include <iostream>
#include "cg.hpp"

namespace tenkai {

namespace register_alloc {

std::ostream& operator<<(std::ostream& os, const Location& loc) {
  switch (loc.type) {
    case LocationType::REGISTER:
      os << std::format("xmm({})", loc.idx);
      break;
    case LocationType::STACK:
      os << std::format("stack({})", loc.idx);
      break;
    case LocationType::INPUT:
      os << std::format("input({})", loc.idx);
      break;
    case LocationType::OUTPUT:
      os << std::format("output({})", loc.idx);
      break;
  }
  return os;
}

// T is temporary. Actuall stack size is determined by the max_stack_usage_
AllocState::AllocState(const std::vector<Operation::Ptr>& inputs, size_t T, size_t n_xmm)
    : xmm_usages_(n_xmm, std::nullopt),
      stack_usages_(T, std::nullopt),
      xmm_ages_(n_xmm, std::nullopt) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& op = inputs[i];
    if (op->kind != OpKind::VALIABLE) {
      throw std::runtime_error("kind is not VALIABLE");
    }
    locations_[op->hash_id] = Location{LocationType::INPUT, i};
  }
}

void AllocState::update_xmm_ages() {
  for (size_t i = 0; i < xmm_ages_.size(); ++i) {
    if (xmm_ages_[i] != std::nullopt) {
      ++*xmm_ages_[i];
    }
  }
}

size_t AllocState::most_unused_xmm() const {
  size_t max_age = 0;
  std::optional<size_t> max_age_idx = std::nullopt;
  for (size_t i = 0; i < xmm_ages_.size(); ++i) {
    if (xmm_ages_[i] == std::nullopt) {
      return i;
    }
    if (*xmm_ages_[i] > max_age) {
      max_age = *xmm_ages_[i];
      max_age_idx = i;
    }
  }
  return *max_age_idx;
}

std::optional<size_t> AllocState::get_available_xmm() const {
  auto it = std::find(xmm_usages_.begin(), xmm_usages_.end(), std::nullopt);
  if (it == xmm_usages_.end()) {
    return std::nullopt;
  }
  return std::distance(xmm_usages_.begin(), it);
}

std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq) {
  std::vector<std::unordered_set<HashType>> table(opseq.size());
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
  auto alloc_state = AllocState(inputs, opseq.size(), 6);

  for (size_t t = 0; t < opseq.size(); ++t) {
    auto& op = opseq[t];
    if (op->args.size() == 0) {
      if (op->kind != OpKind::VALIABLE) {
        throw std::runtime_error("kind is not VALIABLE");
      }
      // record_load_from_input(hash_id)
    } else {
      for (auto& operand : op->args) {
        // call record_prepare_operand_on_register(hash_id, disappera_hashids)
        // disappear hash _id can be reused
      }
      // record_prepare_result_on_register(hash_id, disappera_hashids)
    }
  }
  return {};
}

}  // namespace register_alloc
}  // namespace tenkai
