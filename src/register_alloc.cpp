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

size_t AllocState::get_available_stack() const {
  auto it = std::find(stack_usages_.begin(), stack_usages_.end(), std::nullopt);
  if (it == stack_usages_.end()) {
    throw std::runtime_error("stack is full");
  }
  return std::distance(stack_usages_.begin(), it);
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
    TransitionSet tset;
    auto& op = opseq[t];

    if (op->args.size() == 0) {
      if (op->kind != OpKind::VALIABLE) {
        throw std::runtime_error("kind is not VALIABLE");
      }
      // determine source location
      auto it_inp_idx =
          std::find_if(inputs.begin(), inputs.end(),
                       [op](const Operation::Ptr& input) { return input->hash_id == op->hash_id; });
      auto inp_idx = std::distance(inputs.begin(), it_inp_idx);
      Location loc_src{LocationType::INPUT, inp_idx};

      // determine destination location
      auto xmm_idx = alloc_state.get_available_xmm();
      if (xmm_idx == std::nullopt) {
        // determine the xmm to be spilled
        auto spill_xmm_idx = alloc_state.most_unused_xmm();
        auto spill_hash_id = alloc_state.xmm_usages_[spill_xmm_idx];
        auto& stash_loc_src = alloc_state.locations_[*spill_hash_id];

        // determine the stack to fill
        auto spill_dst_stack_idx = alloc_state.get_available_stack();
        Location loc_spill_dst{LocationType::STACK, spill_dst_stack_idx};

        // update alloc_state
        alloc_state.xmm_usages_[spill_xmm_idx].reset();
        alloc_state.xmm_ages_[spill_xmm_idx].reset();
        alloc_state.stack_usages_[spill_dst_stack_idx] = spill_hash_id;
        alloc_state.locations_[*spill_hash_id] = loc_spill_dst;

        // record
        tset.push_back({*spill_hash_id, stash_loc_src, loc_spill_dst});

        // now that we have a free xmm so determine loc_dst
        xmm_idx = spill_xmm_idx;
      }
      Location loc_dst = Location{LocationType::REGISTER, *xmm_idx};
      tset.push_back({op->hash_id, loc_src, loc_dst});
    } else {
      auto disappear_hash_ids =
          disappear_table[t];  // these are used as operands but will no longer be used
      // load operands to xmm
      for (auto& operand : op->args) {
      }
    }
  }
  return {};
}

}  // namespace register_alloc
}  // namespace tenkai
