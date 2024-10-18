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

AllocState::AllocState(const std::vector<Operation::Ptr>& opseq,
                       const std::vector<Operation::Ptr>& inputs,
                       size_t n_xmm)
    : xmm_usages_(n_xmm, std::nullopt),
      stack_usages_(opseq.size(), std::nullopt),
      xmm_ages_(n_xmm, std::nullopt),
      max_stack_usage_(0),
      history_(opseq.size(), std::vector<Transition>()),
      t_(std::nullopt) {
  for (auto& op : opseq) {
    if (op->kind == OpKind::VALIABLE) {
      auto it = std::find_if(inputs.begin(), inputs.end(), [&](const Operation::Ptr& input) {
        return input->hash_id == op->hash_id;
      });
      size_t idx = std::distance(inputs.begin(), it);
      locations_[op->hash_id] = Location{LocationType::INPUT, idx};
    }
  }
}

void AllocState::step() {
  for (size_t i = 0; i < xmm_ages_.size(); ++i) {
    if (xmm_ages_[i] != std::nullopt) {
      ++*xmm_ages_[i];
    }
  }
  if (t_ == std::nullopt) {
    t_ = 0;
  } else {
    ++*t_;
  }
}

void AllocState::record_load_from_input(HashType hash_id, size_t xmm_idx) {
  // update history
  auto&& loc_src = locations_[hash_id];  // must exist
  auto loc_dest = Location{LocationType::REGISTER, xmm_idx};
  Transition&& transition = std::make_tuple(hash_id, loc_src, loc_dest);
  history_[*t_].emplace_back(transition);

  // update current state
  xmm_usages_[xmm_idx] = hash_id;
  xmm_ages_[xmm_idx] = 0;
  locations_[hash_id] = std::move(loc_dest);
}

void AllocState::recored_xmm_assigned_as_op_result(HashType hash_id, size_t xmm_idx) {
  // update history
  auto&& loc_src = std::nullopt;
  auto&& loc_dest = Location{LocationType::REGISTER, xmm_idx};
  Transition&& transition = std::make_tuple(hash_id, loc_src, loc_dest);
  history_[*t_].emplace_back(transition);

  // update current state
  xmm_usages_[xmm_idx] = hash_id;
  xmm_ages_[xmm_idx] = 0;
  locations_[hash_id] = Location{LocationType::REGISTER, xmm_idx};
}

void AllocState::record_spill_away_register(size_t xmm_idx, std::optional<size_t> stack_idx) {
  if (xmm_usages_[xmm_idx] == std::nullopt) {
    throw std::runtime_error("xmm_idx is not used");
  }
  if (stack_idx == std::nullopt) {
    auto it = std::find(stack_usages_.begin(), stack_usages_.end(), std::nullopt);
    if (it == stack_usages_.end()) {
      throw std::runtime_error("no stack is available");
    }
    stack_idx = std::distance(stack_usages_.begin(), it);
  }
  // update history
  auto&& loc_src = Location{LocationType::REGISTER, xmm_idx};
  auto&& loc_dest = Location{LocationType::STACK, *stack_idx};
  Transition&& transition = std::make_tuple(*xmm_usages_[xmm_idx], loc_src, loc_dest);
  history_[*t_].emplace_back(transition);

  // update current state
  HashType hash_id = *xmm_usages_[xmm_idx];
  locations_[hash_id] = Location{LocationType::STACK, *stack_idx};

  stack_usages_[*stack_idx] = xmm_usages_[xmm_idx];
  xmm_usages_[xmm_idx] = std::nullopt;
  xmm_ages_[xmm_idx] = std::nullopt;

  // increment max stack usage
  if ((*stack_idx + 1) > max_stack_usage_) {
    max_stack_usage_ = *stack_idx + 1;
  }
}

void AllocState::record_load_to_register(size_t stack_idx, size_t xmm_idx) {
  if (stack_usages_[stack_idx] == std::nullopt) {
    throw std::runtime_error("anyting is not stored in the stack");
  }
  // update history
  auto&& loc_src = Location{LocationType::STACK, stack_idx};
  auto&& loc_dest = Location{LocationType::REGISTER, xmm_idx};
  Transition&& transition = std::make_tuple(*stack_usages_[stack_idx], loc_src, loc_dest);
  history_[*t_].emplace_back(transition);

  // update current state
  HashType hash_id = *stack_usages_[stack_idx];
  locations_[hash_id] = Location{LocationType::REGISTER, xmm_idx};
  xmm_usages_[xmm_idx] = stack_usages_[stack_idx];
  stack_usages_[stack_idx] = std::nullopt;
  xmm_ages_[xmm_idx] = 0;
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

void AllocState::print_history() const {
  size_t total_spill = 0;
  for (size_t i_step = 0; i_step < history_.size(); ++i_step) {
    std::cout << std::format("step {}", i_step) << std::endl;
    const auto& transitions = history_[i_step];
    for (const auto& transition : transitions) {
      const HashType& hash_id = std::get<0>(transition);
      const std::optional<Location>& source = std::get<1>(transition);
      const Location& dest = std::get<2>(transition);
      std::cout << std::format("{}: ", hash_id);
      if (source == std::nullopt) {
        std::cout << "OpResult";
      } else {
        std::cout << *source;
      }
      std::cout << " -> " << dest << std::endl;
      if (dest.type == LocationType::STACK) {
        ++total_spill;
      }
    }
  }
  std::cout << std::format("total spill: {}", total_spill) << std::endl;
  std::cout << std::format("max stack usage: {}", max_stack_usage_) << std::endl;
};

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
  auto alloc_state = AllocState(opseq, inputs, 16);

  for (size_t t = 0; t < opseq.size(); ++t) {
    alloc_state.step();
    auto& op = opseq[t];
    if (op->args.size() == 0) {
      if (op->kind != OpKind::VALIABLE) {
        throw std::runtime_error("kind is not VALIABLE");
      }
      size_t dest_xmm_idx = get_available_xmm(alloc_state);
      alloc_state.record_load_from_input(op->hash_id, dest_xmm_idx);
    } else {
      // operand register preparation
      for (auto& operand : op->args) {
        load_to_xmm(alloc_state, operand->hash_id);
      }

      // result register preparation
      std::optional<size_t> result_xmm_idx = std::nullopt;
      auto& disappear_hashids = disappear_table[t];
      for (auto& operand : op->args) {
        auto it = disappear_hashids.find(operand->hash_id);
        if (it != disappear_hashids.end()) {  // if the operand is going to disappear
          result_xmm_idx = load_to_xmm(alloc_state, operand->hash_id);
          break;
        }
      }
      if (result_xmm_idx == std::nullopt) {
        result_xmm_idx = get_available_xmm(alloc_state);
      }
      alloc_state.recored_xmm_assigned_as_op_result(op->hash_id, *result_xmm_idx);
    }
  }
  alloc_state.print_history();
  return alloc_state.get_history();
}

size_t RegisterAllocator::load_to_xmm(AllocState& as, HashType hash_id) {
  const auto& locations = as.get_locations();
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
  if (as.get_xmm_usages()[xmm_idx] != std::nullopt) {
    as.record_spill_away_register(xmm_idx, std::nullopt);
    as.record_load_to_register(loc.idx, xmm_idx);
  }
  return xmm_idx;
}

size_t RegisterAllocator::get_available_xmm(AllocState& as) {
  auto xmm_idx_cand = as.get_available_xmm();
  if (xmm_idx_cand == std::nullopt) {
    size_t xmm_idx = as.most_unused_xmm();
    as.record_spill_away_register(xmm_idx, std::nullopt);
    return xmm_idx;
  }
  return *xmm_idx_cand;
}

}  // namespace register_alloc
}  // namespace tenkai
