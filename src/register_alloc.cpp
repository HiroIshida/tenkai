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

std::ostream& operator<<(std::ostream& os, const Transition& trans) {
  if (std::holds_alternative<RawTransition>(trans)) {
    const auto& raw_trans = std::get<RawTransition>(trans);
    auto [hash_id, loc_src, loc_dst] = raw_trans;
    os << std::format("Var(id={}): ", hash_id);
    os << loc_dst << " <- " << loc_src << std::endl;
    return os;
  } else if (std::holds_alternative<OpTransition>(trans)) {
    const auto op_trans = std::get<OpTransition>(trans);
    auto [hash_id, xmms_src, loc_dst] = op_trans;
    os << std::format("Var(id={}): ", hash_id);
    os << loc_dst << " <- Operation(";
    if (xmms_src.size() == 0) {
      os << ")" << std::endl;
    } else {
      for (size_t i = 0; i < xmms_src.size() - 1; ++i) {
        os << std::format("xmm({}), ", xmms_src[i]);
      }
      os << std::format("xmm({}))", xmms_src.back()) << std::endl;
    }
    return os;
  } else if (std::holds_alternative<ConstantSubstitution>(trans)) {
    const auto sub_trans = std::get<ConstantSubstitution>(trans);
    auto [hash_id, value, loc_dst] = sub_trans;
    os << std::format("Var(id={}): ", hash_id);
    os << loc_dst << " <- " << value << std::endl;
    return os;
  } else {
    throw std::runtime_error("not implemented");
  }
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

std::vector<TransitionSet> RegisterAllocator::allocate() {
  for (size_t t = 0; t < opseq_.size(); ++t) {
    auto& op = opseq_[t];

    if (op->kind == OpKind::VALIABLE || op->kind == OpKind::CONSTANT) {
      std::variant<double, Location> loc_src;  // double for constant

      if (op->kind == OpKind::VALIABLE) {
        // determine source location
        auto it_inp_idx = std::find_if(
            inputs_.begin(), inputs_.end(),
            [op](const Operation::Ptr& input) { return input->hash_id == op->hash_id; });
        auto inp_idx = std::distance(inputs_.begin(), it_inp_idx);
        loc_src = Location{LocationType::INPUT, inp_idx};
      } else if (op->kind == OpKind::CONSTANT) {
        loc_src = op->constant_value.value();
      }

      // determine destination location
      auto xmm_idx = alloc_state_.get_available_xmm();
      if (xmm_idx == std::nullopt) {
        xmm_idx = spill_and_prepare_xmm();
      }
      Location loc_dst = Location{LocationType::REGISTER, *xmm_idx};

      // update alloc_state_
      alloc_state_.xmm_usages_[*xmm_idx] = op->hash_id;
      alloc_state_.xmm_ages_[*xmm_idx] = 0;
      alloc_state_.locations_[op->hash_id] = loc_dst;

      // record
      if (op->kind == OpKind::VALIABLE) {
        transition_sets_[t].emplace_back(
            RawTransition{op->hash_id, std::get<Location>(loc_src), loc_dst});
      } else if (op->kind == OpKind::CONSTANT) {
        transition_sets_[t].emplace_back(
            ConstantSubstitution{op->hash_id, std::get<double>(loc_src), loc_dst});
      }
    } else if (op->kind == OpKind::SIN ||
               op->kind == OpKind::COS) {  // when calling exeternal functions
      // in this case we must move the operands to xmm0 and stash all the xmm registers to stack
      // and the result will be stored in xmm0
      if (op->args.size() != 1) {
        throw std::runtime_error("SIN or COS must have only one operand");
      }
      const auto& opr_loc_now = alloc_state_.locations_[op->args[0]->hash_id];
      prepare_value_on_xmm(op->args[0]->hash_id, 0);

      // following x86-64 nasm calling convention
      for (size_t i = 0; i < alloc_state_.xmm_usages_.size(); ++i) {
        if (alloc_state_.xmm_usages_[i] != std::nullopt) {
          spill_xmm(i);
        }
      }

      auto loc_dst = Location{LocationType::REGISTER, 0};
      // update alloc_state_
      alloc_state_.xmm_usages_[0] = op->hash_id;
      alloc_state_.xmm_ages_[0] = 0;
      alloc_state_.locations_[op->hash_id] = loc_dst;

      // record
      transition_sets_[t_].emplace_back(OpTransition{op->hash_id, {}, loc_dst});
    } else {
      // set the age of registers whereon the operands are stored to 0
      for (auto& operand : op->args) {
        const auto& op_loc_now = alloc_state_.locations_[operand->hash_id];
        if (op_loc_now.type == LocationType::REGISTER) {
          alloc_state_.xmm_ages_[op_loc_now.idx] = 0;
        }
      }

      for (auto& operand : op->args) {
        const auto& op_loc_now = alloc_state_.locations_[operand->hash_id];
        if (op_loc_now.type != LocationType::REGISTER) {
          std::optional<size_t> xmm_idx = alloc_state_.get_available_xmm();
          if (xmm_idx == std::nullopt) {
            xmm_idx = alloc_state_.most_unused_xmm();
          }
          prepare_value_on_xmm(operand->hash_id, *xmm_idx);
        }
      }

      // now that we know that all operands are on xmm,
      std::vector<size_t> xmms_src;
      for (auto& operand : op->args) {
        const auto& loc = alloc_state_.locations_[operand->hash_id];
        xmms_src.push_back(loc.idx);
      }
      if (op->kind == OpKind::NEGATE) {
        // use special xmm register to store bit mask for negation
        xmms_src.push_back(temp_xmm_idx_);
      }

      // untrack the hashids that will disappear
      // and free the registers and stack locations
      const auto& disappear_hash_ids = disappear_hashid_table_[t];
      for (auto hash_id : disappear_hash_ids) {
        auto& loc = alloc_state_.locations_[hash_id];
        if (loc.type == LocationType::REGISTER) {
          auto xmm_idx = loc.idx;
          alloc_state_.xmm_usages_[xmm_idx].reset();
          alloc_state_.xmm_ages_[xmm_idx].reset();
          alloc_state_.locations_.erase(hash_id);
        } else if (loc.type == LocationType::STACK) {
          auto stack_idx = loc.idx;
          alloc_state_.stack_usages_[stack_idx].reset();
          alloc_state_.locations_.erase(hash_id);
        } else {
        }
      }

      // now allocate the result! (same as above)
      std::optional<size_t> result_xmm_idx = alloc_state_.get_available_xmm();
      if (result_xmm_idx == std::nullopt) {
        result_xmm_idx = spill_and_prepare_xmm();
      }
      Location loc_dst = Location{LocationType::REGISTER, *result_xmm_idx};

      // update alloc_state_
      alloc_state_.xmm_usages_[*result_xmm_idx] = op->hash_id;
      alloc_state_.xmm_ages_[*result_xmm_idx] = 0;
      alloc_state_.locations_[op->hash_id] = loc_dst;

      // record
      transition_sets_[t_].push_back(OpTransition{op->hash_id, xmms_src, loc_dst});

      // if the result will be output, then copy it to the output location
      // find output index
      auto it_out_idx = std::find_if(
          outputs_.begin(), outputs_.end(),
          [op](const Operation::Ptr& output) { return output->hash_id == op->hash_id; });
      if (it_out_idx != outputs_.end()) {
        auto out_idx = std::distance(outputs_.begin(), it_out_idx);
        Location loc_src = loc_dst;
        Location loc_dst{LocationType::OUTPUT, out_idx};
        transition_sets_[t_].emplace_back(RawTransition{op->hash_id, loc_src, loc_dst});
        // bit strange but update alloc_state_ is not needed
        // as it will anyway treated as disappeared in the next step
      }
    }
    step();
  }
  return transition_sets_;
}

void RegisterAllocator::spill_xmm(size_t idx) {
  auto hash_id = alloc_state_.xmm_usages_[idx];
  auto loc_src = alloc_state_.locations_[*hash_id];
  auto stack_idx = alloc_state_.get_available_stack();
  Location loc_dst{LocationType::STACK, stack_idx};

  // update alloc_state_
  alloc_state_.xmm_usages_[idx].reset();
  alloc_state_.xmm_ages_[idx].reset();
  alloc_state_.stack_usages_[stack_idx] = hash_id;
  alloc_state_.locations_[*hash_id] = loc_dst;

  // record
  transition_sets_[t_].emplace_back(RawTransition{*hash_id, loc_src, loc_dst});
}

void RegisterAllocator::prepare_value_on_xmm(HashType hash_id, size_t dst_xmm_idx) {
  // if the target xmm idx is already used by other value,
  // then spill it out to stack always
  if (alloc_state_.xmm_usages_[dst_xmm_idx] == hash_id) {
    return;
  }

  if (alloc_state_.xmm_usages_[dst_xmm_idx] != std::nullopt) {
    spill_xmm(dst_xmm_idx);
  }

  const auto dst = Location{LocationType::REGISTER, dst_xmm_idx};
  const auto src = alloc_state_.locations_[hash_id];

  // update alloc state_
  if (src.type == LocationType::REGISTER) {
    alloc_state_.xmm_usages_[src.idx] = std::nullopt;
    alloc_state_.xmm_ages_[src.idx] = std::nullopt;
  } else if (src.type == LocationType::STACK) {
    alloc_state_.stack_usages_[src.idx] = std::nullopt;
  } else {
    throw std::runtime_error("unexpected location type");
  }
  alloc_state_.xmm_usages_[dst_xmm_idx] = hash_id;
  alloc_state_.xmm_ages_[dst_xmm_idx] = 0;
  alloc_state_.locations_[hash_id] = dst;

  // record
  transition_sets_[t_].emplace_back(RawTransition{hash_id, src, dst});
}

size_t RegisterAllocator::spill_and_prepare_xmm() {
  auto spill_xmm_idx = alloc_state_.most_unused_xmm();
  spill_xmm(spill_xmm_idx);
  return spill_xmm_idx;
}

}  // namespace register_alloc
}  // namespace tenkai
