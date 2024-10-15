#include "compile.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_set>
#include "cg.hpp"

namespace tenkai {

template <typename T>
std::optional<size_t> find_index(T elem, const std::vector<T>& vec) {
  auto it = std::find(vec.begin(), vec.end(), elem);
  if (it == vec.end()) {
    return std::nullopt;
  }
  return std::distance(vec.begin(), it);
}

// this value must be multiply of page size (4096)?
// must be shared both xbyak constructor and mmap (why? really)
constexpr size_t max_code_size = 4096 * 8;

Compiler::Compiler(const std::vector<Operation::Ptr>& inputs,
                   const std::vector<Operation::Ptr>& outputs,
                   bool avx512)
    : inputs_(inputs),
      outputs_(outputs),
      operations_(flatten(inputs, outputs)),
      disappear_hashid_table_(compute_disappear_hashid_table(operations_)),
      xmm_usage_((avx512) ? 31 : 15, std::nullopt),
      xmm_survival_period_(xmm_usage_.size(), std::nullopt),
      stack_usage_(1024, std::nullopt),
      ophash_to_location_() {}

std::vector<Operation::Ptr> Compiler::flatten(const std::vector<Operation::Ptr>& inputs,
                                              const std::vector<Operation::Ptr>& outputs) {
  std::vector<Operation::Ptr> operations;
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

  // do CSE
  std::unordered_set<int32_t> visited;
  std::vector<Operation::Ptr> result;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    if (visited.find((*it)->hash_id) != visited.end()) {
      continue;
    }
    visited.insert((*it)->hash_id);
    result.push_back(*it);
  }
  // std::cout << "flatten operations:\n";
  // for(size_t t = 0; t <result.size(); ++t){
  //   auto op = result[t];
  //   std::cout << std::format("  t: {}, hashid: {}, kind: {}", t, op->hash_id,
  //   to_string(op->kind)); if (op->args.size() > 0) {
  //     std::cout << ", args: ";
  //     for(auto& arg : op->args) {
  //       std::cout << std::format("{}, ", arg->hash_id);
  //     }
  //   }
  //   std::cout << "\n";
  // }
  return result;
}

std::vector<std::vector<int32_t>> Compiler::compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& operations) {
  std::vector<std::vector<int32_t>> table(operations.size());
  std::unordered_set<int32_t> visited;
  for (int i = operations.size() - 1; i >= 0; --i) {
    for (auto& arg_op : operations[i]->args) {
      bool unappended = visited.find(arg_op->hash_id) == visited.end();
      if (unappended) {
        table[i].push_back(arg_op->hash_id);
        visited.insert(arg_op->hash_id);
      }
    }
  }
  // std::cout << "disappear hashid table:\n";
  // for (size_t i = 0; i < table.size(); ++i) {
  //   std::cout << std::format("  t={}: ", i);
  //   for (auto hash_id : table[i]) {
  //     std::cout << std::format("{}, ", hash_id);
  //   }
  //   std::cout << "\n";
  // }
  return table;
}

std::vector<uint8_t> Compiler::generate_code() {
  double (*sin_ptr)(double) = std::sin;
  void* sin_vptr = reinterpret_cast<void*>(sin_ptr);
  double (*cos_ptr)(double) = std::cos;
  void* cos_vptr = reinterpret_cast<void*>(cos_ptr);

  auto gen = Xbyak::CodeGenerator(max_code_size);
  gen.endbr64();
  gen.push(gen.r12);
  gen.push(gen.r13);
  gen.push(gen.rbp);
  gen.mov(gen.rbp, gen.rsp);
  size_t sub_size = 8 * stack_usage_.size() + 8;  // +8 for x86_64 ABI (16 byte alignment)
  gen.sub(gen.rsp, sub_size);
  gen.mov(gen.r12, gen.rdi);
  gen.mov(gen.r13, gen.rsi);

  for (size_t t = 0; t < operations_.size(); ++t) {
    const auto& op = operations_[t];
    const auto& location = ophash_to_location_[op->hash_id];
    auto available_xmm_idx = get_available_xmm_index(gen);
    auto xmm_result = Xbyak::Xmm(available_xmm_idx);
    xmm_survival_period_[available_xmm_idx] = 0;

    // update register/memory state
    ophash_to_location_[op->hash_id] = Location{available_xmm_idx, true};
    xmm_usage_[available_xmm_idx] = op->hash_id;

    if (op->kind == OpKind::VALIABLE) {
      auto input_idx = find_index(op, inputs_);
      gen.vmovsd(xmm_result, gen.ptr[gen.r12 + input_idx.value() * 8]);
    } else {
      if (op->args.size() == 2) {
        auto xmm_idx_left = get_xmm_register_idx(op->args[0]->hash_id, {available_xmm_idx}, gen);
        auto xmm_idx_right =
            get_xmm_register_idx(op->args[1]->hash_id, {available_xmm_idx, xmm_idx_left}, gen);
        if (op->kind == OpKind::ADD) {
          gen.vaddsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
        } else if (op->kind == OpKind::MUL) {
          gen.vmulsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
        } else if (op->kind == OpKind::SUB) {
          gen.vsubsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
        } else {
          throw std::runtime_error("unsupported operation");
        }
      } else if (op->args.size() == 1) {
        // external call
        // move all
        move_to_xmm0(op->args[0]->hash_id, gen);
        for (size_t i = 1; i < xmm_usage_.size(); ++i) {
          if (xmm_usage_[i].has_value()) {
            bool is_result_xmm = *xmm_usage_[i] == op->hash_id;
            if (!is_result_xmm) {
              spill_register(i, gen);
            }
          }
        }
        if (op->kind == OpKind::SIN) {
          gen.call(sin_vptr);
        } else if (op->kind == OpKind::COS) {
          gen.call(cos_vptr);
        }
        gen.vmovsd(xmm_result, Xbyak::Xmm(0));
      }
    }

    // if op is output, store the result to memory
    auto output_idx = find_index(op, outputs_);
    if (output_idx.has_value()) {
      gen.vmovsd(gen.ptr[gen.r13 + output_idx.value() * 8], xmm_result);
    }
    untrack_disappear_hashid(t);
    update_xmm_suvival_period();
  }

  gen.mov(gen.rsp, gen.rbp);
  gen.pop(gen.rbp);
  gen.pop(gen.r13);
  gen.pop(gen.r12);
  gen.ret();

  auto code = std::vector<uint8_t>(gen.getSize());
  std::copy(gen.getCode(), gen.getCode() + gen.getSize(), code.begin());
  return code;
}

JitFunc<double> Compiler::compile() {
  auto code = generate_code();
  void* mem = mmap(NULL, max_code_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  uint8_t* instruction = static_cast<uint8_t*>(mem);
  std::memcpy(instruction, code.data(), code.size());
  mprotect(mem, max_code_size, PROT_READ | PROT_EXEC) == -1;
  auto add_func = reinterpret_cast<JitFunc<double>>(instruction);
  return add_func;
}

void Compiler::move_to_xmm0(int32_t hash_id, Xbyak::CodeGenerator& gen) {
  auto location = ophash_to_location_[hash_id];
  if (!location.has_value()) {
    throw std::runtime_error("location not found. cannot move to xmm0");
  }
  if (location->is_xmm) {
    if (location->idx == 0) {
      return;  // already in xmm0
    }
    gen.vmovsd(Xbyak::Xmm(0), Xbyak::Xmm(location->idx));
  } else {
    gen.vmovsd(Xbyak::Xmm(0), gen.ptr[gen.rbp - (location->idx + 1) * 8]);
  }
}

uint8_t Compiler::get_xmm_register_idx(int32_t hash_id,
                                       std::vector<uint32_t>&& dont_spill_xmm,
                                       Xbyak::CodeGenerator& gen) {
  auto location = ophash_to_location_[hash_id];
  if (!location.has_value()) {
    throw std::runtime_error(
        std::format("{} location not found. cannot get xmm register", hash_id));
  }
  if (location->is_xmm) {
    return location->idx;
  }
  auto xmm_spill = find_most_unused_xmm_idx(std::move(dont_spill_xmm));
  spill_register(xmm_spill, gen);
  gen.vmovsd(Xbyak::Xmm(xmm_spill), gen.ptr[gen.rbp - (location->idx + 1) * 8]);
  return xmm_spill;
}

void Compiler::untrack_disappear_hashid(size_t t) {
  for (int32_t hash_id : disappear_hashid_table_[t]) {
    auto location = ophash_to_location_[hash_id];
    if (location.has_value() && location->is_xmm) {
      xmm_usage_[location->idx] = std::nullopt;
      xmm_survival_period_[location->idx] = std::nullopt;
    } else {
      if (!location.has_value()) {
        throw std::runtime_error("location not found. cannot untrack disappear hashid");
      }
      stack_usage_[location->idx] = std::nullopt;
    }
    ophash_to_location_[hash_id] = std::nullopt;
  }
}

void Compiler::update_xmm_suvival_period() {
  for (size_t i = 0; i < xmm_survival_period_.size(); ++i) {
    if (xmm_survival_period_[i].has_value()) {
      xmm_survival_period_[i] = *xmm_survival_period_[i] + 1;
    }
  }
}

size_t Compiler::find_most_unused_xmm_idx(std::vector<uint32_t>&& exclude) {
  size_t most_unused_idx = 0;
  size_t most_unused_period = 0;
  for (size_t i = 0; i < xmm_survival_period_.size(); ++i) {
    if (xmm_survival_period_[i].has_value() && *xmm_survival_period_[i] > most_unused_period) {
      bool is_excluded = std::find(exclude.begin(), exclude.end(), i) != exclude.end();
      if (is_excluded) {
        continue;
      }
      most_unused_idx = i;
      most_unused_period = *xmm_survival_period_[i];
    }
  }
  return most_unused_idx;
}

uint8_t Compiler::get_avaialble_stack_index() {
  for (size_t i = 0; i < stack_usage_.size(); ++i) {
    if (stack_usage_[i] == std::nullopt) {
      return i;
    }
  }
  throw std::runtime_error("stack is full. this should not happen");
}

uint8_t Compiler::get_available_xmm_index(Xbyak::CodeGenerator& gen) {
  for (size_t i = 1; i < xmm_usage_.size(); ++i) {
    if (xmm_usage_[i] == std::nullopt) {
      return i;
    }
  }
  // spill register
  auto xmm_idx = find_most_unused_xmm_idx();
  spill_register(xmm_idx, gen);
  return xmm_idx;
}

void Compiler::spill_register(uint8_t xmm_idx, Xbyak::CodeGenerator& gen) {
  auto hash_id = *xmm_usage_[xmm_idx];
  auto stack_idx = get_avaialble_stack_index();
  gen.vmovsd(gen.ptr[gen.rbp - (stack_idx + 1) * 8], Xbyak::Xmm(xmm_idx));

  // update internal state
  xmm_usage_[xmm_idx] = std::nullopt;
  xmm_survival_period_[xmm_idx] = std::nullopt;
  stack_usage_[stack_idx] = hash_id;
  auto new_loc = Location{stack_idx, false};
  ophash_to_location_[hash_id] = new_loc;
}

}  // namespace tenkai
