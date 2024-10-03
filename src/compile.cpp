#include "compile.hpp"
#include <algorithm>
#include <format>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include "cg.hpp"
#include "xbyak.h"

namespace tenkai {

template <typename T>
std::optional<size_t> find_index(T elem, const std::vector<T>& vec) {
  auto it = std::find(vec.begin(), vec.end(), elem);
  if (it == vec.end()) {
    return std::nullopt;
  }
  return std::distance(vec.begin(), it);
}

class Compiler {
 public:
  struct Location {
    size_t idx;
    bool is_xmm;
  };

  Compiler(const std::vector<Operation::Ptr>& inputs,
           const std::vector<Operation::Ptr>& outputs,
           bool avx512) {
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
    inputs_ = inputs;
    outputs_ = outputs;
    operations_ = std::vector<Operation::Ptr>(operations.rbegin(), operations.rend());
    xmm_usage_ = std::vector<std::optional<int32_t>>((avx512) ? 31 : 15, std::nullopt);
    xmm_survival_period_ = std::vector<std::optional<int32_t>>(xmm_usage_.size(), std::nullopt);

    // auto stack_size = std::max(operations.size() - xmm_usage_.size(), static_cast<size_t>(0));
    size_t stack_size = 100;  // for now
    stack_usage_ = std::vector<std::optional<int32_t>>(stack_size, std::nullopt);
    ophash_to_location_ = std::unordered_map<int32_t, std::optional<Location>>();
  }

  std::vector<uint8_t> generate_code() {
    auto gen = Xbyak::CodeGenerator();
    gen.endbr64();
    std::cout << "endbr64" << std::endl;
    gen.push(gen.rbp);
    std::cout << "push rbp" << std::endl;
    gen.mov(gen.rbp, gen.rsp);
    std::cout << "mov rbp, rsp" << std::endl;
    gen.sub(gen.rsp, 8 * stack_usage_.size());
    std::cout << std::format("sub rsp, {}", 8 * stack_usage_.size()) << std::endl;
    // start
    for (auto& op : operations_) {
      const auto& location = ophash_to_location_[op->hash_id];
      if (location.has_value()) {  // cached
        continue;
      }
      auto available_xmm_idx = get_available_xmm_index(gen);
      auto xmm_result = Xbyak::Xmm(available_xmm_idx);
      xmm_survival_period_[available_xmm_idx] = 0;

      // update register/memory state
      ophash_to_location_[op->hash_id] = Location{available_xmm_idx, true};
      xmm_usage_[available_xmm_idx] = op->hash_id;

      if (op->kind == OpKind::VALIABLE) {
        auto input_idx = find_index(op, inputs_);
        gen.vmovsd(xmm_result, gen.ptr[gen.rdi + input_idx.value() * 8]);
        std::cout << std::format("vmovsd xmm{}, [rdi + {} * 8]", available_xmm_idx,
                                 input_idx.value())
                  << std::endl;
      } else {
        if (op->args.size() == 2) {
          auto xmm_idx_left = get_xmm_register_idx(op->args[0]->hash_id, {available_xmm_idx}, gen);
          auto xmm_idx_right =
              get_xmm_register_idx(op->args[1]->hash_id, {available_xmm_idx, xmm_idx_left}, gen);
          if (op->kind == OpKind::ADD) {
            gen.vaddsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
            std::cout << std::format("vaddsd xmm{}, xmm{}, xmm{}", available_xmm_idx, xmm_idx_left,
                                     xmm_idx_right)
                      << std::endl;
          } else if (op->kind == OpKind::MUL) {
            gen.vmulsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
            std::cout << std::format("vmulsd xmm{}, xmm{}, xmm{}", available_xmm_idx, xmm_idx_left,
                                     xmm_idx_right)
                      << std::endl;
          } else if (op->kind == OpKind::SUB) {
            gen.vsubsd(xmm_result, Xbyak::Xmm(xmm_idx_left), Xbyak::Xmm(xmm_idx_right));
            std::cout << std::format("vsubsd xmm{}, xmm{}, xmm{}", available_xmm_idx, xmm_idx_left,
                                     xmm_idx_right)
                      << std::endl;
          } else {
            throw std::runtime_error("unsupported operation");
          }
        }
      }

      // if op is output, store the result to memory
      auto output_idx = find_index(op, outputs_);
      if (output_idx.has_value()) {
        gen.vmovsd(gen.ptr[gen.rsi + output_idx.value() * 8], xmm_result);
        std::cout << std::format("vmovsd [rsi + {} * 8], xmm{}", output_idx.value(),
                                 available_xmm_idx)
                  << std::endl;
      }

      update_xmm_suvival_period();
    }

    gen.add(gen.rsp, 8 * stack_usage_.size());
    std::cout << std::format("add rsp, {}", 8 * stack_usage_.size()) << std::endl;
    gen.pop(gen.rbp);
    std::cout << "pop rbp" << std::endl;
    gen.ret();
    std::cout << "ret" << std::endl;

    auto code = std::vector<uint8_t>(gen.getSize());
    std::copy(gen.getCode(), gen.getCode() + gen.getSize(), code.begin());
    return code;
  }

 private:
  uint8_t get_xmm_register_idx(int32_t hash_id,
                               std::vector<uint32_t>&& dont_spill_xmm,
                               Xbyak::CodeGenerator& gen) {
    auto location = ophash_to_location_[hash_id];
    if (!location.has_value()) {
      throw std::runtime_error("location not found");
    }
    if (location->is_xmm) {
      return location->idx;
    }
    auto xmm_spill = find_most_unused_xmm_idx(std::move(dont_spill_xmm));
    spill_register(xmm_spill, gen);
    gen.vmovsd(Xbyak::Xmm(xmm_spill), gen.ptr[gen.rbp - (location->idx + 1) * 8]);
    std::cout << std::format("vmovsd xmm{}, [rbp - {} * 8]", xmm_spill, location->idx + 1)
              << std::endl;
    return xmm_spill;
  }

  void update_xmm_suvival_period() {
    for (size_t i = 0; i < xmm_survival_period_.size(); ++i) {
      if (xmm_survival_period_[i].has_value()) {
        xmm_survival_period_[i] = *xmm_survival_period_[i] + 1;
      }
    }
  }

  size_t find_most_unused_xmm_idx(std::vector<uint32_t>&& exclude = {}) {
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

  uint8_t get_avaialble_stack_index() {
    for (size_t i = 0; i < stack_usage_.size(); ++i) {
      if (stack_usage_[i] == std::nullopt) {
        return i;
      }
    }
    throw std::runtime_error("stack is full. this should not happen");
  }

  uint8_t get_available_xmm_index(Xbyak::CodeGenerator& gen) {
    for (size_t i = 0; i < xmm_usage_.size(); ++i) {
      if (xmm_usage_[i] == std::nullopt) {
        return i;
      }
    }
    // spill register
    auto xmm_idx = find_most_unused_xmm_idx();
    spill_register(xmm_idx, gen);
    return xmm_idx;
  }

  void spill_register(uint8_t xmm_idx, Xbyak::CodeGenerator& gen) {
    auto hash_id = *xmm_usage_[xmm_idx];
    auto stack_idx = get_avaialble_stack_index();
    gen.vmovsd(gen.ptr[gen.rbp - (stack_idx + 1) * 8], Xbyak::Xmm(xmm_idx));
    std::cout << std::format("vmovsd [rbp - {} * 8], xmm{}", stack_idx + 1, xmm_idx) << std::endl;

    // update internal state
    xmm_usage_[xmm_idx] = std::nullopt;
    stack_usage_[stack_idx] = hash_id;
    auto new_loc = Location{stack_idx, false};
    ophash_to_location_[hash_id] = new_loc;
  }

  std::vector<Operation::Ptr> inputs_;
  std::vector<Operation::Ptr> outputs_;
  std::vector<Operation::Ptr> operations_;
  std::vector<std::optional<int32_t>> xmm_usage_;
  std::vector<std::optional<int32_t>> xmm_survival_period_;
  std::vector<std::optional<int32_t>> stack_usage_;
  std::unordered_map<int32_t, std::optional<Location>> ophash_to_location_;
};

std::vector<uint8_t> compile(const std::vector<Operation::Ptr>& inputs,
                             const std::vector<Operation::Ptr>& outputs) {
  Compiler c(inputs, outputs, false);
  return c.generate_code();
}
}  // namespace tenkai
