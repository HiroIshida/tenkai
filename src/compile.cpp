#include "compile.hpp"
#include <algorithm>
#include <format>
#include <iostream>
#include <stack>
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

struct Compiler : Xbyak::CodeGenerator {
  Compiler(const std::vector<Operation::Ptr>& inputs, const std::vector<Operation::Ptr>& outputs) {
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

    size_t n_available_xmm = 32;
    std::vector<Operation::Ptr> op_to_xmm(n_available_xmm, nullptr);

    auto get_xmm = [&](Operation::Ptr op) -> size_t {
      for (size_t i = 0; i < n_available_xmm; ++i) {
        if (op_to_xmm[i] == op) {
          return i;
        }
      }
      throw std::runtime_error("no xmm register found");
    };

    auto get_available_xmm_idx = [&]() -> size_t {
      for (size_t i = 0; i < n_available_xmm; ++i) {
        if (op_to_xmm[i] == nullptr) {
          return i;
        }
      }
      throw std::runtime_error("no available xmm register");
    };

    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
      auto op = *it;
      if (op->kind == OpKind::VALIABLE) {
        size_t input_idx = *find_index(op, inputs);
        size_t available_xmm_idx = get_available_xmm_idx();
        vmovsd(Xbyak::Xmm(available_xmm_idx), ptr[rdi + input_idx * 8]);
        // std::cout << std::format("vmovsd xmm{}, [rdi + {} * 8]\n", available_xmm_idx, input_idx);
        op_to_xmm[available_xmm_idx] = op;
      } else {
        auto output_idx = find_index(op, outputs);
        bool is_output = output_idx.has_value();

        if (op->kind == OpKind::ADD) {
          size_t left_xmm = get_xmm(op->args[0]);
          size_t right_xmm = get_xmm(op->args[1]);

          size_t new_xmm = get_available_xmm_idx();
          vaddsd(Xbyak::Xmm(new_xmm), Xbyak::Xmm(left_xmm), Xbyak::Xmm(right_xmm));
          // std::cout << std::format("vaddsd xmm{}, xmm{}, xmm{}\n", new_xmm, left_xmm, right_xmm);
          op_to_xmm[new_xmm] = op;

          if (is_output) {
            vmovsd(ptr[rsi + output_idx.value() * 8], Xbyak::Xmm(new_xmm));
            // std::cout << std::format("movsd [rsi + {} * 8], xmm{}\n", output_idx.value(),
            // new_xmm);
          }
        }
      }
    }
    ret();
    // std::cout << "ret" << std::endl;
  }
};

std::vector<uint8_t> compile(const std::vector<Operation::Ptr>& inputs,
                             const std::vector<Operation::Ptr>& outputs) {
  Compiler c(inputs, outputs);
  const uint8_t* top_addr = c.getCode();
  const uint8_t* bottom_addr = top_addr + c.getSize();

  auto code = std::vector<uint8_t>(c.getSize());
  std::copy(top_addr, bottom_addr, code.begin());
  return code;
}
}  // namespace tenkai
