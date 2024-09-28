#include <unordered_map>
#include "cg.hpp"
#include "iostream"
#include "stack"

void unroll(const std::string& func_name,
            const std::vector<Operation::Ptr>& inputs,
            const std::vector<Operation::Ptr>& outputs) {
  std::vector<Operation::Ptr> operations;
  std::stack<Operation::Ptr> opstack;
  for (auto& output : outputs) {
    opstack.push(output);
  }
  while (!opstack.empty()) {
    auto op = opstack.top();
    opstack.pop();
    if (!op->is_nullaryop()) {
      operations.push_back(op);

      if (op->lhs != nullptr) {
        opstack.push(op->lhs);
      }
      if (op->rhs != nullptr) {
        opstack.push(op->rhs);
      }
    }
  }

  auto remapped_name = [&](const Operation::Ptr op) -> std::string {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == op) {
        return "input[" + std::to_string(i) + "]";
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i] == op) {
        return "output[" + std::to_string(i) + "]";
      }
    }
    return op->name;
  };

  // code generation
  std::cout << "#include <cmath>" << std::endl;
  std::cout << "template <typename T>" << std::endl;
  std::cout << "void "
            << "unrolled_" << func_name << "(T* input, T* output) {"
            << std::endl;

  std::unordered_map<std::string, bool> is_evaluated;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    auto op = *it;
    if (is_evaluated.find(remapped_name(op)) != is_evaluated.end()) {
      continue;
    }
    if (remapped_name(op).find("output") == std::string::npos) {
      std::cout << "  auto " << remapped_name(op) << " = ";
    } else {
      std::cout << "  " << remapped_name(op) << " = ";
    }
    if (op->is_nullaryop()) {
      throw std::runtime_error("must not reach here");
    } else if (op->is_unaryop()) {
      switch (op->kind) {
        case OpKind::COS:
          std::cout << "cos(" << remapped_name(op->lhs) << ");" << std::endl;
          break;
        case OpKind::SIN:
          std::cout << "sin(" << remapped_name(op->lhs) << ");" << std::endl;
          break;
        case OpKind::NEGATE:
          std::cout << "-" << remapped_name(op->lhs) << ";" << std::endl;
          break;
        default:
          throw std::runtime_error("unknown operator");
      }
    } else {
      std::cout << remapped_name(op->lhs) << " ";
      switch (op->kind) {
        case OpKind::ADD:
          std::cout << "+";
          break;
        case OpKind::SUB:
          std::cout << "-";
          break;
        case OpKind::MUL:
          std::cout << "*";
          break;
        default:
          throw std::runtime_error("unknown operator");
      }
      std::cout << " " << remapped_name(op->rhs) << ";" << std::endl;
    }
    is_evaluated[remapped_name(op)] = true;
  }
  std::cout << "}" << std::endl;
}
