#include <unordered_map>
#include "cg.hpp"
#include "iostream"
#include "stack"

void Operation::unroll(const std::vector<std::string>& input_args) {
  std::vector<Operation::Ptr> operations;
  std::stack<Operation::Ptr> opstack;
  opstack.push(shared_from_this());
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

  auto remap = [&](const std::string& name) {
    for (int i = 0; i < input_args.size(); i++) {
      if (input_args[i] == name) {
        return "input[" + std::to_string(i) + "]";
      }
    }
    return name;
  };

  std::cout << "#include <cmath>" << std::endl;
  std::cout << "template <typename T>" << std::endl;
  std::cout << "T "
            << "unrolled_" << name << "(T* input) {" << std::endl;

  std::unordered_map<std::string, bool> is_evaluated;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    auto op = *it;
    if (is_evaluated.find(remap(op->name)) != is_evaluated.end()) {
      continue;
    }
    std::cout << "  auto " << remap(op->name) << " = ";
    if (op->is_nullaryop()) {
      throw std::runtime_error("must not reach here");
    } else if (op->is_unaryop()) {
      switch (op->kind) {
        case OpKind::COS:
          std::cout << "cos(" << remap(op->lhs->name) << ");" << std::endl;
          break;
        case OpKind::SIN:
          std::cout << "sin(" << remap(op->lhs->name) << ");" << std::endl;
          break;
        case OpKind::NEGATE:
          std::cout << "-" << remap(op->lhs->name) << ";" << std::endl;
          break;
        default:
          throw std::runtime_error("unknown operator");
      }
    } else {
      std::cout << remap(op->lhs->name) << " ";
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
      std::cout << " " << remap(op->rhs->name) << ";" << std::endl;
    }
    is_evaluated[remap(op->name)] = true;
  }
  std::cout << "  return " << name << ";" << std::endl;
  std::cout << "}" << std::endl;
}
