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
    if (op->kind != OpKind::VALUE) {
      operations.push_back(op);
      opstack.push(op->lhs);
      opstack.push(op->rhs);
    }
  }

  std::unordered_map<std::string, bool> is_evaluated;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    auto op = *it;
    if (is_evaluated.find(op->name) != is_evaluated.end()) {
      continue;
    }
    if (op->kind == OpKind::VALUE) {
      std::cout << "// value " << op->name << std::endl;
      continue;
    }
    std::cout << "double " << op->name << " = " << op->lhs->name << " ";
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
    std::cout << " " << op->rhs->name << ";" << std::endl;
    is_evaluated[op->name] = true;
  }
  std::cout << "return " << name << ";" << std::endl;
}
