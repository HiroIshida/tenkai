#include "cg.hpp"
#include <algorithm>
#include <random>
#include <stack>
#include <stdexcept>

namespace tenkai {

std::string generate_random_string(size_t length) {
  // c++ does not have a built-in random string generator??
  const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, charset.size() - 1);
  std::string result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result += charset[distribution(generator)];
  }
  return result;
}

Operation::Operation() : kind(OpKind::NIL) {
  name = generate_random_string(8);
}

Operation::Operation(const std::string& name) : kind(OpKind::NIL), name(name) {}

Operation::Operation(OpKind kind, std::vector<Operation::Ptr> leafs) : kind(kind), args(leafs) {
  name = generate_random_string(8);
}

Operation::Ptr Operation::create(OpKind kind, std::vector<Operation::Ptr>&& leafs) {
  return std::make_shared<Operation>(kind, leafs);
}

Operation::Ptr Operation::make_var() {
  Operation::Ptr value = std::make_shared<Operation>();
  value->kind = OpKind::VALIABLE;
  return value;
}

Operation::Ptr Operation::make_zero() {
  Operation::Ptr zero = std::make_shared<Operation>("0.0");
  zero->kind = OpKind::ZERO;
  return zero;
}

Operation::Ptr Operation::make_one() {
  Operation::Ptr one = std::make_shared<Operation>("1.0");
  one->kind = OpKind::ONE;
  return one;
}

Operation::Ptr Operation::make_ext_func(const std::string& name, Operation::Ptr arg) {
  Operation::Ptr func = std::make_shared<Operation>(name);
  func->kind = OpKind::NIL;
  func->args.push_back(arg);
  return func;
}

Operation::Ptr Operation::make_constant(double value) {
  Operation::Ptr constant = std::make_shared<Operation>(std::to_string(value));
  constant->kind = OpKind::CONSTANT;
  return constant;
}

std::vector<Operation::Ptr> Operation::get_leafs() {
  std::vector<Operation::Ptr> leafs;
  auto is_added = [&leafs](Operation::Ptr op) {
    return std::find(leafs.begin(), leafs.end(), op) != leafs.end();
  };
  std::stack<Operation::Ptr> stack;
  stack.push(shared_from_this());
  while (!stack.empty()) {
    auto op = stack.top();
    stack.pop();
    if (op->kind == OpKind::VALIABLE) {
      leafs.push_back(op);
    } else {
      for (auto& arg : op->args) {
        if (!is_added(arg)) {
          stack.push(arg);
        }
      }
    }
  }
  return leafs;
}

Operation::Ptr operator+(Operation::Ptr lhs, Operation::Ptr rhs) {
  if (lhs->kind == OpKind::ZERO) {
    return rhs;
  }
  if (rhs->kind == OpKind::ZERO) {
    return lhs;
  }
  if (rhs->kind == OpKind::CONSTANT && lhs->kind == OpKind::CONSTANT) {
    return Operation::make_constant(std::stod(lhs->name) + std::stod(rhs->name));
  }
  return Operation::create(OpKind::ADD, {lhs, rhs});
}
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs) {
  if (lhs->kind == OpKind::ZERO) {
    return rhs;
  }
  if (rhs->kind == OpKind::ZERO) {
    return lhs;
  }
  if (rhs->kind == OpKind::CONSTANT && lhs->kind == OpKind::CONSTANT) {
    return Operation::make_constant(std::stod(lhs->name) - std::stod(rhs->name));
  }
  return Operation::create(OpKind::SUB, {lhs, rhs});
}
Operation::Ptr operator*(Operation::Ptr lhs, Operation::Ptr rhs) {
  if (lhs->kind == OpKind::ZERO || rhs->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  if (lhs->kind == OpKind::ONE) {
    return rhs;
  }
  if (rhs->kind == OpKind::ONE) {
    return lhs;
  }
  if (rhs->kind == OpKind::CONSTANT && lhs->kind == OpKind::CONSTANT) {
    return Operation::make_constant(std::stod(lhs->name) * std::stod(rhs->name));
  }
  return Operation::create(OpKind::MUL, {lhs, rhs});
}
Operation::Ptr cos(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_one();
  }
  return Operation::create(OpKind::COS, {op});
}
Operation::Ptr sin(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  return Operation::create(OpKind::SIN, {op});
}
Operation::Ptr operator-(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  return Operation::create(OpKind::NEGATE, {op});
}

};  // namespace tenkai
