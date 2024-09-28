#include "cg.hpp"
#include <algorithm>
#include <random>
#include <stack>
#include <stdexcept>

std::string generate_random_string(size_t length) {
  // c++ does not have a built-in random string generator??
  const std::string charset =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
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
  name = generate_random_string(6);
}

Operation::Operation(const std::string& name) : kind(OpKind::NIL), name(name) {}

Operation::Operation(Operation::Ptr lhs, Operation::Ptr rhs, OpKind kind)
    : lhs(lhs), rhs(rhs), kind(kind) {
  name = generate_random_string(6);
}

Operation::Ptr Operation::create(Operation::Ptr lhs,
                                 Operation::Ptr rhs,
                                 OpKind kind) {
  auto op = std::make_shared<Operation>(lhs, rhs, kind);
  lhs->requireds.push_back(op);
  if (rhs != nullptr) {
    rhs->requireds.push_back(op);
  }
  return op;
}

Operation::Ptr Operation::make_value(std::string&& name) {
  Operation::Ptr value = std::make_shared<Operation>(name);
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
      if (!is_added(op->lhs)) {
        stack.push(op->lhs);
      }
      if (!is_added(op->rhs)) {
        stack.push(op->rhs);
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
  return Operation::create(lhs, rhs, OpKind::ADD);
}
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs) {
  if (lhs->kind == OpKind::ZERO) {
    return rhs;
  }
  if (rhs->kind == OpKind::ZERO) {
    return lhs;
  }
  return Operation::create(lhs, rhs, OpKind::SUB);
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
  return Operation::create(lhs, rhs, OpKind::MUL);
}
Operation::Ptr cos(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_one();
  }
  return Operation::create(op, nullptr, OpKind::COS);
}
Operation::Ptr sin(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  return Operation::create(op, nullptr, OpKind::SIN);
}
Operation::Ptr negate(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  return Operation::create(op, nullptr, OpKind::NEGATE);
}
