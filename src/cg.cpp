#include "cg.hpp"
#include <algorithm>
#include <cmath>
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

int32_t generate_random_int() {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, 2147483647);
  return distribution(generator);
}

int32_t division_hash(int32_t x) {
  constexpr int32_t prime = 2147483647;
  return x % prime;
}

Operation::Operation() : kind(OpKind::NIL) {
  hash_id = generate_random_int();
  name = generate_random_string(8);
}

Operation::Operation(const std::string& name) : kind(OpKind::NIL), name(name) {
  hash_id = generate_random_int();
}

Operation::Operation(OpKind kind, std::vector<Operation::Ptr> leafs, int32_t hash_id)
    : kind(kind), args(leafs), hash_id(hash_id) {
  name = generate_random_string(8);
}

Operation::Ptr Operation::create(OpKind kind,
                                 std::vector<Operation::Ptr>&& leafs,
                                 int32_t hash_id) {
  return std::make_shared<Operation>(kind, leafs, hash_id);
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

Operation::Ptr Operation::make_ext_func(std::string&& name, std::vector<Operation::Ptr>&& args) {
  Operation::Ptr func = std::make_shared<Operation>(name);
  func->kind = OpKind::EXTCALL;
  func->args = std::move(args);
  func->ext_func_name = std::move(name);
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
  auto this_hash_id = division_hash(lhs->hash_id + rhs->hash_id);
  return Operation::create(OpKind::ADD, {lhs, rhs}, this_hash_id);
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
  auto this_hash_id = division_hash(lhs->hash_id - rhs->hash_id);
  return Operation::create(OpKind::SUB, {lhs, rhs}, this_hash_id);
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
  auto this_hash_id = division_hash(lhs->hash_id * rhs->hash_id);
  return Operation::create(OpKind::MUL, {lhs, rhs}, this_hash_id);
}
Operation::Ptr cos(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_one();
  }
  return Operation::create(OpKind::COS, {op}, 0);
}
Operation::Ptr sin(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  return Operation::create(OpKind::SIN, {op}, 0);
}
Operation::Ptr operator-(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  auto this_hash_id = division_hash(-op->hash_id);
  return Operation::create(OpKind::NEGATE, {op}, this_hash_id);
}

};  // namespace tenkai
