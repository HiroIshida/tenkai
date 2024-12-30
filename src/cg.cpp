#include "cg.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stack>
#include <stdexcept>
#include <string>

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

int32_t djb2_hash(const std::string& str) {
  unsigned long hash = 5381;
  for (char c : str) {
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}

Operation::Operation() : kind(OpKind::NIL) {
  hash_id = generate_random_int();
}

Operation::Operation(OpKind kind, std::vector<Operation::Ptr> leafs, int32_t hash_id)
    : kind(kind), args(leafs), hash_id(hash_id) {}

Operation::Ptr Operation::create(OpKind kind,
                                 std::vector<Operation::Ptr>&& leafs,
                                 int32_t hash_id) {
  auto created = std::make_shared<Operation>(kind, leafs, hash_id);
  for (auto& leaf : leafs) {
    leaf->callers.push_back(created);
  }
  return created;
}

Operation::Ptr Operation::make_var() {
  Operation::Ptr value = std::make_shared<Operation>();
  value->kind = OpKind::LOAD;
  return value;
}

Operation::Ptr Operation::make_zero() {
  Operation::Ptr zero = std::make_shared<Operation>();
  zero->kind = OpKind::ZERO;
  zero->constant_value = 0.0;
  return zero;
}

Operation::Ptr Operation::make_one() {
  Operation::Ptr one = std::make_shared<Operation>();
  one->kind = OpKind::ONE;
  one->constant_value = 1.0;
  return one;
}

Operation::Ptr Operation::make_ext_func(std::string&& name, std::vector<Operation::Ptr>&& args) {
  Operation::Ptr func = std::make_shared<Operation>();
  func->kind = OpKind::EXTCALL;
  func->args = std::move(args);
  func->ext_func_name = std::move(name);
  return func;
}

Operation::Ptr Operation::make_constant(double value) {
  Operation::Ptr constant = std::make_shared<Operation>();
  constant->kind = OpKind::CONSTANT;
  constant->constant_value = value;
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
    if (op->kind == OpKind::LOAD) {
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
    return Operation::make_constant(*lhs->constant_value + *rhs->constant_value);
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
    return Operation::make_constant(*lhs->constant_value - *rhs->constant_value);
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
    return Operation::make_constant(*lhs->constant_value * *rhs->constant_value);
  }
  auto this_hash_id = division_hash(lhs->hash_id * rhs->hash_id);
  return Operation::create(OpKind::MUL, {lhs, rhs}, this_hash_id);
}
Operation::Ptr cos(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_one();
  }
  auto tmp = "(cos)" + std::to_string(op->hash_id);
  auto this_hash_id = djb2_hash(tmp);
  return Operation::create(OpKind::COS, {op}, this_hash_id);
}
Operation::Ptr sin(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  auto tmp = "(sin)" + std::to_string(op->hash_id);
  auto this_hash_id = djb2_hash(tmp);
  return Operation::create(OpKind::SIN, {op}, this_hash_id);
}
Operation::Ptr operator-(Operation::Ptr op) {
  if (op->kind == OpKind::ZERO) {
    return Operation::make_zero();
  }
  auto this_hash_id = division_hash(-op->hash_id);
  return Operation::create(OpKind::NEGATE, {op}, this_hash_id);
}

};  // namespace tenkai
