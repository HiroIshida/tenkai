#pragma once

#include <memory>
#include <string>
#include <vector>

std::string generate_random_string(size_t length);

enum class OpKind { NIL, ADD, SUB, MUL, COS, SIN, VALIABLE, ZERO, ONE };

struct Operation : std::enable_shared_from_this<Operation> {
  using Ptr = std::shared_ptr<Operation>;
  using WeakPtr = std::weak_ptr<Operation>;
  Operation();
  Operation(const std::string& name);
  Operation(Operation::Ptr lhs, Operation::Ptr rhs, OpKind kind);
  static Operation::Ptr create(Operation::Ptr lhs,
                               Operation::Ptr rhs,
                               OpKind kind);
  static Operation::Ptr make_value(std::string&& name);
  static Operation::Ptr make_zero();
  static Operation::Ptr make_one();
  std::vector<Operation::Ptr> get_leafs();
  void unroll(const std::vector<std::string>& arg_names);
  inline bool is_nullaryop() const { return lhs == nullptr && rhs == nullptr; }
  inline bool is_unaryop() const { return lhs != nullptr && rhs == nullptr; }

  // member variables
  OpKind kind;
  Operation::Ptr lhs;
  Operation::Ptr rhs;
  std::string name;
  std::vector<WeakPtr> requireds;
};

Operation::Ptr operator+(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator*(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr cos(Operation::Ptr op);  // rhs is nullptr
Operation::Ptr sin(Operation::Ptr op);  // rhs is nullptr
