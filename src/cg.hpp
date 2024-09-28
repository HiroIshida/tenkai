#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

std::string generate_random_string(size_t length);

enum class OpKind { NIL, ADD, SUB, MUL, COS, SIN, NEGATE, VALIABLE, ZERO, ONE };

struct Operation : std::enable_shared_from_this<Operation> {
  using Ptr = std::shared_ptr<Operation>;
  using WeakPtr = std::weak_ptr<Operation>;
  Operation();
  Operation(const std::string& name);
  Operation(Operation::Ptr lhs, Operation::Ptr rhs, OpKind kind);
  static Operation::Ptr create(Operation::Ptr lhs,
                               Operation::Ptr rhs,
                               OpKind kind);
  static Operation::Ptr make_var(std::string&& name);
  static Operation::Ptr make_zero();
  static Operation::Ptr make_one();
  std::vector<Operation::Ptr> get_leafs();
  inline bool is_nullaryop() const { return lhs == nullptr && rhs == nullptr; }
  inline bool is_unaryop() const { return lhs != nullptr && rhs == nullptr; }

  // member variables
  OpKind kind;
  Operation::Ptr lhs;
  Operation::Ptr rhs;
  std::string name;
  std::vector<WeakPtr> requireds;
};

enum class FlattenType { TEMPLATE, DOUBLE, FLOAT };
void flatten(const std::string& func_name,
             const std::vector<Operation::Ptr>& inputs,
             const std::vector<Operation::Ptr>& outputs,
             std::ostream& strm,
             const FlattenType type = FlattenType::DOUBLE);

using jit_func_t = void (*)(double*, double*);
jit_func_t jit_compile(const std::string& func_name,
                       const std::vector<Operation::Ptr>& inputs,
                       const std::vector<Operation::Ptr>& outputs,
                       const FlattenType type = FlattenType::DOUBLE);

Operation::Ptr operator+(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator*(Operation::Ptr lhs, Operation::Ptr rhs);

// unary operators
Operation::Ptr cos(Operation::Ptr op);
Operation::Ptr sin(Operation::Ptr op);
Operation::Ptr negate(Operation::Ptr op);
