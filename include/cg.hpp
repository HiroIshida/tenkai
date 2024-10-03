#pragma once

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace tenkai {

std::string generate_random_string(size_t length);

enum class OpKind { NIL, ADD, SUB, MUL, COS, SIN, NEGATE, VALIABLE, ZERO, ONE, CONSTANT, EXTCALL };

struct Operation : std::enable_shared_from_this<Operation> {
  using Ptr = std::shared_ptr<Operation>;
  using WeakPtr = std::weak_ptr<Operation>;
  Operation();
  Operation(OpKind kind, std::vector<Operation::Ptr> args, int32_t hash_id);
  void to_cpp_expr(std::ostream& strm) const;
  static Operation::Ptr create(OpKind kind, std::vector<Operation::Ptr>&& args, int32_t hash_id);
  static Operation::Ptr make_var();
  static Operation::Ptr make_zero();
  static Operation::Ptr make_one();
  static Operation::Ptr make_ext_func(std::string&& name, std::vector<Operation::Ptr>&& args);
  static Operation::Ptr make_constant(double value);
  std::vector<Operation::Ptr> get_leafs();
  inline bool is_nullaryop() const { return args.size() == 0; }
  inline bool is_unaryop() const { return args.size() == 1; }
  inline Operation::Ptr first() const { return args[0]; }
  inline Operation::Ptr second() const { return args[1]; }
  inline Operation::Ptr third() const { return args[2]; }

  // member variables
  OpKind kind;
  std::vector<Operation::Ptr> args;
  int32_t hash_id;
  std::optional<std::string> ext_func_name;  // used only for EXTCALL kind
  std::optional<double> constant_value;      // used only for zero, one, constant
};

void flatten(const std::string& func_name,
             const std::vector<Operation::Ptr>& inputs,
             const std::vector<Operation::Ptr>& outputs,
             std::ostream& strm,
             const std::string& type_name);

template <typename T>
using JitFunc = void (*)(T*, T*, void**);

template <typename T>
JitFunc<T> jit_compile(const std::vector<Operation::Ptr>& inputs,
                       const std::vector<Operation::Ptr>& outputs,
                       const std::string& backend = "g++",
                       bool disas = false);

Operation::Ptr operator+(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator-(Operation::Ptr lhs, Operation::Ptr rhs);
Operation::Ptr operator*(Operation::Ptr lhs, Operation::Ptr rhs);

// unary operators
Operation::Ptr cos(Operation::Ptr op);
Operation::Ptr sin(Operation::Ptr op);
Operation::Ptr operator-(Operation::Ptr op);

}  // namespace tenkai
