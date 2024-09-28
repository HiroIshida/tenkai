#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stack>
#include <unordered_map>
#include "cg.hpp"

void flatten(const std::string& func_name,
             const std::vector<Operation::Ptr>& inputs,
             const std::vector<Operation::Ptr>& outputs,
             std::ostream& strm,
             const FlattenType type) {
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
  strm << "#include <cmath>" << std::endl;
  strm << "extern \"C\" {" << std::endl;
  if (type == FlattenType::TEMPLATE) {
    strm << "template <typename T>" << std::endl;
  }
  auto type_name = [](FlattenType type) -> std::string {
    switch (type) {
      case FlattenType::TEMPLATE:
        return "T";
      case FlattenType::DOUBLE:
        return "double";
      case FlattenType::FLOAT:
        return "float";
      default:
        throw std::runtime_error("unknown type");
    }
  };
  strm << "void flattend_" << func_name << "(" << type_name(type) << "* input, "
       << type_name(type) << "* output) {" << std::endl;

  std::unordered_map<std::string, bool> is_evaluated;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    auto op = *it;
    if (is_evaluated.find(remapped_name(op)) != is_evaluated.end()) {
      continue;
    }
    if (remapped_name(op).find("output") == std::string::npos) {
      strm << "  auto " << remapped_name(op) << " = ";
    } else {
      strm << "  " << remapped_name(op) << " = ";
    }
    if (op->is_nullaryop()) {
      throw std::runtime_error("must not reach here");
    } else if (op->is_unaryop()) {
      switch (op->kind) {
        case OpKind::COS:
          strm << "cos(" << remapped_name(op->lhs) << ");" << std::endl;
          break;
        case OpKind::SIN:
          strm << "sin(" << remapped_name(op->lhs) << ");" << std::endl;
          break;
        case OpKind::NEGATE:
          strm << "-" << remapped_name(op->lhs) << ";" << std::endl;
          break;
        default:
          throw std::runtime_error("unknown operator");
      }
    } else {
      strm << remapped_name(op->lhs) << " ";
      switch (op->kind) {
        case OpKind::ADD:
          strm << "+";
          break;
        case OpKind::SUB:
          strm << "-";
          break;
        case OpKind::MUL:
          strm << "*";
          break;
        default:
          throw std::runtime_error("unknown operator");
      }
      strm << " " << remapped_name(op->rhs) << ";" << std::endl;
    }
    is_evaluated[remapped_name(op)] = true;
  }
  strm << "}" << std::endl;
  strm << "}" << std::endl;  // for extern "C"
}

jit_func_t jit_compile(const std::string& func_name,
                       const std::vector<Operation::Ptr>& inputs,
                       const std::vector<Operation::Ptr>& outputs,
                       const FlattenType type) {
  auto fs = std::ofstream("/tmp/hoge.cpp");
  flatten(func_name, inputs, outputs, fs, type);
  fs.close();
  std::string cmd = "gcc -O3 -shared -fPIC /tmp/hoge.cpp -o /tmp/hoge.so";
  system(cmd.c_str());
  void* lib = dlopen("/tmp/hoge.so", RTLD_LAZY);
  if (lib == nullptr) {
    throw std::runtime_error("dlopen failed");
  }
  std::string sym = "flattend_" + func_name;
  auto func = reinterpret_cast<jit_func_t>(dlsym(lib, sym.c_str()));
  if (func == nullptr) {
    throw std::runtime_error("dlsym failed");
  }
  return func;
}
