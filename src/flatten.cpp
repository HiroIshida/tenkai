#include <dlfcn.h>
#include <format>
#include <fstream>
#include <iostream>
#include <stack>
#include <typeinfo>
#include <unordered_map>
#include "cg.hpp"

namespace tenkai {

template <typename T>
constexpr const char* getTypeString() {
  if constexpr (std::is_same_v<T, double>)
    return "double";
  else if constexpr (std::is_same_v<T, float>)
    return "float";
  else
    return "unknown";
}

template <typename T>
constexpr void opkind_to_cppfunc_name(OpKind kind, std::ostream& strm) {
  std::string type_name = getTypeString<T>();
  // use std::plus .. to handle these primitive operation as function
  switch (kind) {
    // clang-format off
    case OpKind::ADD: strm << std::format("std::plus<{}>()", type_name); break;
    case OpKind::SUB: strm << std::format("std::minus<{}>()", type_name); break;
    case OpKind::MUL: strm << std::format("std::multiplies<{}>()", type_name); break;
    case OpKind::COS: strm << "cos"; break;
    case OpKind::SIN: strm << "sin"; break;
    case OpKind::NEGATE: strm << std::format("std::negate<{}>()", type_name); break;
    default: throw std::runtime_error("unknown operator");
      // clang-format on
  }
}

void flatten(const std::string& func_name,
             const std::vector<Operation::Ptr>& inputs,
             const std::vector<Operation::Ptr>& outputs,
             std::ostream& strm,
             const std::string& type_name) {
  // Currently no overlapping is allowed between inputs and outputs..
  for (auto& input : inputs) {
    for (auto& output : outputs) {
      if (input == output) {
        throw std::runtime_error("detected overlapping between inputs and outputs");
      }
    }
  }

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
      for (auto& arg : op->args) {
        if (arg != nullptr) {
          opstack.push(arg);
        }
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
  strm << "#include <functional>" << std::endl;
  strm << "extern \"C\" {" << std::endl;
  strm << "void " << func_name << "(" << type_name << "* input, " << type_name << "* output) {"
       << std::endl;
  std::unordered_map<std::string, bool> is_evaluated;
  for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
    auto op = *it;
    if (is_evaluated.find(remapped_name(op)) != is_evaluated.end()) {
      continue;
    }
    if (op->is_nullaryop()) {
      throw std::runtime_error("must not reach here");
    }

    strm << "  ";  // for indent
    bool is_intermediate = (remapped_name(op).find("output") == std::string::npos);
    if (is_intermediate) {
      strm << "auto ";
    }
    strm << remapped_name(op) << " = ";
    opkind_to_cppfunc_name<double>(op->kind, strm);
    strm << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i != 0) {
        strm << ", ";
      }
      strm << remapped_name(op->args[i]);
    }
    strm << ");" << std::endl;
    is_evaluated[remapped_name(op)] = true;
  }
  strm << "}" << std::endl;
  strm << "}" << std::endl;  // for extern "C"
}

template <typename T>
JitFunc<T> jit_compile(const std::vector<Operation::Ptr>& inputs,
                       const std::vector<Operation::Ptr>& outputs,
                       const std::string& backend,
                       bool disas) {
  std::string type_name;
  if constexpr (std::is_same<T, double>::value) {
    type_name = "double";
  } else if constexpr (std::is_same<T, float>::value) {
    type_name = "float";
  } else {
    throw std::runtime_error("unsupported type");
  }

  std::string func_name = "generated_" + generate_random_string(16);
  std::string source_name = "/tmp/" + func_name + ".cpp";
  std::string so_name = "/tmp/" + func_name + ".so";

  auto fs = std::ofstream(source_name);
  flatten(func_name, inputs, outputs, fs, type_name);
  fs.close();
  std::string cmd = backend + " -O3 -shared -fPIC " + source_name + " -o " + so_name;

  auto ret = system(cmd.c_str());
  if (ret != 0) {
    throw std::runtime_error("failed to compile");
  }

  if (disas) {
    std::string disas_cmd = "objdump --disassemble=" + func_name + " " + so_name;
    std::cout << disas_cmd << std::endl;
    system(disas_cmd.c_str());
  }

  void* lib = dlopen(so_name.c_str(), RTLD_LAZY);
  if (lib == nullptr) {
    throw std::runtime_error("dlopen failed");
  }
  auto func = reinterpret_cast<JitFunc<T>>(dlsym(lib, func_name.c_str()));
  if (func == nullptr) {
    throw std::runtime_error("dlsym failed");
  }

  remove(source_name.c_str());
  remove(so_name.c_str());
  return func;
}

template JitFunc<double> jit_compile<double>(const std::vector<Operation::Ptr>& inputs,
                                             const std::vector<Operation::Ptr>& outputs,
                                             const std::string& backend,
                                             bool disas);

template JitFunc<float> jit_compile<float>(const std::vector<Operation::Ptr>& inputs,
                                           const std::vector<Operation::Ptr>& outputs,
                                           const std::string& backend,
                                           bool disas);

}  // namespace tenkai
