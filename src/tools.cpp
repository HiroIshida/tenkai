#include "tools.hpp"
#include <algorithm>
#include <format>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include "cg.hpp"

namespace tenkai {

void write_to_dotfile(const std::vector<Operation::Ptr>& inputs,
                      const std::vector<Operation::Ptr>& outputs,
                      std::ostream& os) {
  std::unordered_map<int32_t, std::string> op2name_map;
  size_t counter = 0;
  auto get_name = [&](int32_t hash_id) {
    if (op2name_map.find(hash_id) == op2name_map.end()) {
      std::string node_name = "node" + std::to_string(counter);
      op2name_map[hash_id] = node_name;
      counter++;
    }
    return op2name_map[hash_id];
  };

  os << "digraph OperationGraph {\n";

  std::stack<Operation::Ptr> opstack;
  std::unordered_set<int32_t> visited;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& op = outputs[i];
    opstack.push(op);
  }

  bool separate_constant_node = true;

  // Traverse the graph in DFS order
  while (!opstack.empty()) {
    auto op = opstack.top();
    opstack.pop();
    if (visited.find(op->hash_id) != visited.end()) {
      continue;
    }
    visited.insert(op->hash_id);

    if (op->kind == OpKind::SIN || op->kind == OpKind::COS) {
      os << std::format("  {} [label={}, color=red, style=filled];\n", get_name(op->hash_id),
                        to_string(op->kind));
    } else {
      os << std::format("  {} [label={}];\n", get_name(op->hash_id), to_string(op->kind));
    }

    for (const auto& arg : op->args) {
      if (separate_constant_node && arg->kind == OpKind::CONSTANT) {
        std::string constant_node_name = "node" + std::to_string(counter);
        counter++;
        os << std::format("  {} [label={}];\n", constant_node_name, to_string(arg->kind));
        os << std::format("  {} -> {};\n", get_name(op->hash_id), constant_node_name);
      } else {
        os << std::format("  {} -> {};\n", get_name(op->hash_id), get_name(arg->hash_id));
        if (visited.find(arg->hash_id) == visited.end()) {
          opstack.push(arg);
        }
      }
    }
  }

  os << " { rank=source; ";
  for (const auto& output : outputs) {
    os << get_name(output->hash_id) << "; ";
  }
  os << " }\n";
  os << " { rank=sink; ";
  for (const auto& input : inputs) {
    os << get_name(input->hash_id) << "; ";
  }
  os << " }\n";
  os << "}\n";
}

}  // namespace tenkai
