#pragma once
#include "cg.hpp"
#include "xbyak.h"

namespace tenkai {

class Compiler {
 public:
  struct Location {
    size_t idx;
    bool is_xmm;
  };

  static std::vector<Operation::Ptr> flatten(const std::vector<Operation::Ptr>& inputs,
                                             const std::vector<Operation::Ptr>& outputs);
  Compiler(const std::vector<Operation::Ptr>& inputs,
           const std::vector<Operation::Ptr>& outputs,
           bool avx512);
  std::vector<uint8_t> generate_code();
  JitFunc<double> compile();

 private:
  void move_to_xmm0(int32_t hash_id, Xbyak::CodeGenerator& gen);
  uint8_t get_xmm_register_idx(int32_t hash_id,
                               std::vector<uint32_t>&& dont_spill_xmm,
                               Xbyak::CodeGenerator& gen);
  void update_xmm_suvival_period();
  size_t find_most_unused_xmm_idx(std::vector<uint32_t>&& exclude = {});
  uint8_t get_avaialble_stack_index();
  uint8_t get_available_xmm_index(Xbyak::CodeGenerator& gen);
  void spill_register(uint8_t xmm_idx, Xbyak::CodeGenerator& gen);

  std::vector<Operation::Ptr> inputs_;
  std::vector<Operation::Ptr> outputs_;
  std::vector<Operation::Ptr> operations_;
  std::vector<std::optional<int32_t>> xmm_usage_;
  std::vector<std::optional<int32_t>> xmm_survival_period_;
  std::vector<std::optional<int32_t>> stack_usage_;
  std::unordered_map<int32_t, std::optional<Location>> ophash_to_location_;
};

}  // namespace tenkai
