#pragma once
#include "cg.hpp"

namespace tenkai {

std::vector<uint8_t> get_instructions(const std::vector<Operation::Ptr>& inputs,
                                      const std::vector<Operation::Ptr>& outputs);

JitFunc<double> compile(const std::vector<Operation::Ptr>& inputs,
                        const std::vector<Operation::Ptr>& outputs);

}  // namespace tenkai
