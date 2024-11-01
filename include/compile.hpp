#pragma once
#include "cg.hpp"
#include "xbyak.h"

namespace tenkai {

namespace compiler {

std::vector<uint8_t> generate_code(const std::vector<Operation::Ptr>& inputs,
                                   const std::vector<Operation::Ptr>& outputs);
JitFunc<double> compile(const std::vector<Operation::Ptr>& inputs,
                        const std::vector<Operation::Ptr>& outputs);

}  // namespace compiler

}  // namespace tenkai
