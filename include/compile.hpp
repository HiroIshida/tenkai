#pragma once
#include "cg.hpp"

namespace tenkai {

std::vector<uint8_t> compile(const std::vector<Operation::Ptr>& inputs,
                             const std::vector<Operation::Ptr>& outputs);

}
