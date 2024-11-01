#include "compile.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_set>
#include <variant>
#include "cg.hpp"
#include "operation_scheduler.hpp"
#include "register_alloc.hpp"
#include "xbyak.h"

namespace tenkai {
namespace compiler {

template <typename T>
std::optional<size_t> find_index(T elem, const std::vector<T>& vec) {
  auto it = std::find(vec.begin(), vec.end(), elem);
  if (it == vec.end()) {
    return std::nullopt;
  }
  return std::distance(vec.begin(), it);
}

// this value must be multiply of page size (4096)?
// must be shared both xbyak constructor and mmap (why? really)
constexpr size_t max_code_size = 4096 * 8;

std::vector<uint8_t> generate_code(const std::vector<Operation::Ptr>& inputs,
                                   const std::vector<Operation::Ptr>& outputs) {
  double (*sin_ptr)(double) = std::sin;
  void* sin_vptr = reinterpret_cast<void*>(sin_ptr);
  double (*cos_ptr)(double) = std::cos;
  void* cos_vptr = reinterpret_cast<void*>(cos_ptr);

  const auto& opseq_flatten = DepthFirstScheduler().flatten(inputs, outputs);
  const auto& transset_seq =
      register_alloc::RegisterAllocator(opseq_flatten, inputs, outputs, 16).allocate();

  auto gen = Xbyak::CodeGenerator(max_code_size);
  gen.endbr64();
  gen.push(gen.r12);
  gen.push(gen.r13);
  gen.push(gen.rbp);
  gen.mov(gen.rbp, gen.rsp);
  size_t sub_size = 1024;  // temporary
  gen.sub(gen.rsp, sub_size);
  gen.mov(gen.r12, gen.rdi);
  gen.mov(gen.r13, gen.rsi);

  for (size_t i = 0; i < opseq_flatten.size(); ++i) {
    std::cout << "===============================" << std::endl;
    const auto& op = opseq_flatten[i];
    const register_alloc::TransitionSet& transset = transset_seq[i];
    std::cout << std::format("operation name: {}", to_string(op->kind)) << std::endl;
    for (const register_alloc::Transition& trans : transset) {
      std::cout << trans;
      if (std::holds_alternative<register_alloc::RawTransition>(trans)) {
        std::variant<std::monostate, Xbyak::Address, Xbyak::Xmm> src, dst;
        const auto& raw_trans = std::get<register_alloc::RawTransition>(trans);
        switch (raw_trans.src.type) {
          case register_alloc::LocationType::INPUT:
            src = gen.ptr[gen.r12 + raw_trans.src.idx * 8];
            break;
          case register_alloc::LocationType::REGISTER:
            src = Xbyak::Xmm(raw_trans.src.idx);
            break;
          case register_alloc::LocationType::STACK:
            src = gen.ptr[gen.rbp - (raw_trans.src.idx + 1) * 8];
            break;
          default:
            throw std::runtime_error("not implemented");
        }
        switch (raw_trans.dst.type) {
          case register_alloc::LocationType::REGISTER:
            dst = Xbyak::Xmm(raw_trans.dst.idx);
            break;
          case register_alloc::LocationType::STACK:
            dst = gen.ptr[gen.rbp - (raw_trans.dst.idx + 1) * 8];
            break;
          case register_alloc::LocationType::OUTPUT:
            dst = gen.ptr[gen.r13 + raw_trans.dst.idx * 8];
            break;
          default:
            throw std::runtime_error("not implemented");
        }

        if (std::holds_alternative<Xbyak::Address>(src) &&
            std::holds_alternative<Xbyak::Xmm>(dst)) {
          gen.vmovsd(std::get<Xbyak::Xmm>(dst), std::get<Xbyak::Address>(src));
        } else if (std::holds_alternative<Xbyak::Xmm>(src) &&
                   std::holds_alternative<Xbyak::Address>(dst)) {
          gen.vmovsd(std::get<Xbyak::Address>(dst), std::get<Xbyak::Xmm>(src));
        } else if (std::holds_alternative<Xbyak::Xmm>(src) &&
                   std::holds_alternative<Xbyak::Xmm>(dst)) {
          gen.vmovsd(std::get<Xbyak::Xmm>(dst), std::get<Xbyak::Xmm>(src));
        } else {
          throw std::runtime_error("not implemented");
        }
      } else if (std::holds_alternative<register_alloc::ConstantSubstitution>(trans)) {
        // TODO: shoule I prepare data section? but I guess just directly mov is
        // faster becuase no need to load from memory
        const auto& sub_trans = std::get<register_alloc::ConstantSubstitution>(trans);
        uint64_t value_as_uint64 = std::bit_cast<uint64_t>(sub_trans.value);
        gen.mov(gen.rax, value_as_uint64);
        gen.movq(Xbyak::Xmm(sub_trans.dst.idx), gen.rax);
      } else if (std::holds_alternative<register_alloc::OpTransition>(trans)) {
        const auto& op_trans = std::get<register_alloc::OpTransition>(trans);
        auto dst = Xbyak::Xmm(op_trans.dst.idx);

        auto instr_operand_xmm_size = op_trans.xmms_src.size();  // different from op.args.size()
        if (instr_operand_xmm_size == 0) {
          switch (op->kind) {
            case OpKind::SIN:
              gen.call(sin_vptr);
              break;
            case OpKind::COS:
              gen.call(cos_vptr);
              break;
            default:
              throw std::runtime_error("not implemented");
          }
        } else if (instr_operand_xmm_size == 1) {
          throw std::runtime_error("not implemented");
        } else if (instr_operand_xmm_size == 2) {
          auto arg0 = Xbyak::Xmm(op_trans.xmms_src[0]);
          auto arg1 = Xbyak::Xmm(op_trans.xmms_src[1]);
          switch (op->kind) {
            case OpKind::NEGATE: {
              gen.mov(gen.rax, 0x8000000000000000);
              gen.movq(arg1, gen.rax);
              gen.vxorpd(dst, arg0, arg1);
              break;
            }
            case OpKind::ADD:
              gen.vaddsd(dst, arg0, arg1);
              break;
            case OpKind::SUB:
              gen.vsubsd(dst, arg0, arg1);
              break;
            case OpKind::MUL:
              gen.vmulsd(dst, arg0, arg1);
              break;
            default:
              throw std::runtime_error(
                  std::format("not implemented operation name: {}", to_string(op->kind)));
          }
        }
      } else {
        throw std::runtime_error("not implemented");
      }
    }
  }

  gen.mov(gen.rsp, gen.rbp);
  gen.pop(gen.rbp);
  gen.pop(gen.r13);
  gen.pop(gen.r12);
  gen.ret();

  auto code = std::vector<uint8_t>(gen.getSize());
  std::copy(gen.getCode(), gen.getCode() + gen.getSize(), code.begin());
  return code;
}

JitFunc<double> compile(const std::vector<Operation::Ptr>& inputs,
                        const std::vector<Operation::Ptr>& outputs) {
  auto code = generate_code(inputs, outputs);
  void* mem = mmap(NULL, max_code_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  uint8_t* instruction = static_cast<uint8_t*>(mem);
  std::memcpy(instruction, code.data(), code.size());
  mprotect(mem, max_code_size, PROT_READ | PROT_EXEC) == -1;
  auto add_func = reinterpret_cast<JitFunc<double>>(instruction);
  return add_func;
}

}  // namespace compiler
}  // namespace tenkai
