#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include "cg.hpp"

namespace tenkai {

namespace register_alloc {

using HashType = int32_t;
enum class LocationType { REGISTER, STACK, INPUT, OUTPUT };
struct Location {
  LocationType type;
  size_t idx;
};

std::ostream& operator<<(std::ostream& os, const Location& loc);

struct ConstantSubstitution {
  HashType hash_id;
  double value;
  Location dst;
};

struct RawTransition {
  HashType hash_id;
  Location src;
  Location dst;
};

struct OpTransition {
  HashType hash_id;
  std::vector<size_t> xmms_src;  // operands must be on xmm
  Location dst;
};

using Transition = std::variant<RawTransition, OpTransition, ConstantSubstitution>;

std::ostream& operator<<(std::ostream& os, const Transition& trans);

using TransitionSet = std::vector<Transition>;

struct AllocState {
  AllocState(const std::vector<Operation::Ptr>& inputs, size_t T, size_t n_xmm);

  // query
  std::optional<size_t> get_available_xmm() const;
  size_t get_available_stack() const;

  // members
  std::vector<std::optional<HashType>> xmm_usages_;
  std::vector<std::optional<HashType>> stack_usages_;
  std::unordered_map<HashType, Location> locations_;
};

struct LiveRange {
  size_t appear;
  size_t disappear;
};

std::unordered_map<HashType, LiveRange> compute_live_ranges(
    const std::vector<Operation::Ptr>& opseq);

/* for each time step, compute the hashid that will disappear */
std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq);

class RegisterAllocator {
 public:
  RegisterAllocator(const std::vector<Operation::Ptr>& opseq,
                    const std::vector<Operation::Ptr>& inputs,
                    const std::vector<Operation::Ptr>& outputs,
                    size_t n_xmm = 16)
      : opseq_(opseq),
        live_ranges_(compute_live_ranges(opseq)),
        inputs_(inputs),
        outputs_(outputs),
        disappear_hashid_table_(compute_disappear_hashid_table(opseq)),
        alloc_state_(inputs, opseq.size(), n_xmm - 1),
        transition_sets_(opseq.size()),
        t_(0),
        temp_xmm_idx_(n_xmm - 1) {}

  std::vector<TransitionSet> allocate();

 private:
  void spill_xmm(size_t idx);
  void prepare_value_on_xmm(HashType hash_id, size_t dst_xmm_idx);
  size_t spill_and_prepare_xmm();
  size_t determine_spill_xmm() const;
  void step() { ++t_; }

  std::vector<Operation::Ptr> opseq_;
  std::unordered_map<HashType, LiveRange> live_ranges_;
  std::vector<Operation::Ptr> inputs_;
  std::vector<Operation::Ptr> outputs_;
  std::vector<std::unordered_set<HashType>> disappear_hashid_table_;
  AllocState alloc_state_;
  std::vector<TransitionSet> transition_sets_;
  size_t t_;
  size_t temp_xmm_idx_;  // used for temporary xmm e.g. bit mask in negation
};

}  // namespace register_alloc
}  // namespace tenkai
