#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

using Transition =
    std::tuple<HashType, std::optional<Location>, Location>;  // optional<Location> is to express
                                                              // the result of the operation

using TransitionSet = std::vector<Transition>;

struct AllocState {
  AllocState(const std::vector<Operation::Ptr>& inputs, size_t T, size_t n_xmm);

  // query
  size_t most_unused_xmm() const;
  std::optional<size_t> get_available_xmm() const;
  size_t get_available_stack() const;

  // udpate
  void update_xmm_ages();

  // members
  std::vector<std::optional<HashType>> xmm_usages_;
  std::vector<std::optional<HashType>> stack_usages_;
  std::vector<std::optional<size_t>> xmm_ages_;
  std::unordered_map<HashType, Location> locations_;
};

/* for each time step, compute the hashid that will disappear */
std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq);

class RegisterAllocator {
 public:
  RegisterAllocator(const std::vector<Operation::Ptr>& opseq,
                    const std::vector<Operation::Ptr>& inputs,
                    const std::vector<Operation::Ptr>& outputs)
      : opseq_(opseq),
        inputs_(inputs),
        outputs_(outputs),
        disappear_hashid_table_(compute_disappear_hashid_table(opseq)),
        alloc_state_(inputs, opseq.size(), 16) {}

  std::vector<TransitionSet> allocate();

 private:
  std::vector<Operation::Ptr> opseq_;
  std::vector<Operation::Ptr> inputs_;
  std::vector<Operation::Ptr> outputs_;
  std::vector<std::unordered_set<HashType>> disappear_hashid_table_;
  AllocState alloc_state_;
  std::vector<TransitionSet> transition_sets_;
};

}  // namespace register_alloc
}  // namespace tenkai
