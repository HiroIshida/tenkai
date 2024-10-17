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
using Transition =
    std::tuple<HashType, std::optional<Location>, Location>;  // optional<Location> is to express
                                                              // the result of the operation
using TransitionSet = std::vector<Transition>;

struct AllocState {
  AllocState(const std::vector<Operation::Ptr>& opseq,
             const std::vector<Operation::Ptr>& inputs,
             size_t n_xmm);

  void tell_xmm_assigned(HashType hash_id, size_t xmm_idx);
  void spill_away_register(size_t xmm_idx, std::optional<size_t> stack_idx);
  void load_to_register(size_t stack_idx, size_t xmm_idx);
  size_t most_unused_xmm() const;
  std::optional<size_t> get_available_xmm() const;
  std::vector<std::optional<HashType>> xmm_usage;
  std::vector<std::optional<HashType>> stack_usage;
  std::vector<std::optional<size_t>> xmm_age;
  std::unordered_map<HashType, Location> location;
};

/* for each time step, compute the hashid that will disappear */
std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq);

class RegisterAllocator {
 public:
  std::vector<TransitionSet> allocate(const std::vector<Operation::Ptr>& opseq,
                                      const std::vector<Operation::Ptr>& inputs,
                                      const std::vector<Operation::Ptr>& outputs);

  // allocstate operations might comes with spill and load
  static size_t load_to_xmm(AllocState& as, HashType hash_id);
  static size_t get_available_xmm(AllocState& as);
};

}  // namespace register_alloc
}  // namespace tenkai
