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
  std::vector<std::optional<HashType>> xmm_usage;
  std::vector<std::optional<HashType>> stack_usage;
  std::unordered_map<HashType, Location> location;
};

/* for each time step, compute the hashid that will disappear */
std::vector<std::unordered_set<HashType>> compute_disappear_hashid_table(
    const std::vector<Operation::Ptr>& opseq);

class RegsiterAllocatorBase {
 public:
  virtual std::vector<TransitionSet> allocate(const std::vector<Operation::Ptr>& opseq) = 0;
  virtual ~RegsiterAllocatorBase() {}
};

class AdhocRegisterAllocator : public RegsiterAllocatorBase {
 public:
  std::vector<TransitionSet> allocate(const std::vector<Operation::Ptr>& opseq) override;
};

}  // namespace register_alloc
}  // namespace tenkai
