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

class AllocState {
 public:
  AllocState(const std::vector<Operation::Ptr>& opseq,
             const std::vector<Operation::Ptr>& inputs,
             size_t n_xmm);

  void step();

  // some destructive operations
  void record_load_from_input(HashType hash_id, size_t xmm_idx);
  void recored_xmm_assigned_as_op_result(HashType hash_id, size_t xmm_idx);
  void record_away_register(size_t xmm_idx, std::optional<size_t> stack_idx);
  void record_load_to_register(size_t stack_idx, size_t xmm_idx);

  // some queries
  size_t most_unused_xmm() const;
  std::optional<size_t> get_available_xmm() const;
  std::vector<TransitionSet> get_history() const { return history_; }

  // some debug utils
  void print_history() const;

  // only accept read-only access
  inline const auto& get_xmm_usages() const { return xmm_usages_; }
  inline const auto& get_stack_usages() const { return stack_usages_; }
  inline const auto& get_xmm_ages() const { return xmm_ages_; }
  inline const auto& get_locations() const { return locations_; }

 private:
  // current states
  std::vector<std::optional<HashType>> xmm_usages_;
  std::vector<std::optional<HashType>> stack_usages_;
  std::vector<std::optional<size_t>> xmm_ages_;
  std::unordered_map<HashType, Location> locations_;

  // history stuff
  std::optional<size_t> t_;
  std::vector<TransitionSet> history_;
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
