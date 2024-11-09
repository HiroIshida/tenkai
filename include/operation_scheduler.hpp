#include "cg.hpp"

namespace tenkai {

namespace compiler {

class SchedulerInterface {
 public:
  virtual std::vector<Operation::Ptr> flatten(const std::vector<Operation::Ptr>& inputs,
                                              const std::vector<Operation::Ptr>& outputs) = 0;
};

class DepthFirstScheduler : public SchedulerInterface {
 public:
  std::vector<Operation::Ptr> flatten(const std::vector<Operation::Ptr>& inputs,
                                      const std::vector<Operation::Ptr>& outputs) override;
};

class ExtCallFirstScheduler : public SchedulerInterface {
 public:
  std::vector<Operation::Ptr> flatten(const std::vector<Operation::Ptr>& inputs,
                                      const std::vector<Operation::Ptr>& outputs) override;
};

}  // namespace compiler
}  // namespace tenkai
