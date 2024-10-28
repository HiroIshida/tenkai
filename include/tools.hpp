#include <ostream>
#include "cg.hpp"

namespace tenkai {

void write_to_dotfile(const std::vector<Operation::Ptr>& inputs,
                      const std::vector<Operation::Ptr>& outputs,
                      std::ostream& os);

}  // namespace tenkai
