#pragma once

#include <cstdint>

#include "likhanov_m_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace likhanov_m_hypercube {

class LikhanovMHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LikhanovMHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  [[nodiscard]] static bool IsPowerOfTwo(int value);
  [[nodiscard]] static std::uint64_t ComputeLocalEdges(std::uint64_t start, std::uint64_t end, InType n);
};

}  // namespace likhanov_m_hypercube
