#pragma once

#include "likhanov_m_global_optimization/common/include/common.hpp"
#include "task/include/task.hpp"

namespace likhanov_m_global_optimization {

class LikhanovMGlobalOptimizationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LikhanovMGlobalOptimizationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace likhanov_m_global_optimization
