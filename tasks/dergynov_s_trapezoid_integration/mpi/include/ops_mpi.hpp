#pragma once

#include "dergynov_s_trapezoid_integration/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergynov_s_trapezoid_integration {

class DergynovSTrapezoidIntegrationMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit DergynovSTrapezoidIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dergynov_s_trapezoid_integration
