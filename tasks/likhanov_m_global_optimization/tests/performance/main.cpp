#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>

#include "likhanov_m_global_optimization/common/include/common.hpp"
#include "likhanov_m_global_optimization/mpi/include/ops_mpi.hpp"
#include "likhanov_m_global_optimization/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace likhanov_m_global_optimization {

class LikhanovMGlobalOptimizationRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  int k_count = 5000;
  InType input_data{};

  void SetUp() override {
    input_data = k_count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return std::isfinite(output_data) && !std::isnan(output_data) && output_data >= 0.0;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(LikhanovMGlobalOptimizationRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LikhanovMGlobalOptimizationMPI, LikhanovMGlobalOptimizationSEQ>(
        PPC_SETTINGS_example_processes_3);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LikhanovMGlobalOptimizationRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(LikhanovMGlobalOptimizationPerf, LikhanovMGlobalOptimizationRunPerfTests, kGtestValues,
                         kPerfTestName);

}  // namespace
}  // namespace likhanov_m_global_optimization
