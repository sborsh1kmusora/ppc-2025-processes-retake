#include <gtest/gtest.h>

#include "likhanov_m_hypercube/common/include/common.hpp"
#include "likhanov_m_hypercube/mpi/include/ops_mpi.hpp"
#include "likhanov_m_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace likhanov_m_hypercube {

class LikhanovMHypercubeRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_data_ = 25;
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    const OutType expected = static_cast<OutType>(input_data_) * (static_cast<OutType>(1) << (input_data_ - 1));
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(LikhanovMHypercubeRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LikhanovMHypercubeMPI, LikhanovMHypercubeSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LikhanovMHypercubeRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(HypercubePerfTests, LikhanovMHypercubeRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace likhanov_m_hypercube
