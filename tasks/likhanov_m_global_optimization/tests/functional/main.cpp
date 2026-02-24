#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "likhanov_m_global_optimization/common/include/common.hpp"
#include "likhanov_m_global_optimization/mpi/include/ops_mpi.hpp"
#include "likhanov_m_global_optimization/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace likhanov_m_global_optimization {

class LikhanovMGlobalOptimizationRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      return std::isfinite(output_data) && output_data >= 0.0;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

using TaskFactory = std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>;

std::shared_ptr<ppc::task::Task<InType, OutType>> CreateSEQ(InType in) {
  return std::make_shared<LikhanovMGlobalOptimizationSEQ>(in);
}

std::shared_ptr<ppc::task::Task<InType, OutType>> CreateMPI(InType in) {
  return std::make_shared<LikhanovMGlobalOptimizationMPI>(in);
}

const std::array<TestType, 5> kParams = {std::make_tuple(5, "5"), std::make_tuple(10, "10"), std::make_tuple(20, "20"),
                                         std::make_tuple(30, "30"), std::make_tuple(40, "40")};

auto GenerateParams() {
  std::vector<std::tuple<TaskFactory, std::string, TestType>> params;

  for (const auto &param : kParams) {
    params.emplace_back(CreateSEQ, "SEQ", param);
    params.emplace_back(CreateMPI, "MPI", param);
  }

  return params;
}

TEST_P(LikhanovMGlobalOptimizationRunFuncTests, GlobalOptimization) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(LikhanovMGlobalOptimization, LikhanovMGlobalOptimizationRunFuncTests,
                         ::testing::ValuesIn(GenerateParams()),
                         [](const testing::TestParamInfo<LikhanovMGlobalOptimizationRunFuncTests::ParamType> &info) {
                           const auto &task_name = std::get<1>(info.param);
                           const auto &test_param = std::get<2>(info.param);
                           return task_name + "_" + LikhanovMGlobalOptimizationRunFuncTests::PrintTestParam(test_param);
                         });

TEST(LikhanovMGlobalOptimizationConsistency, SEQvsMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const InType input = 30;

  LikhanovMGlobalOptimizationSEQ seq_task(input);
  LikhanovMGlobalOptimizationMPI mpi_task(input);

  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(mpi_task.Validation());

  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(mpi_task.PreProcessing());

  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(mpi_task.Run());

  ASSERT_TRUE(seq_task.PostProcessing());
  ASSERT_TRUE(mpi_task.PostProcessing());

  const double seq_result = seq_task.GetOutput();
  const double mpi_result = mpi_task.GetOutput();

  if (rank == 0) {
    const double k_tolerance = 1e-2;
    ASSERT_NEAR(seq_result, mpi_result, k_tolerance);
  }
}

TEST(LikhanovMGlobalOptimizationConvergence, MoreIterationsBetter) {
  const InType small_iter = 5;
  const InType big_iter = 40;

  LikhanovMGlobalOptimizationSEQ small_task(small_iter);
  LikhanovMGlobalOptimizationSEQ big_task(big_iter);

  ASSERT_TRUE(small_task.Validation());
  ASSERT_TRUE(big_task.Validation());

  ASSERT_TRUE(small_task.PreProcessing());
  ASSERT_TRUE(big_task.PreProcessing());

  ASSERT_TRUE(small_task.Run());
  ASSERT_TRUE(big_task.Run());

  ASSERT_TRUE(small_task.PostProcessing());
  ASSERT_TRUE(big_task.PostProcessing());

  const double small_res = small_task.GetOutput();
  const double big_res = big_task.GetOutput();

  ASSERT_LE(big_res, small_res + 1e-6);
}

}  // namespace
}  // namespace likhanov_m_global_optimization
