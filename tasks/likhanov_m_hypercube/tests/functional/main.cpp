#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "likhanov_m_hypercube/common/include/common.hpp"
#include "likhanov_m_hypercube/mpi/include/ops_mpi.hpp"
#include "likhanov_m_hypercube/seq/include/ops_seq.hpp"

namespace likhanov_m_hypercube {

using TestParam = std::tuple<InType>;

namespace {

[[nodiscard]] std::uint64_t ReferenceEdges(InType n) {
  return static_cast<std::uint64_t>(n) * (static_cast<std::uint64_t>(1) << (n - 1));
}

template <typename TaskType>
void ValidateTask(TaskType &task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
}

template <typename TaskType>
void RunTask(TaskType &task) {
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

template <typename TaskType>
void RunAndValidateTask(TaskType &task) {
  ValidateTask(task);
  RunTask(task);
}

auto hypercube_test_name_generator = [](const testing::TestParamInfo<TestParam> &info) {
  return "N" + std::to_string(std::get<0>(info.param));
};

}  // namespace

class LikhanovMHypercubeRunFuncTests : public ::testing::TestWithParam<TestParam> {};

TEST_P(LikhanovMHypercubeRunFuncTests, SeqCorrectness) {
  const InType n = std::get<0>(GetParam());
  LikhanovMHypercubeSEQ task(n);

  RunAndValidateTask(task);

  const auto expected = static_cast<OutType>(ReferenceEdges(n));
  EXPECT_EQ(task.GetOutput(), expected);
}

TEST_P(LikhanovMHypercubeRunFuncTests, MpiMatchesSeq) {
  const InType n = std::get<0>(GetParam());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LikhanovMHypercubeMPI mpi_task(n);
  RunAndValidateTask(mpi_task);

  if (rank == 0) {
    LikhanovMHypercubeSEQ seq_task(n);
    RunAndValidateTask(seq_task);

    EXPECT_EQ(mpi_task.GetOutput(), seq_task.GetOutput());
  }
}

namespace {
INSTANTIATE_TEST_SUITE_P(HypercubeTests, LikhanovMHypercubeRunFuncTests,
                         ::testing::Values(std::make_tuple(1), std::make_tuple(2), std::make_tuple(3),
                                           std::make_tuple(4), std::make_tuple(5), std::make_tuple(6)),
                         hypercube_test_name_generator);
}  // namespace

}  // namespace likhanov_m_hypercube
