#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <tuple>

#include "likhanov_m_hypercube/common/include/common.hpp"
#include "likhanov_m_hypercube/mpi/include/ops_mpi.hpp"
#include "likhanov_m_hypercube/seq/include/ops_seq.hpp"

namespace likhanov_m_hypercube {

using TestParam = std::tuple<InType>;

static std::uint64_t ReferenceEdges(InType n) {
  return static_cast<std::uint64_t>(n) * (static_cast<std::uint64_t>(1) << (n - 1));
}

class LikhanovMHypercubeRunFuncTests : public ::testing::TestWithParam<TestParam> {};

TEST_P(LikhanovMHypercubeRunFuncTests, SeqCorrectness) {
  const InType n = std::get<0>(GetParam());

  LikhanovMHypercubeSEQ task(n);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const OutType expected = static_cast<OutType>(ReferenceEdges(n));

  EXPECT_EQ(task.GetOutput(), expected);
}

TEST_P(LikhanovMHypercubeRunFuncTests, MpiMatchesSeq) {
  const InType n = std::get<0>(GetParam());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LikhanovMHypercubeMPI mpi_task(n);

  ASSERT_TRUE(mpi_task.Validation());
  ASSERT_TRUE(mpi_task.PreProcessing());
  ASSERT_TRUE(mpi_task.Run());
  ASSERT_TRUE(mpi_task.PostProcessing());

  if (rank == 0) {
    LikhanovMHypercubeSEQ seq_task(n);

    ASSERT_TRUE(seq_task.Validation());
    ASSERT_TRUE(seq_task.PreProcessing());
    ASSERT_TRUE(seq_task.Run());
    ASSERT_TRUE(seq_task.PostProcessing());

    EXPECT_EQ(mpi_task.GetOutput(), seq_task.GetOutput());
  }
}

INSTANTIATE_TEST_SUITE_P(HypercubeTests, LikhanovMHypercubeRunFuncTests,
                         ::testing::Values(std::make_tuple(1), std::make_tuple(2), std::make_tuple(3),
                                           std::make_tuple(4), std::make_tuple(5), std::make_tuple(6)));

}  // namespace likhanov_m_hypercube
