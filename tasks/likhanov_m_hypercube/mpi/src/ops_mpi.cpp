#include "likhanov_m_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>

#include "likhanov_m_hypercube/common/include/common.hpp"

namespace likhanov_m_hypercube {

LikhanovMHypercubeMPI::LikhanovMHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMHypercubeMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  return size > 0 && ((size & (size - 1)) == 0);
}

bool LikhanovMHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool LikhanovMHypercubeMPI::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!IsPowerOfTwo(size)) {
    return false;
  }

  const int dimension = static_cast<int>(std::log2(size));
  const InType n = GetInput();

  const std::uint64_t vertices = static_cast<std::uint64_t>(1) << n;

  const auto u_rank = static_cast<std::uint64_t>(rank);
  const auto u_size = static_cast<std::uint64_t>(size);

  const std::uint64_t chunk = vertices / u_size;
  const std::uint64_t remainder = vertices % u_size;

  const std::uint64_t start = (u_rank * chunk) + (u_rank < remainder ? u_rank : remainder);

  const std::uint64_t local_size = chunk + (u_rank < remainder ? 1ULL : 0ULL);

  const std::uint64_t end = start + local_size;

  std::uint64_t local_edges = ComputeLocalEdges(start, end, n);

  std::uint64_t sum = local_edges;

  for (int k = 0; k < dimension; ++k) {
    int partner = rank ^ (1 << k);

    if ((rank & (1 << k)) != 0) {
      MPI_Send(&sum, 1, MPI_UINT64_T, partner, 0, MPI_COMM_WORLD);
      break;
    }
    if (partner < size) {
      std::uint64_t received = 0;
      MPI_Recv(&received, 1, MPI_UINT64_T, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      sum += received;
    }
  }

  MPI_Bcast(&sum, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(sum);

  return true;
}

bool LikhanovMHypercubeMPI::IsPowerOfTwo(int value) {
  return value > 0 && ((value & (value - 1)) == 0);
}

std::uint64_t LikhanovMHypercubeMPI::ComputeLocalEdges(std::uint64_t start, std::uint64_t end, InType n) {
  std::uint64_t local_edges = 0;

  for (std::uint64_t vertex = start; vertex < end; ++vertex) {
    for (InType bit = 0; bit < n; ++bit) {
      const std::uint64_t neighbor = vertex ^ (1ULL << bit);
      if (vertex < neighbor) {
        ++local_edges;
      }
    }
  }

  return local_edges;
}

bool LikhanovMHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace likhanov_m_hypercube
