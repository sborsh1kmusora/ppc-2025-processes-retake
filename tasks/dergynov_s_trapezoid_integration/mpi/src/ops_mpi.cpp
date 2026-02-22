#include "dergynov_s_trapezoid_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "dergynov_s_trapezoid_integration/common/include/common.hpp"

namespace dergynov_s_trapezoid_integration {

DergynovSTrapezoidIntegrationMPI::DergynovSTrapezoidIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DergynovSTrapezoidIntegrationMPI::ValidationImpl() {
  const auto &in = GetInput();
  return (in.n > 0) && (in.a < in.b);
}

bool DergynovSTrapezoidIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DergynovSTrapezoidIntegrationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType in = (rank == 0) ? GetInput() : InType{};
  MPI_Bcast(&in, sizeof(InType), MPI_BYTE, 0, MPI_COMM_WORLD);
  GetInput() = in;

  const double a = in.a;
  const double b = in.b;
  const int n = in.n;

  if (n <= 0 || !(a < b)) {
    if (rank == 0) {
      GetOutput() = 0.0;
    }
    return true;
  }

  int base = n / size;
  int rem = n % size;
  int local_n = base + (rank < rem ? 1 : 0);
  int start = (rank * base) + std::min(rank, rem);
  int end = start + local_n;

  const double h = (b - a) / static_cast<double>(n);

  double local_sum = 0.0;
  for (int i = start; i < end; ++i) {
    double x1 = a + (h * i);
    double x2 = a + (h * (i + 1));
    local_sum += 0.5 * (Function(x1, in.func_id) + Function(x2, in.func_id)) * h;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool DergynovSTrapezoidIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dergynov_s_trapezoid_integration
