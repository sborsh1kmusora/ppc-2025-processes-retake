#include "likhanov_m_global_optimization/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "likhanov_m_global_optimization/common/include/common.hpp"

namespace likhanov_m_global_optimization {

LikhanovMGlobalOptimizationMPI::LikhanovMGlobalOptimizationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMGlobalOptimizationMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool LikhanovMGlobalOptimizationMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

namespace {

struct Rect {
  double x1, x2;
  double y1, y2;
  double R;
};

double TestFunction(double x, double y) {
  return ((x - 1.5) * (x - 1.5)) + ((y + 2.0) * (y + 2.0));
}

void ComputeDistribution(int total, int rank, int size, int &local_n, int &offset) {
  int base = total / size;
  int rem = total % size;

  local_n = base + (rank < rem ? 1 : 0);

  offset = 0;
  for (int i = 0; i < rank; ++i) {
    offset += base + (i < rem ? 1 : 0);
  }
}

void ComputeLocalMax(std::vector<Rect> &rects, int offset, int local_n, double m, double &local_max_r,
                     int &local_index) {
  local_max_r = -1e18;
  local_index = -1;

  for (int i = 0; i < local_n; ++i) {
    int idx = offset + i;
    Rect &r = rects[idx];

    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);

    double dx = r.x2 - r.x1;
    double dy = r.y2 - r.y1;

    double diag = std::sqrt((dx * dx) + (dy * dy));
    double val = TestFunction(xc, yc);

    r.R = val - (m * diag);

    if (r.R > local_max_r) {
      local_max_r = r.R;
      local_index = idx;
    }
  }
}

void SplitBestRect(std::vector<Rect> &rects, int best_index) {
  Rect r = rects[best_index];

  double xm = 0.5 * (r.x1 + r.x2);
  double ym = 0.5 * (r.y1 + r.y2);

  rects.erase(rects.begin() + best_index);

  rects.push_back({r.x1, xm, r.y1, ym, 0});
  rects.push_back({xm, r.x2, r.y1, ym, 0});
  rects.push_back({r.x1, xm, ym, r.y2, 0});
  rects.push_back({xm, r.x2, ym, r.y2, 0});
}

double ComputeBestValue(const std::vector<Rect> &rects) {
  double best_val = 1e18;

  for (const auto &r : rects) {
    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);

    best_val = std::min(best_val, TestFunction(xc, yc));
  }

  return best_val;
}
}  // namespace

bool LikhanovMGlobalOptimizationMPI::RunImpl() {
  int rank = 0;
  int size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int max_iters = GetInput();
  const double m = 2.0;

  std::vector<Rect> rects;

  if (rank == 0) {
    rects.push_back({-5.0, 5.0, -5.0, 5.0, 0.0});
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    int rect_count = (rank == 0) ? static_cast<int>(rects.size()) : 0;

    MPI_Bcast(&rect_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      rects.resize(rect_count);
    }

    int byte_count = static_cast<int>(rect_count * sizeof(Rect));

    MPI_Bcast(rects.data(), byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);

    int local_n = 0;
    int offset = 0;

    ComputeDistribution(rect_count, rank, size, local_n, offset);

    double local_max_r = 0.0;
    int local_index = -1;

    ComputeLocalMax(rects, offset, local_n, m, local_max_r, local_index);

    struct {
      double value;
      int index;
    } local_data{.value = local_max_r, .index = local_index}, global_data{.value = 0.0, .index = -1};

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0 && global_data.index >= 0) {
      SplitBestRect(rects, global_data.index);
    }
  }

  if (rank == 0) {
    double best_val = ComputeBestValue(rects);
    GetOutput() = static_cast<int>(best_val);
  }

  return true;
}

bool LikhanovMGlobalOptimizationMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace likhanov_m_global_optimization
