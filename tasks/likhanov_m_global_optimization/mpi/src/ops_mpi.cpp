#include "likhanov_m_global_optimization/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "likhanov_m_global_optimization/common/include/common.hpp"
#include "util/include/util.hpp"

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

bool LikhanovMGlobalOptimizationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int max_iters = GetInput();
  const double m = 2.0;

  const double x_min = -5.0;
  const double x_max = 5.0;
  const double y_min = -5.0;
  const double y_max = 5.0;

  struct Rect {
    double x1, x2;
    double y1, y2;
    double R;
  };

  auto f = [](double x, double y) { return (x - 1.5) * (x - 1.5) + (y + 2.0) * (y + 2.0); };

  std::vector<Rect> rects;

  if (rank == 0) {
    rects.push_back({x_min, x_max, y_min, y_max, 0.0});
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    int rect_count = 0;

    if (rank == 0) {
      rect_count = static_cast<int>(rects.size());
    }

    MPI_Bcast(&rect_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
      rects.resize(rect_count);
    }

    MPI_Bcast(rects.data(), rect_count * sizeof(Rect), MPI_BYTE, 0, MPI_COMM_WORLD);

    int base = rect_count / size;
    int rem = rect_count % size;

    int local_n = base + (rank < rem ? 1 : 0);

    int offset = 0;
    for (int i = 0; i < rank; ++i) {
      offset += base + (i < rem ? 1 : 0);
    }

    double local_max_r = -1e18;
    int local_index = -1;

    for (int i = 0; i < local_n; ++i) {
      int idx = offset + i;
      Rect &r = rects[idx];

      double xc = 0.5 * (r.x1 + r.x2);
      double yc = 0.5 * (r.y1 + r.y2);

      double dx = r.x2 - r.x1;
      double dy = r.y2 - r.y1;

      double diag = std::sqrt(dx * dx + dy * dy);

      double val = f(xc, yc);

      r.R = val - m * diag;

      if (r.R > local_max_r) {
        local_max_r = r.R;
        local_index = idx;
      }
    }

    struct {
      double value = 0.0;
      int index = -1;
    } local_data{}, global_data{};

    local_data.value = local_max_r;
    local_data.index = local_index;

    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (rank == 0) {
      int best = global_data.index;

      Rect r = rects[best];

      double xm = 0.5 * (r.x1 + r.x2);
      double ym = 0.5 * (r.y1 + r.y2);

      rects.erase(rects.begin() + best);

      rects.push_back({r.x1, xm, r.y1, ym, 0});
      rects.push_back({xm, r.x2, r.y1, ym, 0});
      rects.push_back({r.x1, xm, ym, r.y2, 0});
      rects.push_back({xm, r.x2, ym, r.y2, 0});
    }
  }

  if (rank == 0) {
    double best_val = 1e18;

    for (auto &r : rects) {
      double xc = 0.5 * (r.x1 + r.x2);
      double yc = 0.5 * (r.y1 + r.y2);
      best_val = std::min(best_val, f(xc, yc));
    }

    GetOutput() = best_val;
  }

  return true;
}

bool LikhanovMGlobalOptimizationMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace likhanov_m_global_optimization
