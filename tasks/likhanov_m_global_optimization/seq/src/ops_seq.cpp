#include "likhanov_m_global_optimization/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "likhanov_m_global_optimization/common/include/common.hpp"

namespace likhanov_m_global_optimization {

LikhanovMGlobalOptimizationSEQ::LikhanovMGlobalOptimizationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMGlobalOptimizationSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool LikhanovMGlobalOptimizationSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool LikhanovMGlobalOptimizationSEQ::RunImpl() {
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

  auto f = [](double x, double y) { return ((x - 1.5) * (x - 1.5)) + ((y + 2.0) * (y + 2.0)); };

  std::vector<Rect> rects;
  rects.push_back({x_min, x_max, y_min, y_max, 0.0});

  for (int iter = 0; iter < max_iters; ++iter) {
    double max_r = -1e18;
    int best_index = -1;

    for (size_t i = 0; i < rects.size(); ++i) {
      Rect &r = rects[i];

      double xc = 0.5 * (r.x1 + r.x2);
      double yc = 0.5 * (r.y1 + r.y2);

      double dx = r.x2 - r.x1;
      double dy = r.y2 - r.y1;

      double diag = std::sqrt((dx * dx) + (dy * dy));

      double val = f(xc, yc);

      r.R = val - (m * diag);

      if (r.R > max_r) {
        max_r = r.R;
        best_index = static_cast<int>(i);
      }
    }

    Rect r = rects[best_index];

    double xm = 0.5 * (r.x1 + r.x2);
    double ym = 0.5 * (r.y1 + r.y2);

    rects.erase(rects.begin() + best_index);

    rects.push_back({r.x1, xm, r.y1, ym, 0});
    rects.push_back({xm, r.x2, r.y1, ym, 0});
    rects.push_back({r.x1, xm, ym, r.y2, 0});
    rects.push_back({xm, r.x2, ym, r.y2, 0});
  }

  double best_val = 1e18;

  for (auto &r : rects) {
    double xc = 0.5 * (r.x1 + r.x2);
    double yc = 0.5 * (r.y1 + r.y2);
    best_val = std::min(best_val, f(xc, yc));
  }

  GetOutput() = static_cast<int>(best_val);

  return true;
}

bool LikhanovMGlobalOptimizationSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace likhanov_m_global_optimization
