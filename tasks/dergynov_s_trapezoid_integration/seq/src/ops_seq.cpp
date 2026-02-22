#include "dergynov_s_trapezoid_integration/seq/include/ops_seq.hpp"

#include "dergynov_s_trapezoid_integration/common/include/common.hpp"

namespace dergynov_s_trapezoid_integration {

DergynovSTrapezoidIntegrationSEQ::DergynovSTrapezoidIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DergynovSTrapezoidIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return (in.n > 0) && (in.a < in.b) && IsValidFunctionId(in.func_id);
}

bool DergynovSTrapezoidIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DergynovSTrapezoidIntegrationSEQ::RunImpl() {
  const auto &in = GetInput();

  const double a = in.a;
  const double b = in.b;
  const int n = in.n;

  if (n <= 0 || !(a < b)) {
    return false;
  }

  const double h = (b - a) / static_cast<double>(n);

  double sum = 0.0;
  for (int i = 1; i < n; ++i) {
    sum += Function(a + (h * i), in.func_id);
  }

  GetOutput() = h * (0.5 * (Function(a, in.func_id) + Function(b, in.func_id)) + sum);

  return true;
}

bool DergynovSTrapezoidIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dergynov_s_trapezoid_integration
