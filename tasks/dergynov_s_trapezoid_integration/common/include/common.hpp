#pragma once

#include <cmath>
#include <cstdint>
#include <tuple>

#include "task/include/task.hpp"

namespace dergynov_s_trapezoid_integration {

struct Input {
  double a;
  double b;
  int n;
  int func_id;
};

using InType = Input;
using OutType = double;
using TestType = std::tuple<int, const char *>;
using BaseTask = ppc::task::Task<InType, OutType>;

enum class FunctionId : std::uint8_t {
  kLinear = 0,
  kQuadratic = 1,
  kSin = 2,
};

inline bool IsValidFunctionId(int id) {
  return id >= 0 && id <= 2;
}

inline double Function(double x, int id) {
  if (!IsValidFunctionId(id)) {
    return 0.0;
  }

  switch (static_cast<FunctionId>(id)) {
    case FunctionId::kLinear:
      return x;
    case FunctionId::kQuadratic:
      return x * x;
    case FunctionId::kSin:
      return std::sin(x);
    default:
      return 0.0;
  }
}

inline double GetExactIntegral(const InType &in) {
  if (!IsValidFunctionId(in.func_id)) {
    return 0.0;
  }

  const double a = in.a;
  const double b = in.b;

  switch (static_cast<FunctionId>(in.func_id)) {
    case FunctionId::kLinear:
      return (b * b - a * a) / 2.0;
    case FunctionId::kQuadratic:
      return (b * b * b - a * a * a) / 3.0;
    case FunctionId::kSin:
      return std::cos(a) - std::cos(b);
    default:
      return 0.0;
  }
}

}  // namespace dergynov_s_trapezoid_integration
