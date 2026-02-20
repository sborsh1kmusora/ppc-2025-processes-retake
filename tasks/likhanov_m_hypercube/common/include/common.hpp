#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace likhanov_m_hypercube {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace likhanov_m_hypercube
