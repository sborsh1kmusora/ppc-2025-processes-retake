#include "likhanov_m_hypercube/seq/include/ops_seq.hpp"

#include <cstdint>

#include "likhanov_m_hypercube/common/include/common.hpp"

namespace likhanov_m_hypercube {

LikhanovMHypercubeSEQ::LikhanovMHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LikhanovMHypercubeSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool LikhanovMHypercubeSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool LikhanovMHypercubeSEQ::RunImpl() {
  const InType dimension = GetInput();

  const std::uint64_t vertices = static_cast<std::uint64_t>(1) << dimension;
  std::uint64_t total_edges = 0;

  for (std::uint64_t vertex = 0; vertex < vertices; ++vertex) {
    for (InType bit = 0; bit < dimension; ++bit) {
      const std::uint64_t neighbor = vertex ^ (1ULL << bit);
      if (vertex < neighbor) {
        ++total_edges;
      }
    }
  }

  GetOutput() = static_cast<OutType>(total_edges);
  return true;
}

bool LikhanovMHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace likhanov_m_hypercube
