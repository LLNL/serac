#include "interface.hpp"

namespace serac {

std::tuple<geom_array, uint32_t> quadrature_point_offsets(const mfem::Mesh & mesh, int q) {

  geom_array element_counts = geometry_counts(mesh);
  geom_array qpts_per_elem{};

  for (auto geom : supported_geometries) {
    if (dimension_of(geom) == mesh.Dimension()) {
      qpts_per_elem[uint32_t(geom)] = uint32_t(num_quadrature_points(geom, q));
    } else {
      qpts_per_elem[uint32_t(geom)] = 0;
      element_counts[uint32_t(geom)] = 0;
    }
  }

  uint32_t num_quadrature_points = 0;
  std::array<uint32_t, mfem::Geometry::NUM_GEOMETRIES> qpt_offsets{};
  for (auto geom : supported_geometries) {
    uint32_t g = uint32_t(geom);
    qpt_offsets[g] = num_quadrature_points;
    num_quadrature_points += element_counts[g] * qpts_per_elem[g];
  }

  return std::tuple{qpt_offsets, num_quadrature_points};

}

}