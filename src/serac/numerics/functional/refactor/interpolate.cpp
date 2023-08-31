#include "interface.hpp"

#include "serac/numerics/functional/geometry.hpp"

namespace serac {

void interpolate(axom::Array<double, 2>& u_q, const FiniteElementState& u, int q)
{
  auto [qpt_offsets, num_quadrature_points] = quadrature_point_offsets(u.mesh(), q);

  uint32_t num_components = uint32_t(u.gridFunction().VectorDim());
  u_q.resize(num_quadrature_points, num_components);

  std::cout << num_quadrature_points << " " << num_components << std::endl;

  // ...
}

void gradient(axom::Array<double, 3>& du_dX_q, const FiniteElementState& u, int q)
{
  int dim = u.mesh().Dimension();
  auto [qpt_offsets, num_quadrature_points] = quadrature_point_offsets(u.mesh(), q);

  uint32_t num_components = uint32_t(u.gridFunction().VectorDim());
  du_dX_q.resize(num_quadrature_points, num_components, dim);

  std::cout << num_quadrature_points << " " << num_components << " " << dim << std::endl;

  // ...
}

}  // namespace serac
