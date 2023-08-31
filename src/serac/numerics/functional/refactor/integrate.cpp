#include "interface.hpp"

namespace serac {

void integrate(FiniteElementDual& f, const axom::Array<double, 2> /*source*/, const axom::Array<double, 3> /*flux*/, int q) {

  auto [qpt_offsets, num_quadrature_points] = quadrature_point_offsets(f.mesh(), q);

  f[0] = num_quadrature_points;

}

}