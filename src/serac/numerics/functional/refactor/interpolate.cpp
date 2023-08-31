#include "interface.hpp"

namespace serac {

void interpolate(axom::Array<double, 3>& u_q, const FiniteElementState& u, int q)
{
  // ...
}

void gradient(axom::Array<double, 4>& du_dX_q, const FiniteElementState& u, int q)
{
  // ...
}

}  // namespace serac
