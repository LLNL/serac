#include "interface.hpp"

#include "serac/numerics/functional/geometry.hpp"
#include "serac/numerics/functional/element_restriction.hpp"

#include "serac/numerics/functional/refactor/ndview.hpp"

namespace serac {

template <mfem::Geometry::Type geom>
void interpolate_kernel(ndview< double, 3 > u_Q, ndview< double, 4 > du_dxi_Q, ndview < double, 3 > u_E, int p)
{
  int num_elements = u_E.shape[0];
  int components = u_E.shape[1];
  int qpts_per_elem = u_Q.shape[0] / num_elements;

  finite_element< geom, H1Dynamic > element{p + 1};

  std::vector< double > B;
  std::vector< double > G;

  for (uint32_t e = 0; e < num_elements; e++) {
    for (int i = 0; i < components; i++) {
      element::interpolate(u_Q(e, i), du_dX_Q(e, i), rule)
    }
  }
}

void interpolate(axom::Array<double, 2>& u_Q, axom::Array<double, 3>& du_dxi_Q, const FiniteElementState& u_T, int q)
{
  auto mem_type = mfem::Device::GetMemoryType();
  auto fespace = u_T.gridFunction().ParFESpace();

  // T->L
  const mfem::Operator * P = fespace->GetProlongationMatrix();
  mfem::Vector u_L(P->Height(), mem_type);
  P->Mult(u_T, u_L);

  // L->E
  BlockElementRestriction G(fespace);
  mfem::BlockVector u_E(G.bOffsets(), mem_type);
  G.Gather(u_L, u_E);

  // E->Q
  int dim = u_T.mesh().Dimension();
  auto [qpt_offsets, num_quadrature_points] = quadrature_point_offsets(u_T.mesh(), q);
  uint32_t num_components = uint32_t(fespace->GetVDim());
  u_Q.resize(num_quadrature_points, num_components);
  du_dxi_Q.resize(num_quadrature_points, num_components, dim);
  std::cout << num_quadrature_points << " " << num_components << std::endl;


  // ...
}

}  // namespace serac
