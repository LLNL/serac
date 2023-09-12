#include "interface.hpp"

#include "serac/numerics/functional/geometry.hpp"
#include "serac/numerics/functional/element_restriction.hpp"

#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/refactor/ndview.hpp"

namespace serac {

template <mfem::Geometry::Type geom>
void interpolate_kernel(ndview< double, 3 > u_Q, 
                        ndview< double, 4 > du_dxi_Q, 
                        ndview < double, 3 > u_E, 
                        uint32_t p, 
                        QuadratureRule rule) {

  uint32_t num_elements = u_E.shape[0];
  uint32_t components = u_E.shape[1];
  uint32_t qpts_per_elem = u_Q.shape[0] / num_elements;

  finite_element< geom, H1Dynamic > element{p + 1};

  std::vector< double > B_storage(qpts_per_elem * (p + 1));
  std::vector< double > G_storage(qpts_per_elem * (p + 1));
  ndview<double,2> B(&B_storage[0], {qpts_per_elem, (p + 1)});
  ndview<double,2> G(&G_storage[0], {qpts_per_elem, (p + 1)});

  std::vector<double> buffer_storage(element.buffer_size(qpts_per_elem));
  ndview<double,2> qpts(&rule.points[0], {uint32_t(rule.points.size()), 1});

  element.evaluate_shape_functions(B, G, qpts);

  for (uint32_t e = 0; e < num_elements; e++) {
    for (uint32_t i = 0; i < components; i++) {
      element.interpolate(
        ndview<double,1>(&u_Q(e, i, 0), {u_Q.shape[2]}), 
        ndview<double,2>(&du_dxi_Q(e, i, 0, 0), {du_dxi_Q.shape[2], du_dxi_Q.shape[3]}),
        ndview<double,1>(&u_E(e, i, 0), {u_E.shape[2]}), 
        B, G, &buffer_storage[0]
      );
    }
  }
}

void interpolate(axom::Array<double, 2>& u_Q, axom::Array<double, 3>& du_dxi_Q, const FiniteElementState& u_T, uint32_t q)
{
  auto mem_type = mfem::Device::GetMemoryType();
  auto fespace = u_T.gridFunction().ParFESpace();
  
  // variable order spaces aren't supported
  assert(!fespace->IsVariableOrder());

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
  auto [element_counts, qpt_offsets, num_qpts] = quadrature_point_offsets(u_T.mesh(), q);
  uint32_t num_components = uint32_t(fespace->GetVDim());
  u_Q.resize(num_qpts, num_components);
  du_dxi_Q.resize(num_qpts, num_components, dim);
  std::cout << num_qpts << " " << num_components << std::endl;

  // for (auto geom : supported_geometries) {
  if (element_counts[mfem::Geometry::SQUARE] > 0) {
    uint32_t qoffset = qpt_offsets[mfem::Geometry::SQUARE];

    interpolate_kernel<mfem::Geometry::SQUARE>(
      ndview< double, 3 >(&u_Q(qoffset, 0), {}),
      ndview< double, 4 >(&du_dxi_Q(qoffset, 0, 0), {}),
      ndview< double, 3 >(&u_E(mfem::Geometry::SQUARE), {}), 
      uint32_t(fespace->GetMaxElementOrder()),
      GaussLegendreRule(mfem::Geometry::SQUARE, PointsPerDimension{q})
    );
  }

  // ...
}

}  // namespace serac
