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
                        finite_element< geom, H1Dynamic > element,
                        QuadratureRule rule) {

  uint32_t num_elements = u_E.shape[0];
  uint32_t components = u_E.shape[1];
  uint32_t qpts_per_elem = u_Q.shape[0] / num_elements;

  ndview<double,2> qpts(&rule.points[0], {uint32_t(rule.points.size()), 1});

  auto [Bsize, Gsize] = element.evaluate_shape_functions_buffer_size(qpts);
  std::vector< double > B_buffer(Bsize);
  std::vector< double > G_buffer(Gsize);

  ndview<double,2> B(&B_buffer[0], {});
  ndview<double,2> G(&G_buffer[0], {});
  element.evaluate_shape_functions(B, G, qpts);

  std::vector<double> buffer(element.interpolate_buffer_size(qpts_per_elem));
  for (uint32_t e = 0; e < num_elements; e++) {
    for (uint32_t i = 0; i < components; i++) {
      element.interpolate(u_Q(e, i), du_dxi_Q(e, i), u_E(e, i), B, G, &buffer[0]);
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
  uint32_t dim = uint32_t(u_T.mesh().Dimension());
  auto [element_counts, qpt_offsets, num_qpts] = quadrature_point_offsets(u_T.mesh(), q);
  uint32_t num_components = uint32_t(fespace->GetVDim());
  u_Q.resize(num_qpts, num_components);
  du_dxi_Q.resize(num_qpts, num_components, dim);
  uint32_t p = uint32_t(fespace->GetMaxElementOrder());
  std::cout << num_qpts << " " << num_components << std::endl;

  // for (auto geom : supported_geometries) {
  if (element_counts[mfem::Geometry::SQUARE] > 0) {
    constexpr auto geom = mfem::Geometry::SQUARE;
    uint32_t offset = qpt_offsets[uint32_t(geom)];
    uint32_t qpts_per_elem = uint32_t(num_quadrature_points(geom, int(q)));
    uint32_t nelems = element_counts[uint32_t(geom)];
    finite_element< geom, H1Dynamic > element{p+1};

    interpolate_kernel(
      ndview< double, 3 >(&u_Q(offset, 0), {nelems, num_components, qpts_per_elem}),
      ndview< double, 4 >(&du_dxi_Q(offset, 0, 0), {nelems, num_components, dim, qpts_per_elem}),
      ndview< double, 3 >(&u_E(geom), {nelems, num_components, element.ndof()}),
      element,
      GaussLegendreRule(geom, PointsPerDimension{q})
    );
  }

  // ...
}

}  // namespace serac
