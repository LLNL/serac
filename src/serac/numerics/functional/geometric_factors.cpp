#include "serac/numerics/functional/geometric_factors.hpp"
#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

template <int Q, mfem::Geometry::Type geom, typename function_space>
void compute_geometric_factors(mfem::Vector& positions_q, mfem::Vector& jacobians_q, const mfem::Vector& positions_e,
                               uint32_t num_elements)
{
  static constexpr TensorProductQuadratureRule<Q> rule{};

  constexpr int spatial_dim   = function_space::components;
  constexpr int geometry_dim  = dimension_of(geom);
  constexpr int qpts_per_elem = num_quadrature_points(geom, Q);

  using element_type  = finite_element<geom, function_space>;
  using position_type = tensor<double, spatial_dim, qpts_per_elem>;
  using jacobian_type = tensor<double, geometry_dim, spatial_dim, qpts_per_elem>;

  auto X_q = reinterpret_cast<position_type*>(positions_q.ReadWrite());
  auto J_q = reinterpret_cast<jacobian_type*>(jacobians_q.ReadWrite());
  auto X   = reinterpret_cast<const typename element_type::dof_type*>(positions_e.Read());

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    // load the positions for the nodes in this element
    auto X_e = X[e];

    // calculate the values and derivatives (w.r.t. xi) of X at each quadrature point
    auto quadrature_values = element_type::interpolate(X_e, rule);

    // mfem wants to store this data in a different layout, so we have to transpose it
    for (int q = 0; q < qpts_per_elem; q++) {
      auto [value, gradient] = quadrature_values[q];
      for (int i = 0; i < spatial_dim; i++) {
        X_q[e](i, q) = value[i];
        if constexpr (std::is_same_v<decltype(value), decltype(gradient)>) {
          J_q[e](0, i, q) = gradient[i];
        }
        if constexpr (!std::is_same_v<decltype(value), decltype(gradient)>) {
          for (int j = 0; j < geometry_dim; j++) {
            J_q[e](j, i, q) = gradient(i, j);
          }
        }
      }
    }
  }
}

GeometricFactors::GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type g)
{
  auto* nodes = mesh->GetNodes();
  auto* fes   = nodes->FESpace();

  auto         restriction = serac::ElementRestriction(fes, g);
  mfem::Vector X_e(int(restriction.ESize()));
  restriction.Gather(*nodes, X_e);

  // assumes all elements are the same order
  int p = fes->GetElementOrder(0);

  int spatial_dim   = mesh->SpaceDimension();
  int geometry_dim  = dimension_of(g);
  int qpts_per_elem = num_quadrature_points(g, q);

  // NB: we only want the number of elements with the specified
  // geometry, which is not the same as mesh->GetNE() in general
  num_elements = std::size_t(restriction.dof_info.shape()[0]);

  X = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim);
  J = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim * geometry_dim);

#define DISPATCH_KERNEL(GEOM, P, Q)                                                                             \
  if (g == mfem::Geometry::GEOM && p == P && q == Q) {                                                          \
    compute_geometric_factors<Q, mfem::Geometry::GEOM, H1<P, dimension_of(mfem::Geometry::GEOM)> >(X, J, X_e,   \
                                                                                       uint32_t(num_elements)); \
    return;                                                                                                     \
  }

  DISPATCH_KERNEL(TRIANGLE, 1, 1);
  DISPATCH_KERNEL(TRIANGLE, 1, 2);
  DISPATCH_KERNEL(TRIANGLE, 1, 3);
  DISPATCH_KERNEL(TRIANGLE, 1, 4);

  DISPATCH_KERNEL(SQUARE, 1, 1);
  DISPATCH_KERNEL(SQUARE, 1, 2);
  DISPATCH_KERNEL(SQUARE, 1, 3);
  DISPATCH_KERNEL(SQUARE, 1, 4);

  DISPATCH_KERNEL(SQUARE, 2, 1);
  DISPATCH_KERNEL(SQUARE, 2, 2);
  DISPATCH_KERNEL(SQUARE, 2, 3);
  DISPATCH_KERNEL(SQUARE, 2, 4);

  DISPATCH_KERNEL(SQUARE, 3, 1);
  DISPATCH_KERNEL(SQUARE, 3, 2);
  DISPATCH_KERNEL(SQUARE, 3, 3);
  DISPATCH_KERNEL(SQUARE, 3, 4);

  DISPATCH_KERNEL(TETRAHEDRON, 1, 1);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 2);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 3);
  DISPATCH_KERNEL(TETRAHEDRON, 1, 4);

  DISPATCH_KERNEL(CUBE, 1, 1);
  DISPATCH_KERNEL(CUBE, 1, 2);
  DISPATCH_KERNEL(CUBE, 1, 3);
  DISPATCH_KERNEL(CUBE, 1, 4);

  DISPATCH_KERNEL(CUBE, 2, 1);
  DISPATCH_KERNEL(CUBE, 2, 2);
  DISPATCH_KERNEL(CUBE, 2, 3);
  DISPATCH_KERNEL(CUBE, 2, 4);

  DISPATCH_KERNEL(CUBE, 3, 1);
  DISPATCH_KERNEL(CUBE, 3, 2);
  DISPATCH_KERNEL(CUBE, 3, 3);
  DISPATCH_KERNEL(CUBE, 3, 4);

#undef DISPATCH_KERNEL

  std::cout << "should never be reached " << std::endl;
}

GeometricFactors::GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type g, FaceType type)
{
  auto* nodes = mesh->GetNodes();
  auto* fes   = nodes->FESpace();

  auto         restriction = serac::ElementRestriction(fes, g, type);
  mfem::Vector X_e(int(restriction.ESize()));
  restriction.Gather(*nodes, X_e);

  // assumes all elements are the same order
  int p = fes->GetElementOrder(0);

  int spatial_dim   = mesh->SpaceDimension();
  int geometry_dim  = dimension_of(g);
  int qpts_per_elem = num_quadrature_points(g, q);

  // NB: we only want the number of elements with the specified
  // geometry, which is not the same as mesh->GetNE() in general
  num_elements = std::size_t(restriction.dof_info.shape()[0]);

  X = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim);
  J = mfem::Vector(int(num_elements) * qpts_per_elem * spatial_dim * geometry_dim);

#define DISPATCH_KERNEL(GEOM, P, Q)                                                                     \
  if (g == mfem::Geometry::GEOM && p == P && q == Q) {                                                  \
    compute_geometric_factors<Q, mfem::Geometry::GEOM, H1<P, dimension_of(mfem::Geometry::GEOM) + 1> >( \
        X, J, X_e, uint32_t(num_elements));                                                             \
    return;                                                                                             \
  }

  DISPATCH_KERNEL(SEGMENT, 1, 1);
  DISPATCH_KERNEL(SEGMENT, 1, 2);
  DISPATCH_KERNEL(SEGMENT, 1, 3);
  DISPATCH_KERNEL(SEGMENT, 1, 4);

  DISPATCH_KERNEL(SEGMENT, 2, 1);
  DISPATCH_KERNEL(SEGMENT, 2, 2);
  DISPATCH_KERNEL(SEGMENT, 2, 3);
  DISPATCH_KERNEL(SEGMENT, 2, 4);

  DISPATCH_KERNEL(SEGMENT, 3, 1);
  DISPATCH_KERNEL(SEGMENT, 3, 2);
  DISPATCH_KERNEL(SEGMENT, 3, 3);
  DISPATCH_KERNEL(SEGMENT, 3, 4);

  DISPATCH_KERNEL(TRIANGLE, 1, 1);
  DISPATCH_KERNEL(TRIANGLE, 1, 2);
  DISPATCH_KERNEL(TRIANGLE, 1, 3);
  DISPATCH_KERNEL(TRIANGLE, 1, 4);

  DISPATCH_KERNEL(TRIANGLE, 2, 1);
  DISPATCH_KERNEL(TRIANGLE, 2, 2);
  DISPATCH_KERNEL(TRIANGLE, 2, 3);
  DISPATCH_KERNEL(TRIANGLE, 2, 4);

  DISPATCH_KERNEL(TRIANGLE, 3, 1);
  DISPATCH_KERNEL(TRIANGLE, 3, 2);
  DISPATCH_KERNEL(TRIANGLE, 3, 3);
  DISPATCH_KERNEL(TRIANGLE, 3, 4);

  DISPATCH_KERNEL(SQUARE, 1, 1);
  DISPATCH_KERNEL(SQUARE, 1, 2);
  DISPATCH_KERNEL(SQUARE, 1, 3);
  DISPATCH_KERNEL(SQUARE, 1, 4);

  DISPATCH_KERNEL(SQUARE, 2, 1);
  DISPATCH_KERNEL(SQUARE, 2, 2);
  DISPATCH_KERNEL(SQUARE, 2, 3);
  DISPATCH_KERNEL(SQUARE, 2, 4);

  DISPATCH_KERNEL(SQUARE, 3, 1);
  DISPATCH_KERNEL(SQUARE, 3, 2);
  DISPATCH_KERNEL(SQUARE, 3, 3);
  DISPATCH_KERNEL(SQUARE, 3, 4);

#undef DISPATCH_KERNEL

  std::cout << "should never be reached" << std::endl;
}

}  // namespace serac
