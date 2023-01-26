#include "serac/numerics/functional/geometric_factors.hpp"
#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

template <int Q, Geometry geom, typename function_space>
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

Geometry from_mfem(mfem::Geometry::Type geom)
{
  if (geom == mfem::Geometry::Type::SEGMENT) return Geometry::Segment;
  if (geom == mfem::Geometry::Type::SQUARE) return Geometry::Quadrilateral;
  if (geom == mfem::Geometry::Type::TRIANGLE) return Geometry::Triangle;
  if (geom == mfem::Geometry::Type::CUBE) return Geometry::Hexahedron;
  if (geom == mfem::Geometry::Type::TETRAHEDRON) return Geometry::Tetrahedron;
  return Geometry::Point;
}

GeometricFactors::GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type elem_geom)
{
  auto* nodes = mesh->GetNodes();
  auto* fes   = nodes->FESpace();

  auto         restriction = serac::ElementRestriction(fes, elem_geom);
  mfem::Vector X_e(int(restriction.ESize()));
  restriction.Gather(*nodes, X_e);

  // assumes all elements are the same order
  int p = fes->GetElementOrder(0);

  Geometry g = from_mfem(elem_geom);

  int spatial_dim   = mesh->SpaceDimension();
  int geometry_dim  = dimension_of(g);
  int qpts_per_elem = num_quadrature_points(g, q);

  // NB: we only want the number of elements with the specified
  // geometry, which is not the same as mesh->GetNE() in general
  int num_elements = int(restriction.dof_info.shape()[0]);

  X = mfem::Vector(num_elements * qpts_per_elem * spatial_dim);
  J = mfem::Vector(num_elements * qpts_per_elem * spatial_dim * geometry_dim);

#define DISPATCH_KERNEL(GEOM, P, Q)                                                                             \
  if (g == Geometry::GEOM && p == P && q == Q) {                                                                \
    compute_geometric_factors<Q, Geometry::GEOM, H1<P, dimension_of(Geometry::GEOM)> >(X, J, X_e,               \
                                                                                       uint32_t(num_elements)); \
    return;                                                                                                     \
  }

  DISPATCH_KERNEL(Quadrilateral, 1, 1);
  DISPATCH_KERNEL(Quadrilateral, 1, 2);
  DISPATCH_KERNEL(Quadrilateral, 1, 3);
  DISPATCH_KERNEL(Quadrilateral, 1, 4);

  DISPATCH_KERNEL(Quadrilateral, 2, 1);
  DISPATCH_KERNEL(Quadrilateral, 2, 2);
  DISPATCH_KERNEL(Quadrilateral, 2, 3);
  DISPATCH_KERNEL(Quadrilateral, 2, 4);

  DISPATCH_KERNEL(Quadrilateral, 3, 1);
  DISPATCH_KERNEL(Quadrilateral, 3, 2);
  DISPATCH_KERNEL(Quadrilateral, 3, 3);
  DISPATCH_KERNEL(Quadrilateral, 3, 4);

  DISPATCH_KERNEL(Hexahedron, 1, 1);
  DISPATCH_KERNEL(Hexahedron, 1, 2);
  DISPATCH_KERNEL(Hexahedron, 1, 3);
  DISPATCH_KERNEL(Hexahedron, 1, 4);

  DISPATCH_KERNEL(Hexahedron, 2, 1);
  DISPATCH_KERNEL(Hexahedron, 2, 2);
  DISPATCH_KERNEL(Hexahedron, 2, 3);
  DISPATCH_KERNEL(Hexahedron, 2, 4);

  DISPATCH_KERNEL(Hexahedron, 3, 1);
  DISPATCH_KERNEL(Hexahedron, 3, 2);
  DISPATCH_KERNEL(Hexahedron, 3, 3);
  DISPATCH_KERNEL(Hexahedron, 3, 4);

#undef DISPATCH_KERNEL

  std::cout << "should never be reached" << std::endl;
}

GeometricFactors::GeometricFactors(const mfem::Mesh* mesh, int q, mfem::Geometry::Type elem_geom, FaceType type)
{
  auto* nodes = mesh->GetNodes();
  auto* fes   = nodes->FESpace();

  auto         restriction = serac::ElementRestriction(fes, elem_geom, type);
  mfem::Vector X_e(int(restriction.ESize()));
  restriction.Gather(*nodes, X_e);

  // assumes all elements are the same order
  int p = fes->GetElementOrder(0);

  Geometry g = from_mfem(elem_geom);

  int spatial_dim   = mesh->SpaceDimension();
  int geometry_dim  = dimension_of(g);
  int qpts_per_elem = num_quadrature_points(g, q);

  // NB: we only want the number of elements with the specified
  // geometry, which is not the same as mesh->GetNE() in general
  int num_elements = int(restriction.dof_info.shape()[0]);

  X = mfem::Vector(num_elements * qpts_per_elem * spatial_dim);
  J = mfem::Vector(num_elements * qpts_per_elem * spatial_dim * geometry_dim);

#define DISPATCH_KERNEL(GEOM, P, Q)                                                                                 \
  if (g == Geometry::GEOM && p == P && q == Q) {                                                                    \
    compute_geometric_factors<Q, Geometry::GEOM, H1<P, dimension_of(Geometry::GEOM) + 1> >(X, J, X_e,               \
                                                                                           uint32_t(num_elements)); \
    return;                                                                                                         \
  }

  DISPATCH_KERNEL(Segment, 1, 1);
  DISPATCH_KERNEL(Segment, 1, 2);
  DISPATCH_KERNEL(Segment, 1, 3);
  DISPATCH_KERNEL(Segment, 1, 4);

  DISPATCH_KERNEL(Segment, 2, 1);
  DISPATCH_KERNEL(Segment, 2, 2);
  DISPATCH_KERNEL(Segment, 2, 3);
  DISPATCH_KERNEL(Segment, 2, 4);

  DISPATCH_KERNEL(Segment, 3, 1);
  DISPATCH_KERNEL(Segment, 3, 2);
  DISPATCH_KERNEL(Segment, 3, 3);
  DISPATCH_KERNEL(Segment, 3, 4);

  DISPATCH_KERNEL(Quadrilateral, 1, 1);
  DISPATCH_KERNEL(Quadrilateral, 1, 2);
  DISPATCH_KERNEL(Quadrilateral, 1, 3);
  DISPATCH_KERNEL(Quadrilateral, 1, 4);

  DISPATCH_KERNEL(Quadrilateral, 2, 1);
  DISPATCH_KERNEL(Quadrilateral, 2, 2);
  DISPATCH_KERNEL(Quadrilateral, 2, 3);
  DISPATCH_KERNEL(Quadrilateral, 2, 4);

  DISPATCH_KERNEL(Quadrilateral, 3, 1);
  DISPATCH_KERNEL(Quadrilateral, 3, 2);
  DISPATCH_KERNEL(Quadrilateral, 3, 3);
  DISPATCH_KERNEL(Quadrilateral, 3, 4);


#undef DISPATCH_KERNEL

  std::cout << "should never be reached" << std::endl;
}

}  // namespace serac