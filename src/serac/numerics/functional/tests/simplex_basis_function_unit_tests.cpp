#include <gtest/gtest.h>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/numerics/functional/detail/metaprogramming.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/finite_element.hpp"

using namespace serac;

using vec2 = tensor<double, 2>;
using vec3 = tensor<double, 3>;

static constexpr double kronecker_tolerance = 1.0e-16;
static constexpr double grad_tolerance      = 1.0e-10;  // coarser, since comparing to a finite difference approximation
static constexpr int    num_points          = 10;
//static constexpr tensor random_numbers      = {
//    {-0.886787, -0.850126, 0.464212,  -0.0733101, -0.397738, 0.302355,   -0.570758, 0.977727,  0.282365,  -0.768947,
//     0.6216,    0.43598,   -0.696321, 0.92545,    0.183003,  0.121761,   -0.877239, 0.0347577, -0.818463, -0.216474,
//     -0.43894,  0.0178874, -0.869944, -0.733499,  0.255124,  -0.0561095, -0.34607,  -0.305958, 0.414472,  -0.744998}};

template < typename element_type >
auto lagrange_nodes() {
  if constexpr (element_type::geometry == mfem::Geometry::TRIANGLE && element_type::n == 2) {
    return std::vector< tensor< double, 2 > >{{0, 0}, {1, 0}, {0, 1}};
  }

  if constexpr (element_type::geometry == mfem::Geometry::TRIANGLE && element_type::n == 3) {
    return std::vector< tensor< double, 2 > >{{0,0},{0.5,0},{1,0},{0,0.5},{0.5,0.5},{0,1}};
  }

  if constexpr (element_type::geometry == mfem::Geometry::TRIANGLE && element_type::n == 4) {
    return std::vector< tensor< double, 2 > >{{0,0},{0.2763932022500210303590826,0},{0.7236067977499789696409174,0},{1.,0},{0,0.2763932022500210303590826},{0.3333333333333333333333333,0.3333333333333333333333333},{0.7236067977499789696409174,0.2763932022500210303590826},{0,0.7236067977499789696409174},{0.2763932022500210303590826,0.7236067977499789696409174},{0,1.}};
  }


  if constexpr (element_type::geometry == mfem::Geometry::TETRAHEDRON && element_type::n == 2) {
    return std::vector< tensor< double, 3 > >{{0,0,0},{1.,0,0},{0,1.,0},{0,0,1.}};
  }

  if constexpr (element_type::geometry == mfem::Geometry::TETRAHEDRON && element_type::n == 3) {
    return std::vector< tensor< double, 3 > >{{0,0,0},{0.5,0,0},{1.,0,0},{0,0.5,0},{0.5,0.5,0},{0,1.,0},{0,0,0.5},{0.5,0,0.5},{0,0.5,0.5},{0,0,1.}};
  }

  if constexpr (element_type::geometry == mfem::Geometry::TETRAHEDRON && element_type::n == 4) {
    return std::vector< tensor< double, 3 > >{{0,0,0},{0.2763932022500210303590826,0,0},{0.7236067977499789696409174,0,0},{1.,0,0},{0,0.2763932022500210303590826,0},{0.3333333333333333333333333,0.3333333333333333333333333,0},{0.7236067977499789696409174,0.2763932022500210303590826,0},{0,0.7236067977499789696409174,0},{0.2763932022500210303590826,0.7236067977499789696409174,0},{0,1.,0},{0,0,0.2763932022500210303590826},{0.3333333333333333333333333,0,0.3333333333333333333333333},{0.7236067977499789696409174,0,0.2763932022500210303590826},{0,0.3333333333333333333333333,0.3333333333333333333333333},{0.3333333333333333333333333,0.3333333333333333333333333,0.3333333333333333333333333},{0,0.7236067977499789696409174,0.2763932022500210303590826},{0,0,0.7236067977499789696409174},{0.2763932022500210303590826,0,0.7236067977499789696409174},{0,0.2763932022500210303590826,0.7236067977499789696409174},{0,0,1.}};
  }

}

template < typename element_type >
void verify_kronecker_delta_property(double tolerance)
{
  auto nodes = lagrange_nodes< element_type >();
  auto I     = DenseIdentity<element_type::ndof>();

  for (int i = 0; i < element_type::ndof; i++) {
    double error = norm(I[i] - element_type::shape_functions(nodes[i]));
    EXPECT_NEAR(error, 0.0, tolerance);
  }
}

TEST(LinearTriangle, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TRIANGLE, H1<1> > >(1.0e-16);
}

TEST(QuadraticTriangle, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TRIANGLE, H1<2> > >(1.0e-16);
}

TEST(CubicTriangle, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TRIANGLE, H1<3> > >(5.0e-16);
}

TEST(LinearTetrahedron, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TETRAHEDRON, H1<1> > >(1.0e-16);
}

TEST(QuadraticTetrahedron, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TETRAHEDRON, H1<2> > >(1.0e-16);
}

TEST(CubicTetrahedron, KroneckerDeltaProperty) {
  verify_kronecker_delta_property< finite_element< mfem::Geometry::TETRAHEDRON, H1<3> > >(7.0e-16);
}

template < typename element_type >
void verify_basis_function_gradients(double tolerance)
{
  constexpr double eps = 1.0e-7;
  constexpr int dim = element_type::dim;
  using vec = tensor<double, dim>;

  std::vector< vec > random_points;
  if constexpr (dim == 2) {
    random_points = {
      {0.035186, 0.850529}, {0.261589, 0.496929}, 
      {0.0593282, 0.618562}, {0.488774, 0.0887839}, 
      {0.259885, 0.482077}, {0.592555, 0.293245}, 
      {0.228267, 0.123783}, {0.459186, 0.182878}, 
      {0.299645, 0.35773}, {0.160527, 0.54005}
    };
  }
  if constexpr (dim == 3) {
    random_points = {
      {0.142305,0.293363,0.375573}, {0.380177,0.476144,0.0925416},
      {0.0959899,0.223993,0.551841}, {0.0984588,0.376485,0.444559},
      {0.385216,0.132315,0.0977731}, {0.134371,0.472076,0.129275},
      {0.268994,0.179671,0.140298}, {0.0543239,0.532272,0.0826736},
      {0.284743,0.446908,0.059974}, {0.542072,0.247396,0.0856298}
    };
  }
  
  auto I = DenseIdentity<dim>();

  auto phi = element_type::shape_function;
  auto dphi_dxi = element_type::shape_function_gradient;

  double max_error = 0;
  for (auto point : random_points) {
    for (int i = 0; i < element_type::ndof; i++) {
      auto grad = dphi_dxi(point, i);
      for (int j = 0; j < dim; j++) {
        double fd = (phi(point + eps * I[j], i) - phi(point - eps * I[j], i)) / (2 * eps); 
        max_error = std::max(max_error, std::abs(grad[j] - fd));
      }
    }
  }
  EXPECT_NEAR(max_error, 0.0, tolerance);
}

TEST(LinearTriangle, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TRIANGLE, H1<1> > >(1.0e-9);
}

TEST(QuadraticTriangle, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TRIANGLE, H1<2> > >(2.0e-9);
}

TEST(CubicTriangle, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TRIANGLE, H1<3> > >(3.0e-9);
}

TEST(LinearTetrahedron, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TETRAHEDRON, H1<1> > >(1.0e-9);
}

TEST(QuadraticTetrahedron, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TETRAHEDRON, H1<2> > >(2.0e-9);
}

TEST(CubicTetrahedron, Gradients) {
  verify_basis_function_gradients< finite_element< mfem::Geometry::TETRAHEDRON, H1<3> > >(3.0e-9);
}

//template < typename element_type >
//void verify_mass_matrix_integration()
//{
//  element_type::integrate();
//}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  return RUN_ALL_TESTS();
}
