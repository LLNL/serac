#include "../detail/meta.h"
#include "../tensor.hpp"
#include "../finite_element.hpp"

static constexpr int    num_points     = 10;
static constexpr tensor random_numbers = {
    {-0.886787, -0.850126, 0.464212,  -0.0733101, -0.397738, 0.302355,   -0.570758, 0.977727,  0.282365,  -0.768947,
     0.6216,    0.43598,   -0.696321, 0.92545,    0.183003,  0.121761,   -0.877239, 0.0347577, -0.818463, -0.216474,
     -0.43894,  0.0178874, -0.869944, -0.733499,  0.255124,  -0.0561095, -0.34607,  -0.305958, 0.414472,  -0.744998}};

/*
iterate over the node/direction pairs in the element, and verify that
contributions from other shape functions in the element are orthogonal
*/
template <typename element_type>
void verify_kronecker_delta_property()
{
  static constexpr double tolerance  = 1.0e-13;
  static constexpr tensor nodes      = element_type::nodes;
  static constexpr tensor directions = element_type::directions;
  static constexpr tensor I          = Identity<element_type::ndof>();

  for (int i = 0; i < element_type::ndof; i++) {
    if (norm(I[i] - dot(element_type::shape_functions(nodes[i]), directions[i])) > tolerance) {
      exit(1);
    }
  }
}

/*
  compare the direct curl evaluation to a finite difference approximation
*/
template <typename element_type>
void verify_curl_calculation()
{
  static constexpr double eps = 1.0e-6;
  static constexpr int dim = element_type::dim;
  static constexpr tensor I   = Identity<dim>();
  static constexpr tensor random_points =
      make_tensor<num_points, dim>([](int i, int j) { return random_numbers[i * dim + j]; });

  constexpr auto N     = element_type::shape_functions;
  constexpr auto curlN = element_type::shape_function_curl;

  for (int i = 0; i < num_points; i++) {
    auto x = random_points[i];
    if constexpr (dim == 2) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);

      std::cout << norm(curlN(x) - (dot(dN_dx, I[1]) - dot(dN_dy, I[0]))) / norm(curlN(x)) << std::endl;
    }

    if constexpr (dim == 3) {
      auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
      auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);
      auto dN_dz = (N(x + eps * I[2]) - N(x - eps * I[2])) / (2.0 * eps);

      std::cout << norm(transpose(curlN(x))[0] - (dot(dN_dy, I[2]) - dot(dN_dz, I[1]))) << std::endl;
      std::cout << norm(transpose(curlN(x))[1] - (dot(dN_dz, I[0]) - dot(dN_dx, I[2]))) << std::endl;
      std::cout << norm(transpose(curlN(x))[2] - (dot(dN_dx, I[1]) - dot(dN_dy, I[0]))) << std::endl;
    }
  }

  std::cout << std::endl;
}

int main()
{
  verify_kronecker_delta_property<finite_element<::Geometry::Quadrilateral, Hcurl<1>>>();
  verify_kronecker_delta_property<finite_element<::Geometry::Quadrilateral, Hcurl<2>>>();
  verify_kronecker_delta_property<finite_element<::Geometry::Quadrilateral, Hcurl<3>>>();

  verify_kronecker_delta_property<finite_element<::Geometry::Hexahedron, Hcurl<1>>>();
  verify_kronecker_delta_property<finite_element<::Geometry::Hexahedron, Hcurl<2>>>();
  verify_kronecker_delta_property<finite_element<::Geometry::Hexahedron, Hcurl<3>>>();

  verify_curl_calculation<finite_element<::Geometry::Quadrilateral, Hcurl<1>>>();
  verify_curl_calculation<finite_element<::Geometry::Quadrilateral, Hcurl<2>>>();
  verify_curl_calculation<finite_element<::Geometry::Quadrilateral, Hcurl<3>>>();

  verify_curl_calculation<finite_element<::Geometry::Hexahedron, Hcurl<1>>>();
  verify_curl_calculation<finite_element<::Geometry::Hexahedron, Hcurl<2>>>();
  verify_curl_calculation<finite_element<::Geometry::Hexahedron, Hcurl<3>>>();
}