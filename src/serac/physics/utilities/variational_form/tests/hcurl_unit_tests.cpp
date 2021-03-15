#include "../detail/meta.h"
#include "../tensor.hpp"
#include "../finite_element.hpp"

/*
iterate over the node/direction pairs in the element, and verify that
contributions from other shape functions in the element are orthogonal
*/ 
template < int p >
void verify_kronecker_delta_property() {

  using element_type = finite_element< ::Geometry::Quadrilateral, Hcurl<p> >;

  constexpr auto legendre_nodes = GaussLegendreNodes<p>();
  constexpr auto lobatto_nodes = GaussLobattoNodes<p+1>();

  tensor direction{{1.0, 0.0}};
  for (int j = 0; j < p + 1; j++) {
    for (int i = 0; i < p; i++) {
      tensor node{{legendre_nodes[i], lobatto_nodes[j]}};

      auto N = element_type::shape_functions(node);
      std::cout << chop(dot(N, direction)) << std::endl;
    }
  }

  direction = tensor{{0.0, 1.0}};
  for (int j = 0; j < p; j++) {
    for (int i = 0; i < p + 1; i++) {
      tensor node{{lobatto_nodes[i], legendre_nodes[j]}};

      auto N = element_type::shape_functions(node);
      std::cout << chop(dot(N, direction)) << std::endl;
    }
  }

}


/*
  compare the direct curl evaluation to a finite difference approximation
*/
template < int p >
void verify_curl_calculation() {

  static constexpr double eps = 1.0e-6;

  static constexpr tensor I = Identity<2>();

  static constexpr tensor< double, 2 > random_points[8] = {
    {0.721555, 0.907109}, 
    {0.458141, 0.0644415}, 
    {0.825454, 0.910218}, 
    {0.444205, 0.193282}, 
    {0.59453, 0.121452}, 
    {0.0473772, 0.865351}, 
    {0.537781, 0.528525}, 
    {0.745137, 0.572603}
  };

  using element_type = finite_element< ::Geometry::Quadrilateral, Hcurl<p> >;

  constexpr auto N = element_type::shape_functions;
  constexpr auto curlN = element_type::shape_function_curl;

  for (auto x : random_points) {
    auto dN_dx = (N(x + eps * I[0]) - N(x - eps * I[0])) / (2.0 * eps);
    auto dN_dy = (N(x + eps * I[1]) - N(x - eps * I[1])) / (2.0 * eps);

    std::cout << norm(curlN(x) - (dot(dN_dx, I[1]) - dot(dN_dy, I[0]))) << std::endl;
  }

  std::cout << std::endl;

}

int main() {

  verify_kronecker_delta_property<1>();
  verify_kronecker_delta_property<2>();
  verify_kronecker_delta_property<3>();

  verify_curl_calculation<1>();
  verify_curl_calculation<2>();
  verify_curl_calculation<3>();
}