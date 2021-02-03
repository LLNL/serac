#include "detail/meta.h"
#include "tensor.hpp"
#include "finite_element.hpp"


/*
iterate over the node/direction pairs in the element, and verify that
contributions from other shape functions in the element are orthogonal
*/ 
template < PolynomialDegree degree >
void verify_kronecker_delta_property() {

  using element_type = finite_element< ::Geometry::Quadrilateral, Family::HCURL, degree >;

  constexpr int p = static_cast<int>(degree);
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

int main() {
  verify_kronecker_delta_property<PolynomialDegree::Linear>();
  verify_kronecker_delta_property<PolynomialDegree::Quadratic>();
  verify_kronecker_delta_property<PolynomialDegree::Cubic>();
}