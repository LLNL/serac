#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

using namespace serac;

template < int p, int q >
auto batched_hcurl_extrapolation_naive(tensor< double, 3, q, q, q> source,
                                       tensor< double, 3, q, q, q> flux) {

  // clang-format off
  union {
    tensor< double, 3, p + 1, p + 1, p > element_residual;
    struct {
      tensor< double, p + 1, p + 1, p     > element_residual_x;
      tensor< double, p + 1, p    , p + 1 > element_residual_y;
      tensor< double, p    , p + 1, p + 1 > element_residual_z;
    };
  } data{};
  // clang-format on

  auto xi = GaussLegendreNodes<q>();

  tensor<double, q, p > B1;
  tensor<double, q, p+1 > B2;
  tensor<double, q, p+1 > G2;
  for (int i = 0; i < q; i++) {
    B1[i] = GaussLegendreInterpolation<p>(xi[i]);
    B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
    G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
  }

  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {

        for (int u = 0; u < q; u++) {
          for (int v = 0; v < q; v++) {
            for (int w = 0; w < q; w++) {
              data.element_residual_x(k, j, i) += 
                + B1(u, i) * B2(v, j) * B2(w, k) * source(0, w, v, u)
                + B1(u, i) * B2(v, j) * G2(w, k) *   flux(1, w, v, u)
                - B1(u, i) * G2(v, j) * B2(w, k) *   flux(2, w, v, u);
            }
          }
        }

      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {

        for (int u = 0; u < q; u++) {
          for (int v = 0; v < q; v++) {
            for (int w = 0; w < q; w++) {
              data.element_residual_y(k, j, i) += 
                + B2(u, i) * B1(v, j) * B2(w, k) * source(1, w, v, u)
                + G2(u, i) * B1(v, j) * B2(w, k) *   flux(2, w, v, u)
                - B2(u, i) * B1(v, j) * G2(w, k) *   flux(0, w, v, u);
            }
          }
        }

      }
    }
  }

  for (int k = 0; k < p; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
 
        for (int u = 0; u < q; u++) {
          for (int v = 0; v < q; v++) {
            for (int w = 0; w < q; w++) {
              data.element_residual_z(k, j, i) += 
                + B2(u, i) * B2(v, j) * B1(w, k) * source(2, w, v, u)
                + B2(u, i) * G2(v, j) * B1(w, k) *   flux(0, w, v, u)
                - G2(u, i) * B2(v, j) * B1(w, k) *   flux(1, w, v, u);
            }
          }
        }

      }
    }
  }

  return data.element_residual;

}

int main() {

  constexpr int p = 2;
  constexpr int n = p + 1;
  constexpr int q = 4;

  using element_type = finite_element< Geometry::Hexahedron, Hcurl<p> >;

  union {
    element_type::mfem_dof_layout element_values;
    tensor< double, 3 * n * n * p > element_values_1D;
    tensor< double, 3, n, n, p > element_values_4D;
  };

  element_values_4D = make_tensor< 3, n, n, p >([](int c, int i, int j, int k) {
    return sin(c + i - j + 3 * k);
  });

  tensor< double, 3, 3 > C = {{
    {1.0, 2.0, 3.0},
    {4.0, 7.0, 1.0},
    {2.0, 2.0, 8.0},
  }};

  // values of the interpolated field and its curl, at each quadrature point
  tensor< double, 3, q, q, q > value_q{};
  tensor< double, 3, q, q, q > curl_q{};
  tensor< double, 3 * n * n * p > element_residual_1D{};
  
  auto x1D = GaussLegendreNodes<q>();
  
  for (int k = 0; k < q; k++) {
    for (int j = 0; j < q; j++) {
      for (int i = 0; i < q; i++) {
        tensor xi = {{x1D[i], x1D[j], x1D[k]}};
        auto value = dot(element_values_1D, element_type::shape_functions(xi));
        auto curl = dot(element_values_1D, element_type::shape_function_curl(xi));

        for (int c = 0; c < 3; c++) {
          value_q(c, k, j, i) = value[c];
          curl_q(c, k, j, i) = curl[c];
        }

        auto source = dot(C, value);
        auto flux = dot(C, curl);

        element_residual_1D += dot(element_type::shape_functions(xi), source);
        element_residual_1D += dot(element_type::shape_function_curl(xi), flux);
      }
    }
  }

  tensor< double, p + 1, p + 1, q > A1; 
  tensor< double, 2, p + 1, q, q > A2; 
  TensorProductQuadratureRule<q> rule;

  tensor< double, 3, q, q, q > batched_value_q;
  tensor< double, 3, q, q, q > batched_curl_q;
  element_type::interpolation(element_values, rule, A1, A2, batched_value_q, batched_curl_q);

  auto batched_source_q = dot(C, batched_value_q);
  auto batched_flux_q = dot(C, batched_curl_q);

  element_type::mfem_dof_layout element_residual;
  element_type::extrapolation(batched_source_q, batched_flux_q, rule, A2, A1, element_residual);
  auto element_residual_ref = reshape< 3, p + 1, p + 1, p >(element_residual_1D);

  std::cout << "errors in value: " << std::endl;
  std::cout << relative_error(value_q[0], batched_value_q[0]) << std::endl;
  std::cout << relative_error(value_q[1], batched_value_q[1]) << std::endl;
  std::cout << relative_error(value_q[2], batched_value_q[2]) << std::endl;

  std::cout << "errors in curl: " << std::endl;
  std::cout << relative_error(curl_q[0], batched_curl_q[0]) << std::endl;
  std::cout << relative_error(curl_q[1], batched_curl_q[1]) << std::endl;
  std::cout << relative_error(curl_q[2], batched_curl_q[2]) << std::endl;

  std::cout << "errors in residual: " << std::endl;
  std::cout << relative_error(flatten(element_residual.x), flatten(element_residual_ref[0])) << std::endl;
  std::cout << relative_error(flatten(element_residual.y), flatten(element_residual_ref[1])) << std::endl;
  std::cout << relative_error(flatten(element_residual.z), flatten(element_residual_ref[2])) << std::endl;

}
