#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

using namespace serac;

template < typename T, int ... n >
tensor< T, (n * ...) > flatten(tensor< T, n ... > A) {
  tensor< T, (n * ...) > A_flat;
  auto A_ptr = reinterpret_cast<double *>(&A);
  for (int i = 0; i < (n * ...); i++) {
    A_flat[i] = A_ptr[i];
  }
  return A_flat;
}

template < int ... new_dimensions, typename T, int ... old_dimensions >
tensor< T, new_dimensions ... > reshape(tensor< T, old_dimensions ... > A) {
  static_assert((new_dimensions * ...) == (old_dimensions * ...), 
  "error: can't reshape to configuration with different number of elements");

  tensor< T, new_dimensions ... > A_reshaped;
  auto A_ptr = reinterpret_cast<double *>(&A);
  auto A_reshaped_ptr = reinterpret_cast<double *>(&A_reshaped);
  for (int i = 0; i < (old_dimensions * ...); i++) {
    A_reshaped_ptr[i] = A_ptr[i];
  }
  return A_reshaped;
}

template < typename T, int ... n >
double relative_error(tensor< T, n ... > A, tensor< T, n ... > B) {
  return norm(A - B) / norm(A);
}

template < int q, int p >
auto batched_hcurl_interpolation(tensor< double, 3, p + 1, p + 1, p > element_values) {

  // clang-format off
  union {
    tensor< double, 3, p + 1, p + 1, p > copy;
    struct {
      tensor< double, p + 1, p + 1, p     > element_values_x;
      tensor< double, p + 1, p    , p + 1 > element_values_y;
      tensor< double, p    , p + 1, p + 1 > element_values_z;
    };
  } data;
  // clang-format on

  data.copy = element_values;

  tensor< double, 3, q, q, q > value_q{};
  tensor< double, 3, q, q, q > curl_q{};

  auto xi = GaussLegendreNodes<q>();

  tensor<double, q, p > B1;
  tensor<double, q, p+1 > B2;
  tensor<double, q, p+1 > G2;
  for (int i = 0; i < q; i++) {
    B1[i] = GaussLegendreInterpolation<p>(xi[i]);
    B2[i] = GaussLobattoInterpolation<p+1>(xi[i]);
    G2[i] = GaussLobattoInterpolationDerivative<p+1>(xi[i]);
  }

  tensor< double, p + 1, p + 1, q > A1; 
  tensor< double, 2, p + 1, q, q > A2; 

  /////////////////////////////////
  ////////// x-component //////////
  /////////////////////////////////
  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int i = 0; i < p; i++) {
          sum += B1(qx, i) * data.element_values_x(k, j, i);
        }
        A1(k, j, qx) = sum;
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int j = 0; j < (p + 1); j++) {
          sum[0] += B2(qy, j) * A1(k, j, qx);
          sum[1] += G2(qy, j) * A1(k, j, qx);
        }
        A2(0, k, qy, qx) = sum[0];
        A2(1, k, qy, qx) = sum[1];
      }
    }
  }

  for (int qz = 0; qz < q; qz++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[3]{};
        for (int k = 0; k < (p + 1); k++) {
          sum[0] += B2(qz, k) * A2(0, k, qy, qx);
          sum[1] += G2(qz, k) * A2(0, k, qy, qx);
          sum[2] += B2(qz, k) * A2(1, k, qy, qx);
        }
        value_q(0, qz, qy, qx) += sum[0];
        curl_q(1, qz, qy, qx)  += sum[1];
        curl_q(2, qz, qy, qx)  -= sum[2];
      }
    }
  }

  /////////////////////////////////
  ////////// Y-component //////////
  /////////////////////////////////
  for (int k = 0; k < p + 1; k++) {
    for (int i = 0; i < p + 1; i++) {
      for (int qy = 0; qy < q; qy++) {
        double sum = 0.0;
        for (int j = 0; j < p; j++) {
          sum += B1(qy, j) * data.element_values_y(k, j, i);
        }
        A1(k, i, qy) = sum;
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int i = 0; i < (p + 1); i++) {
          sum[0] += B2(qx, i) * A1(k, i, qy);
          sum[1] += G2(qx, i) * A1(k, i, qy);
        }
        A2(0, k, qy, qx) = sum[0];
        A2(1, k, qy, qx) = sum[1];
      }
    }
  }

  for (int qz = 0; qz < q; qz++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[3]{};
        for (int k = 0; k < (p + 1); k++) {
          sum[0] += B2(qz, k) * A2(0, k, qy, qx);
          sum[1] += G2(qz, k) * A2(0, k, qy, qx);
          sum[2] += B2(qz, k) * A2(1, k, qy, qx);
        }
        value_q(1, qz, qy, qx) += sum[0];
        curl_q(2, qz, qy, qx)  += sum[2];
        curl_q(0, qz, qy, qx)  -= sum[1];
      }
    }
  }

  /////////////////////////////////
  ////////// Z-component //////////
  /////////////////////////////////
  for (int j = 0; j < p + 1; j++) {
    for (int i = 0; i < p + 1; i++) {
      for (int qz = 0; qz < q; qz++) {
        double sum = 0.0;
        for (int k = 0; k < p; k++) {
          sum += B1(qz, k) * data.element_values_z(k, j, i);
        }
        A1(j, i, qz) = sum;
      }
    }
  }

  for (int j = 0; j < p + 1; j++) {
    for (int qz = 0; qz < q; qz++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int i = 0; i < (p + 1); i++) {
          sum[0] += B2(qx, i) * A1(j, i, qz);
          sum[1] += G2(qx, i) * A1(j, i, qz);
        }
        A2(0, j, qz, qx) = sum[0];
        A2(1, j, qz, qx) = sum[1];
      }
    }
  }

  for (int qz = 0; qz < q; qz++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[3]{};
        for (int j = 0; j < (p + 1); j++) {
          sum[0] += B2(qy, j) * A2(0, j, qz, qx);
          sum[1] += G2(qy, j) * A2(0, j, qz, qx);
          sum[2] += B2(qy, j) * A2(1, j, qz, qx);
        }
        value_q(2, qz, qy, qx) += sum[0];
        curl_q(0, qz, qy, qx)  += sum[1];
        curl_q(1, qz, qy, qx)  -= sum[2];
      }
    }
  }

  return tuple{value_q, curl_q};
}

template < int p, int q >
auto batched_hcurl_extrapolation_sf(tensor< double, 3, q, q, q> source,
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

  tensor< double, 2, p + 1, q, q > A1; 
  tensor< double, p + 1, p + 1, q > A2; 

  /////////////////////////////////
  ////////// X-component //////////
  /////////////////////////////////
  for (int k = 0; k < p + 1; k++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int qz = 0; qz < q; qz++) {
          sum[0] += B2(qz, k) * source(0, qz, qy, qx) + G2(qz, k) * flux(1, qz, qy, qx);
          sum[1] -= B2(qz, k) * flux(2, qz, qy, qx);
        }
        A1(0, k, qy, qx) = sum[0];
        A1(1, k, qy, qx) = sum[1];
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int qx = 0; qx < q; qx++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B2(qy, j) * A1(0, k, qy, qx);
          sum += G2(qy, j) * A1(1, k, qy, qx);
        }
        A2(k, j, qx) = sum;
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p; i++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B1(qx, i) * A2(k, j, qx);
        }
        data.element_residual_x(k, j, i) = sum;
      }
    }
  }

  /////////////////////////////////
  ////////// Y-component //////////
  /////////////////////////////////
  for (int k = 0; k < p + 1; k++) {
    for (int qy = 0; qy < q; qy++) {
      for (int qx = 0; qx < q; qx++) {
        double sum[2]{};
        for (int qz = 0; qz < q; qz++) {
          sum[0] += B2(qz, k) * source(1, qz, qy, qx) - G2(qz, k) * flux(0, qz, qy, qx);
          sum[1] += B2(qz, k) * flux(2, qz, qy, qx);
        }
        A1(0, k, qy, qx) = sum[0];
        A1(1, k, qy, qx) = sum[1];
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int i = 0; i < p + 1; i++) {
      for (int qy = 0; qy < q; qy++) {
        double sum = 0.0;
        for (int qx = 0; qx < q; qx++) {
          sum += B2(qx, i) * A1(0, k, qy, qx);
          sum += G2(qx, i) * A1(1, k, qy, qx);
        }
        A2(k, i, qy) = sum;
      }
    }
  }

  for (int k = 0; k < p + 1; k++) {
    for (int j = 0; j < p; j++) {
      for (int i = 0; i < p + 1; i++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B1(qy, j) * A2(k, i, qy);
        }
        data.element_residual_y(k, j, i) = sum;
      }
    }
  }

  /////////////////////////////////
  ////////// Z-component //////////
  /////////////////////////////////
  for (int i = 0; i < p + 1; i++) {
    for (int qz = 0; qz < q; qz++) {
      for (int qy = 0; qy < q; qy++) {
        double sum[2]{};
        for (int qx = 0; qx < q; qx++) {
          sum[0] += B2(qx, i) * source(2, qz, qy, qx) - G2(qx, i) * flux(1, qz, qy, qx);
          sum[1] += B2(qx, i) * flux(0, qz, qy, qx);
        }
        A1(0, i, qz, qy) = sum[0];
        A1(1, i, qz, qy) = sum[1];
      }
    }
  }


  for (int j = 0; j < p + 1; j++) {
    for (int i = 0; i < p + 1; i++) {
      for (int qz = 0; qz < q; qz++) {
        double sum = 0.0;
        for (int qy = 0; qy < q; qy++) {
          sum += B2(qy, j) * A1(0, i, qz, qy);
          sum += G2(qy, j) * A1(1, i, qz, qy);
        }
        A2(j, i, qz) = sum;
      }
    }
  }

  for (int k = 0; k < p; k++) {
    for (int j = 0; j < p + 1; j++) {
      for (int i = 0; i < p + 1; i++) {
        double sum = 0.0;
        for (int qz = 0; qz < q; qz++) {
          sum += B1(qz, k) * A2(j, i, qz);
        }
        data.element_residual_z(k, j, i) = sum;
      }
    }
  }

  return data.element_residual;

}

template < int p, int q >
auto batched_hcurl_extrapolation(tensor< double, 3, q, q, q> source,
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
  constexpr int q = 3;

  using element_type = finite_element< Geometry::Hexahedron, Hcurl<p> >;

  union {
    tensor< double, 3, n, n, p > element_values;
    tensor< double, 3 * n * n * p > element_values_1D;
  };

  element_values = make_tensor< 3, n, n, p >([](int c, int i, int j, int k) {
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

  auto [batched_value_q, batched_curl_q] = batched_hcurl_interpolation<q>(element_values);

  auto batched_source_q = dot(C, batched_value_q);
  auto batched_flux_q = dot(C, batched_curl_q);

  auto element_residual = batched_hcurl_extrapolation_sf<p>(batched_source_q, batched_flux_q);
  auto element_residual_ref = reshape< 3, p + 1, p + 1, p >(element_residual_1D);

  //std::cout << value_q[0] << std::endl;
  //std::cout << batched_value_q[0] << std::endl;
  std::cout << "errors in value: " << std::endl;
  std::cout << relative_error(value_q[0], batched_value_q[0]) << std::endl;
  std::cout << relative_error(value_q[1], batched_value_q[1]) << std::endl;
  std::cout << relative_error(value_q[2], batched_value_q[2]) << std::endl;

  std::cout << "errors in curl: " << std::endl;
  std::cout << relative_error(curl_q[0], batched_curl_q[0]) << std::endl;
  std::cout << relative_error(curl_q[1], batched_curl_q[1]) << std::endl;
  std::cout << relative_error(curl_q[2], batched_curl_q[2]) << std::endl;

  std::cout << "errors in residual: " << std::endl;
  std::cout << relative_error(element_residual[0], element_residual_ref[0]) << std::endl;
  std::cout << relative_error(element_residual[1], element_residual_ref[1]) << std::endl;
  std::cout << relative_error(element_residual[2], element_residual_ref[2]) << std::endl;



}
