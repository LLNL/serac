#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"

using namespace serac;

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
  ////////// Y-component //////////
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
    return (c == 1) * sin(c + i - j + 3 * k);
  });

  // values of the interpolated field and its curl, at each quadrature point
  tensor< double, 3, q, q, q > value_q{};
  tensor< double, 3, q, q, q > curl_q{};
  
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
      }
    }
  }

  auto [batched_value_q, batched_curl_q] = batched_hcurl_interpolation<q>(element_values);

  //std::cout << value_q[0] << std::endl;
  //std::cout << batched_value_q[0] << std::endl;
  std::cout << "errors in value: " << std::endl;
  std::cout << norm(value_q[0] - batched_value_q[0]) << std::endl;
  std::cout << norm(value_q[1] - batched_value_q[1]) << std::endl;
  std::cout << norm(value_q[2] - batched_value_q[2]) << std::endl;

  std::cout << "errors in curl: " << std::endl;
  std::cout << norm(curl_q[0] - batched_curl_q[0]) << std::endl;
  std::cout << norm(curl_q[1] - batched_curl_q[1]) << std::endl;
  std::cout << norm(curl_q[2] - batched_curl_q[2]) << std::endl;

}
