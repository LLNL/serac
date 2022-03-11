#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

namespace serac {

template < typename S, typename T >
struct value_and_gradient { S value; T gradient; };

template <Geometry g, int Q>
struct GaussLegendreRule;

template <int Q>
struct GaussLegendreRule<Geometry::Hexahedron, Q> {
  static constexpr auto points_1D  = GaussLegendreNodes<Q>();
  static constexpr auto weights_1D = GaussLegendreWeights<Q>();

  static constexpr double weight(int qx, int qy, int qz) { return weights_1D[qx] * weights_1D[qy] * weights_1D[qz]; }

  __host__ __device__ static constexpr double point(int i) { return GaussLegendreNodes<Q>()(i); }

  static constexpr int size() { return Q * Q * Q; }
};

__host__ __device__ void print(double value) { printf("%f", value); }

template <int m, int... n>
__host__ __device__ void print(const tensor<double, m, n...>& A)
{
  printf("{");
  print(A[0]);
  for (int i = 1; i < m; i++) {
    printf(",");
    print(A[i]);
  }
  printf("}");
}

template < typename trial_space, Geometry geom, int q >
auto BatchPreprocess(const mfem::DeviceTensor< 4, const double > & element_values, GaussLegendreRule<geom, q> rule, int e) {
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {

    tensor<double, q, n> B{};
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
    auto BT = transpose(B);
    auto GT = transpose(G);

    __shared__ tensor<double, 2, q, q, n> A1;
    for (int u = threadIdx.x; u < q; u += blockDim.x) {
      for (int v = threadIdx.y; v < q; v += blockDim.y) {
        for (int w = threadIdx.z; w < n; w += blockDim.z) {
          A1(0, u, v, w) = 0.0;
          A1(1, u, v, w) = 0.0;
        }
      }
    }

    // A1(u, v, k) := B(w, k) * f(u, v, w)
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
      for (int v = threadIdx.y; v < n; v += blockDim.y) {
        for (int w = threadIdx.z; w < n; w += blockDim.z) {
          auto value = element_values(u, v, w, e);
          for (int k = 0; k < n; k++) {
            atomicAdd(&A1(0, u, v, k), B(w, k) * value);
            atomicAdd(&A1(1, u, v, k), G(w, k) * value);
          }
        }
      }
    }
    __syncthreads();

    for (int iz = 0; iz < n; ++iz) {
      tensor< value_and_gradient< double, tensor< double, 2 > >, q, q> interpolated_in_XY{};
      for (int iy = 0; iy < n; ++iy) {
        tensor< value_and_gradient< double, double >, q> interpolated_in_X{};
        for (int ix = 0; ix < n; ++ix) {
          const double s = u_e(ix, iy, iz, e);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_X[qx].value += s * BT(ix, qx);
            interpolated_in_X[qx].gradient += s * GT(ix, qx);
          }
        }
        for (int qy = 0; qy < q; ++qy) {
          const double interpolate_in_Y = BT(iy, qy);
          const double differentiate_in_Y = GT(iy, qy);
          for (int qx = 0; qx < q; ++qx) {
            interpolated_in_XY[qy][qx].value       += interpolated_in_X[qx].value    * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[0] += interpolated_in_X[qx].gradient * interpolate_in_Y;
            interpolated_in_XY[qy][qx].gradient[1] += interpolated_in_X[qx].value    * differentiate_in_Y;
          }
        }
      }
      for (int qz = 0; qz < q; ++qz) {
        const double interpolate_in_Z = BT(iz, qz);
        const double differentiate_in_Z = GT(iz, qz);
        for (int qy = 0; qy < q; ++qy) {
          for (int qx = 0; qx < q; ++qx) {
            u_q[qz][qy][qx].value       += interpolated_in_XY[qy][qx].value       * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[0] += interpolated_in_XY[qy][qx].gradient[0] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[1] += interpolated_in_XY[qy][qx].gradient[1] * interpolate_in_Z;
            u_q[qz][qy][qx].gradient[2] += interpolated_in_XY[qy][qx].value       * differentiate_in_Z;
          }
        }
      }

    }

    return u_q;

  }

}

template <typename trial_space, Geometry geom, int q, typename T>
__device__ void BatchPostprocessCUDA(const T& f, GaussLegendreRule<geom, q>, mfem::DeviceTensor<4, double> r_e, int e)
{
  if constexpr (geom == Geometry::Hexahedron) {

    constexpr GaussLegendreRule<geom, q> rule;
    static constexpr int                 n = trial_space::order + 1;

    tensor<double, q, n> B{};
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }

    __shared__ tensor<double, 4, q, q, n> A1;
    for (int u = threadIdx.x; u < q; u += blockDim.x) {
      for (int v = threadIdx.y; v < q; v += blockDim.y) {
        for (int w = threadIdx.z; w < n; w += blockDim.z) {
          A1(0, u, v, w) = 0.0;
          A1(1, u, v, w) = 0.0;
          A1(2, u, v, w) = 0.0;
          A1(3, u, v, w) = 0.0;
        }
      }
    }
    __syncthreads();

    // A1(u, v, k) := B(w, k) * f(u, v, w)
    for (int u = threadIdx.x; u < q; u += blockDim.x) {
      for (int v = threadIdx.y; v < q; v += blockDim.y) {
        for (int w = threadIdx.z; w < q; w += blockDim.z) {
          for (int k = 0; k < n; k++) {
            atomicAdd(&A1(0, u, v, k), B(w, k) * serac::get<0>(f));
            atomicAdd(&A1(1, u, v, k), B(w, k) * serac::get<1>(f)[0]);
            atomicAdd(&A1(2, u, v, k), B(w, k) * serac::get<1>(f)[1]);
            atomicAdd(&A1(3, u, v, k), G(w, k) * serac::get<1>(f)[2]);
          }
        }
      }
    }
    __syncthreads();

    __shared__ tensor<double, 4, q, n, n> A2;

    // A2(u, j, k) := B(v, j) * A1(u, v, k)
    for (int u = threadIdx.x; u < q; u += blockDim.x) {
      for (int j = threadIdx.y; j < n; j += blockDim.y) {
        for (int k = threadIdx.z; k < n; k += blockDim.z) {
          tensor<double, 4> sum{};
          for (int v = 0; v < q; v++) {
            sum[0] += B(v, j) * A1(0, u, v, k);
            sum[1] += B(v, j) * A1(1, u, v, k);
            sum[2] += G(v, j) * A1(2, u, v, k);
            sum[3] += B(v, j) * A1(3, u, v, k);
          }
          A2(0, u, j, k) = sum[0];
          A2(1, u, j, k) = sum[1];
          A2(2, u, j, k) = sum[2];
          A2(3, u, j, k) = sum[3];
        }
      }
    }
    __syncthreads();

    // r(i, j, k) := B(u, i) * A2(u, j, k)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      for (int j = threadIdx.y; j < n; j += blockDim.y) {
        for (int k = threadIdx.z; k < n; k += blockDim.z) {
          double sum = 0.0;
          for (int u = 0; u < q; u++) {
            sum += B(u, i) * A2(0, u, j, k);
            sum += G(u, i) * A2(1, u, j, k);
            sum += B(u, i) * A2(2, u, j, k);
            sum += B(u, i) * A2(3, u, j, k);
          }
          r_e(i, j, k, e) += sum;
        }
      }
    }
  }
}



}
