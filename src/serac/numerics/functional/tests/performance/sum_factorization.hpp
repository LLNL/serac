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
__device__ auto BatchPreprocessCUDA(const mfem::DeviceTensor< 4, const double > & element_values, GaussLegendreRule<geom, q> rule, int e) {
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {

    // we want to compute the following:
    //
    // r(u, v, w) := (B(u, i) * B(v, j) * B(w, k)) * f(i, j, k)
    //
    // where 
    //   r(u, v, w) are the quadrature-point values at position {u, v, w}, 
    //   B(u, i) is the i^{th} 1D interpolation/differentiation (shape) function, 
    //           evaluated at the u^{th} 1D quadrature point, and
    //   f(i, j, k) are the values at node {i, j, k} to be interpolated 
    //
    // this algorithm carries out the above calculation in 3 steps:
    //
    // A1(i, j, w) := B(w, k) * f(i, j, k)
    // A2(i, v, w) := B(v, j) * A1(i, j, w)
    //  r(u, v, w) := B(u, i) * A2(i, v, w)

    tensor<double, q, n> B{};
    tensor<double, q, n> G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }

    __shared__ tensor<double, 2, n, n, q> A1;
    for (int u = threadIdx.x; u < n; u += blockDim.x) {
      for (int v = threadIdx.y; v < n; v += blockDim.y) {
        for (int w = threadIdx.z; w < q; w += blockDim.z) {
          A1(0, u, v, w) = 0.0;
          A1(1, u, v, w) = 0.0;
        }
      }
    }

    // A1(i, j, w) := B(w, k) * f(i, j, k)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      for (int j = threadIdx.y; j < n; j += blockDim.y) {
        for (int k = threadIdx.z; k < n; k += blockDim.z) {
          auto f_ijk = element_values(i, j, k, e);
          for (int w = 0; w < q; w++) {
            atomicAdd(&A1(0, i, j, w), B(w, k) * f_ijk);
            atomicAdd(&A1(1, i, j, w), G(w, k) * f_ijk);
          }
        }
      }
    }
    __syncthreads();

    __shared__ tensor<double, 3, n, q, q> A2;

    // A2(i, v, w) := B(v, j) * A1(i, j, w)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      for (int v = threadIdx.y; v < q; v += blockDim.y) {
        for (int w = threadIdx.z; w < q; w += blockDim.z) {
          double sum[3]{};
          for (int j = 0; j < n; j++) {
            sum[0] += B(v, j) * A1(0, i, j, w);
            sum[1] += G(v, j) * A1(0, i, j, w);
            sum[2] += B(v, j) * A1(1, i, j, w);
          }
          A2(0, i, v, w) = sum[0];
          A2(1, i, v, w) = sum[1];
          A2(2, i, v, w) = sum[2];
        }
      }
    }
    __syncthreads();

    value_and_gradient< double, tensor<double, 3> > qf_input{};

    // r(u, v, w) := B(u, i) * A2(i, v, w)
    for (int u = threadIdx.x; u < q; u += blockDim.x) {
      for (int v = threadIdx.y; v < q; v += blockDim.y) {
        for (int w = threadIdx.z; w < q; w += blockDim.z) {
          for (int i = 0; i < n; i++) {
            qf_input.value       += B(u, i) * A2(0, i, v, w);
            qf_input.gradient[0] += G(u, i) * A2(0, i, v, w);
            qf_input.gradient[1] += B(u, i) * A2(1, i, v, w);
            qf_input.gradient[2] += B(u, i) * A2(2, i, v, w);
          }
        }
      }
    }

    return qf_input;

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
