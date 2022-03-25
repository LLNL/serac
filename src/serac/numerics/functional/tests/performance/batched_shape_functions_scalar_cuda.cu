#include <string>
#include <iostream>

#include "mfem_PA_kernels.hpp"
#include "mfem_PA_kernels_smem.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

#include "sum_factorization.hpp"
#include "sum_factorization_external_cache.hpp"

namespace serac {

template < typename trial_space, Geometry geom, int q >
auto BatchPreprocess(const mfem::DeviceTensor< 4, const double > & u_e, GaussLegendreRule<geom, q> rule, int e) {
  static constexpr int n = trial_space::order + 1;

  if constexpr (geom == Geometry::Hexahedron) {

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }
    auto BT = transpose(B);
    auto GT = transpose(G);

    tensor< value_and_gradient< double, tensor< double, 3 > >, q, q, q> u_q{};

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

template < typename lambda, typename T, int ... n, Geometry geom, int q >
auto BatchApply(lambda qf, tensor< T, n ... > qf_inputs, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    using output_type = decltype(qf(qf_inputs[0][0][0]));

    tensor< output_type, q, q, q > qf_outputs;

    int q_id = 0;
    for (int qz = 0; qz < q; ++qz) {
      for (int qy = 0; qy < q; ++qy) {
        for (int qx = 0; qx < q; ++qx) {

          auto qf_input = qf_inputs[qz][qy][qx];
          
          auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(qx, qy, qz, i, j, e); });
          auto invJ = inv(J);
          auto dv = det(J) * rule.weight(qx, qy, qz);

          qf_input.gradient = dot(qf_input.gradient, invJ);

          qf_outputs[qz][qy][qx] = qf(qf_input) * dv;

          serac::get<1>(qf_outputs[qz][qy][qx]) = dot(invJ, serac::get<1>(qf_outputs[qz][qy][qx]));

          q_id++;

        }
      }
    }

    return qf_outputs;

  }

}

template < typename trial_space, typename T, Geometry geom, int q >
auto BatchPostprocess(const tensor < T, q, q, q > qf_outputs, GaussLegendreRule<geom, q> rule) {

  if constexpr (geom == Geometry::Hexahedron) {

    static constexpr int n = trial_space::order + 1;

    tensor< double, q, n > B{};
    tensor< double, q, n > G{};
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.points_1D[i]);
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);
    }

    tensor< double, n, n, n > element_residual{};

    for (int qz = 0; qz < q; ++qz) {
      tensor < value_and_gradient< double, tensor< double, 3 > >, n, n > gradXY{};
      for (int qy = 0; qy < q; ++qy) {
        tensor < value_and_gradient< double, tensor< double, 3 > >, n > gradX{};
        for (int qx = 0; qx < q; ++qx) {
          const T qf_output = qf_outputs[qz][qy][qx];
          for (int dx = 0; dx < n; ++dx) {
            const double wx = B(qx, dx);
            const double wDx = G(qx, dx);
            gradX[dx].value       += serac::get<0>(qf_output) * wx;
            gradX[dx].gradient[0] += serac::get<1>(qf_output)[0] * wDx;
            gradX[dx].gradient[1] += serac::get<1>(qf_output)[1] * wx;
            gradX[dx].gradient[2] += serac::get<1>(qf_output)[2] * wx;
          }
        }
        for (int dy = 0; dy < n; ++dy) {
          const double wy = B(qy, dy);
          const double wDy = G(qy, dy);
          for (int dx = 0; dx < n; ++dx) {
            gradXY[dy][dx].value       += gradX[dx].value       * wy;
            gradXY[dy][dx].gradient[0] += gradX[dx].gradient[0] * wy;
            gradXY[dy][dx].gradient[1] += gradX[dx].gradient[1] * wDy;
            gradXY[dy][dx].gradient[2] += gradX[dx].gradient[2] * wy;
          }
        }
      }
      for (int dz = 0; dz < n; ++dz) {
        const double wz = B(qz, dz);
        const double wDz = G(qz, dz);
        for (int dy = 0; dy < n; ++dy) {
          for (int dx = 0; dx < n; ++dx) {
            auto tmp = gradXY[dy][dx];
            element_residual[dx][dy][dz] += (tmp.value + tmp.gradient[0] + tmp.gradient[1]) * wz + tmp.gradient[2] * wDz;
          }
        }
      }
    }

    return element_residual;

  }

}

namespace detail {

template <int n>
SERAC_HOST_DEVICE void Add(const mfem::DeviceTensor<4, double>& r_global, const tensor<double, n, n, n> r_elem, int e)
{
  for (int ix = 0; ix < n; ix++) {
    for (int iy = 0; iy < n; iy++) {
      for (int iz = 0; iz < n; iz++) {
        r_global(ix, iy, iz, e) += r_elem[ix][iy][iz];
      }
    }
  }
}

} // namespace detail

template < Geometry geom, typename test, typename trial, int Q, typename lambda >
void batched_kernel(const mfem::Vector & U_, mfem::Vector & R_, const mfem::Vector & J_, size_t num_elements_, lambda qf_) {

  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_n     = test_element::order + 1;
  static constexpr int  trial_n    = trial_element::order + 1;
  static constexpr auto rule       = GaussLegendreRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.HostRead(), Q, Q, Q, dim, dim, num_elements_);
  auto r = mfem::Reshape(R_.HostReadWrite(), test_n, test_n, test_n, int(num_elements_));
  auto u = mfem::Reshape(U_.HostRead(), trial_n, trial_n, trial_n, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {

    auto args = BatchPreprocess<trial>(u, rule, e);

    auto qf_outputs = BatchApply(qf_, args, rule, J, e);

    auto r_elem = BatchPostprocess<test>(qf_outputs, rule);

    detail::Add(r, r_elem, int(e));

  }

}

template <Geometry g, typename test, typename trial, int Q, typename lambda>
__global__ void reference_cuda_kernel(mfem::DeviceTensor< 2, const double > u, 
                                      mfem::DeviceTensor< 2, double > r, 
                                      mfem::DeviceTensor< 4, const double > J, 
                                      size_t num_elements, 
                                      lambda qf) {

  using test_element          = finite_element<g, test>;
  using trial_element         = finite_element<g, trial>;
  using element_residual_type = typename test_element::residual_type;
  static constexpr auto rule  = GaussQuadratureRule<g, Q>();
  static constexpr int  dim   = dimension_of(g);

  const int grid_stride = blockDim.x * gridDim.x;

  for (int qe = blockIdx.x * blockDim.x + threadIdx.x; qe < num_elements * rule.size(); qe += grid_stride) {

    int e = qe / rule.size();
    int q = qe % rule.size();

    auto u_elem = detail::Load<trial_element>(u, e);

    element_residual_type r_elem{};

    auto   xi  = rule.points[q];
    auto   dxi = rule.weights[q];
    auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
    double dx  = det(J_q) * dxi;

    auto arg = domain_integral::Preprocess<trial_element>(u_elem, xi, J_q);

    auto qf_output = qf(arg);

    r_elem += domain_integral::Postprocess<test_element>(qf_output, xi, J_q) * dx;

    detail::Add(r, r_elem, e);

  }

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
__device__ auto BatchApplyCUDA(lambda qf, T qf_input, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

    auto invJ = inv(J);

    auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

    qf_input.gradient = dot(qf_input.gradient, invJ);

    auto qf_output = qf(qf_input) * dv;

    serac::get<1>(qf_output) = dot(invJ, serac::get<1>(qf_output));

    return qf_output;

  }

}

template <Geometry g, typename test, typename trial, int Q, typename lambda>
__global__ void batched_cuda_kernel(mfem::DeviceTensor< 4, const double > u, 
                                    mfem::DeviceTensor< 4, double > r, 
                                    mfem::DeviceTensor< 6, const double > J, 
                                    size_t num_elements, 
                                    lambda qf) {

  static constexpr auto rule  = GaussLegendreRule<g, Q>();

  // for each element in the domain
  uint32_t e = blockIdx.x;

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(u, rule, e);

  // evalute the q-function
  auto qf_output = BatchApplyCUDA(qf, qf_input, rule, J, e);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(qf_output, rule, r, e);

}

template < typename lambda, typename T, int ... n, Geometry geom, int q >
__device__ auto BatchApplyCUDA_with_cache(lambda qf, T qf_input, GaussLegendreRule<geom, q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 4, q, q, q > & cache) {

  if constexpr (geom == Geometry::Hexahedron) {

    constexpr int dim = 3;

    auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

    auto invJ = inv(J);

    auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

    qf_input.gradient = dot(qf_input.gradient, invJ);

    auto qf_output = qf(qf_input) * dv;

    serac::get<1>(qf_output) = dot(invJ, serac::get<1>(qf_output));

    cache(0, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<0>(qf_output);
    cache(1, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[0];
    cache(2, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[1];
    cache(3, threadIdx.z, threadIdx.y, threadIdx.x) = serac::get<1>(qf_output)[2];

  }

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel_with_cache(mfem::DeviceTensor< 4, const double > u, 
                                               mfem::DeviceTensor< 4, double > r, 
                                               mfem::DeviceTensor< 6, const double > J, 
                                               size_t num_elements, 
                                               lambda qf) {

  static constexpr int n = trial::order + 1;
  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ tensor< double, q, n > B;
  __shared__ tensor< double, q, n > G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  __shared__ tensor < double, n, n, n > u_elem;
  __shared__ tensor < double, 3, n, n, q > A1;
  __shared__ tensor < double, 3, n, q, q > A2;

  __shared__ tensor < double, 4, q, q, q > qf_output;
  __shared__ tensor < double, 3, q, q, n > A3;
  __shared__ tensor < double, 2, q, n, n > A4;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(u_elem, rule, B, G, A1, A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(qf_output, rule, r, e, B, G, A3, A4);

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel_with_union_cache(mfem::DeviceTensor< 4, const double > u, 
                                                     mfem::DeviceTensor< 4, double > r, 
                                                     mfem::DeviceTensor< 6, const double > J, 
                                                     size_t num_elements, 
                                                     lambda qf) {

  static constexpr int n = trial::order + 1;
  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ tensor< double, q, n > B;
  __shared__ tensor< double, q, n > G;

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int i = 0; i < q; i++) {
      B[i] = GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }
  }

  __shared__ union {
    tensor < double, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 3, n, n, q > A1;
    tensor < double, 4, q, q, q > qf_output;
    tensor < double, 3, q, q, n > A3;
  } cache2;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  // load the values for that element
  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        cache1.u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(cache1.u_elem, rule, B, G, cache2.A1, cache1.A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, cache2.qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(cache2.qf_output, rule, r, e, B, G, cache2.A3, cache1.A4);

}

template <Geometry g, typename test, typename trial, int q, int n, typename lambda>
__global__ void batched_cuda_kernel_with_union_cache_const_memory(mfem::DeviceTensor< 4, const double > u, 
                                                     mfem::DeviceTensor< 4, double > r, 
                                                     mfem::DeviceTensor< 6, const double > J, 
                                                     const tensor< double, q, n > B,
                                                     const tensor< double, q, n > G,
                                                     size_t num_elements, 
                                                     lambda qf) {

  static constexpr auto rule  = GaussLegendreRule<g, q>();

  __shared__ union {
    tensor < double, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 3, n, n, q > A1;
    tensor < double, 4, q, q, q > qf_output;
    tensor < double, 3, q, q, n > A3;
  } cache2;

  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
    for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
      for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
        cache1.u_elem(dz, dy, dx) = u(dx, dy, dz, e);
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = BatchPreprocessCUDA<trial>(cache1.u_elem, rule, B, G, cache2.A1, cache1.A2);

  // evalute the q-function
  BatchApplyCUDA_with_cache(qf, qf_input, rule, J, e, cache2.qf_output);

  // integrate the material response against the test-space basis functions
  BatchPostprocessCUDA<test>(cache2.qf_output, rule, r, e, B, G, cache2.A3, cache1.A4);

}

template < typename lambda, typename T, int q >
__device__ auto batch_apply_qf(lambda qf, T qf_input, TensorProductQuadratureRule<q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 1, q, q, q > & cache_source, tensor< double, 3, 1, q, q, q > & cache_flux) {

  constexpr int dim = 3;

  auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });

  auto invJ = inv(J);

  auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);

  serac::get<1>(qf_input) = dot(serac::get<1>(qf_input), invJ);

  auto [source, flux] = qf(qf_input) * dv;

  flux = dot(flux, transpose(invJ));

  cache_source(0, threadIdx.z, threadIdx.y, threadIdx.x) = source;

  cache_flux(0, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][0];
  cache_flux(1, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][1];
  cache_flux(2, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][2];
  __syncthreads();

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_kernel(mfem::DeviceTensor< 5, const double > u, 
                                    mfem::DeviceTensor< 5, double > r, 
                                    mfem::DeviceTensor< 6, const double > J, 
                                    TensorProductQuadratureRule<q> rule,
                                    size_t num_elements, 
                                    lambda qf) {

  static constexpr int n = trial::order + 1;
  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  __shared__ union {
    tensor < double, trial::components, n, n, n > u_elem;
    tensor < double, 3, n, q, q > A2;
    tensor < double, 2, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, 2, n, n, q > A1;
    struct {
      tensor < double, 1, q, q, q > source;
      tensor < double, 3, 1, q, q, q > flux;
    };
    tensor < double, 3, q, q, n > A3;
  } cache2;


  // for each element in the domain
  uint32_t e = blockIdx.x;

  for (int i = 0; i < trial::components; i++) {
    for (int dz = threadIdx.z; dz < n; dz += blockDim.z) {
      for (int dy = threadIdx.y; dy < n; dy += blockDim.y) {
        for (int dx = threadIdx.x; dx < n; dx += blockDim.x) {
          cache1.u_elem(i, dz, dy, dx) = u(dx, dy, dz, i, e);
        }
      }
    }
  }
  __syncthreads(); 

  // interpolate each quadrature point's value
  auto qf_input = trial_element::interpolate(cache1.u_elem, rule, cache2.A1, cache1.A2);

  // evalute the q-function at each quadrature point
  batch_apply_qf(qf, qf_input, rule, J, e, cache2.source, cache2.flux);

  // integrate the material response against the test-space basis functions
  test_element::extrapolate(cache2.source, cache2.flux, rule, r, e, cache2.A3, cache1.A4);

}

} // namespace serac

namespace compiler {
  static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

struct MassAndDiffusionQFunction {
  template < typename T >
  SERAC_HOST_DEVICE auto operator()(T input) {
    auto [u, du_dx] = input;
    auto source = rho * u;
    auto flux = k * du_dx;
    return serac::tuple{source, flux};
  }

  double rho;
  double k;
};

template <typename lambda>
auto time(lambda&& f)
{
  axom::utilities::Timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

constexpr int dim = 3;
constexpr int num_runs = 10;

constexpr double k = 1.0;
constexpr double rho = 1.0;

constexpr MassAndDiffusionQFunction qfunc{rho, k};

mfem::Vector U1D;
mfem::Vector R1D;
mfem::Vector J1D;
mfem::Vector rho_dv_1D;
mfem::Vector k_invJ_invJT_dv_1D;

void initialize_globals() {

  constexpr int n = 4;
  constexpr int q = 4;
  constexpr int num_elements = 128 << 10;

  U1D.SetSize(num_elements * n * n * n);
  R1D.SetSize(num_elements * n * n * n);
  J1D.SetSize(num_elements * dim * dim * q * q * q);
  rho_dv_1D.SetSize(num_elements * q * q * q);
  k_invJ_invJT_dv_1D.SetSize(num_elements * dim * dim * q * q * q);

  U1D.UseDevice(true);
  J1D.UseDevice(true);
  rho_dv_1D.UseDevice(true);
  k_invJ_invJT_dv_1D.UseDevice(true);

  std::default_random_engine generator{0};
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  auto U = mfem::Reshape(U1D.HostReadWrite(), n, n, n, num_elements);
  auto J = mfem::Reshape(J1D.HostReadWrite(), q * q * q, dim, dim, num_elements);

  for (int e = 0; e < num_elements; e++) {

    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(iz, iy, ix, e) = 0.1 * distribution(generator);
        }
      }
    }

    for (int i = 0; i < q * q * q; i++) {
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = (r == c) + 0.1 * distribution(generator);
        }
      }
    }

  }

}

template < typename ... T >
void print(T ... args) {
  (..., (std::cout << " " << args));
  std::cout << std::endl;
}

template < int q, int n >
void run_test_suite(int num_elements) {

  constexpr int p = n - 1;

  using serac::H1;
  using serac::Geometry;

  using test = H1<p>;
  using trial = H1<p>;

  mfem::Vector R1D(num_elements * n * n * n);
  R1D.UseDevice(true);

  // run the CPU kernel once to generate the accepted answers
  {
    R1D = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_kernel<serac::Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    double relative_error = 0.0;
    print("reference_cpu", n, q, num_elements, runtime, relative_error);
  }
  auto answer_reference = R1D;

  {
    R1D = 0.0;

    mfem::DeviceTensor<2, const double > u_d = mfem::Reshape(U1D.Read(), n * n * n, num_elements);
    mfem::DeviceTensor<2, double > r_d = mfem::Reshape(R1D.ReadWrite(), n * n * n, num_elements);
    mfem::DeviceTensor<4, const double > J_d = mfem::Reshape(J1D.Read(), q * q * q, dim, dim, num_elements);
    int blocksize = 128;
    int gridsize = (num_elements * q * q * q + blocksize - 1) / blocksize;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("reference_gpu", n, q, num_elements, runtime, relative_error);
  }


  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("batched_gpu", n, q, num_elements, runtime, relative_error);
  }

  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_cache<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("batched_gpu_with_cache", n, q, num_elements, runtime, relative_error);
  }

  {
    R1D = 0.0;

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_union_cache<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("batched_gpu_with_union_cache", n, q, num_elements, runtime, relative_error);
  }

  {
    R1D = 0.0;

    serac::tensor< double, q, n > B;
    serac::tensor< double, q, n > G;

    serac::GaussLegendreRule<serac::Geometry::Hexahedron, q> rule;

    for (int i = 0; i < q; i++) {
      B[i] = serac::GaussLobattoInterpolation<n>(rule.point(i));
      G[i] = serac::GaussLobattoInterpolationDerivative<n>(rule.point(i));
    }

    mfem::DeviceTensor<4, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, num_elements);
    mfem::DeviceTensor<4, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel_with_union_cache_const_memory<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, B, G, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("batched_gpu_with_union_cache_and_const_memory", n, q, num_elements, runtime, relative_error);
  }

  {
    R1D = 0.0;

    mfem::DeviceTensor<5, const double > u_d = mfem::Reshape(U1D.Read(), n, n, n, 1, num_elements);
    mfem::DeviceTensor<5, double > r_d = mfem::Reshape(R1D.ReadWrite(), n, n, n, 1, num_elements);
    mfem::DeviceTensor<6, const double > J_d = mfem::Reshape(J1D.Read(), q, q, q, dim, dim, num_elements);
    auto rule = serac::MakeGaussLegendreRule<Geometry::Hexahedron, q>();
    dim3 blocksize{q, q, q};
    int gridsize = num_elements;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::batched_cuda_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, rule, num_elements, qfunc);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("batched_gpu_with_element_library", n, q, num_elements, runtime, relative_error);
  }

  serac::GaussLegendreRule<serac::Geometry::Hexahedron, q> rule;
  auto J = mfem::Reshape(J1D.HostReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_dv = mfem::Reshape(rho_dv_1D.HostReadWrite(), q * q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.HostReadWrite(), q * q * q, dim, dim, num_elements);
  for (int e = 0; e < num_elements; e++) {
    for (int i = 0; i < q * q * q; i++) {

      auto J_q = serac::make_tensor< dim, dim >([=](int r, int c){ return J(i, r, c, e); });

      int qx = i % q;
      int qy = (i % (q * q)) / q;
      int qz = i / (q * q);

      double qweight = rule.weight(qx, qy, qz);
      auto invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      double dv = det(J_q) * qweight;

      rho_dv(i, e) = rho * dv; 
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          k_invJ_invJT_dv(i, r, c, e) = k * invJ_invJT[r][c] * dv;
        }
      }

    }
  }

  {
    R1D = 0.0;
    bool symmetric = false;
    mfem::Array<double> b_(n * q);
    mfem::Array<double> bt_(n * q);
    mfem::Array<double> g_(n * q);
    mfem::Array<double> gt_(n * q);
    auto B = mfem::Reshape(b_.HostReadWrite(), q, n);
    auto Bt = mfem::Reshape(bt_.HostReadWrite(), n, q);

    auto G = mfem::Reshape(g_.HostReadWrite(), q, n);
    auto Gt = mfem::Reshape(gt_.HostReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto value = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bt(j, i) = B(i, j) = value[j];
        Gt(j, i) = G(i, j) = derivative[j];
      }
    }

    double mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;

    double diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, bt_, gt_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;

    auto error = R1D;
    error -= answer_reference;
    auto relative_error = error.Norml2() / answer_reference.Norml2();
    print("mfem_mass", n, q, num_elements, mass_runtime, relative_error);
    print("mfem_diffusion", n, q, num_elements, diffusion_runtime, relative_error);
    print("mfem_combined", n, q, num_elements, mass_runtime + diffusion_runtime, relative_error);

    R1D = 0.0;
    mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;

    diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;

    error = R1D;
    error -= answer_reference;
    relative_error = error.Norml2() / answer_reference.Norml2();
    print("mfem_mass_smem", n, q, num_elements, mass_runtime, relative_error);
    print("mfem_diffusion_smem", n, q, num_elements, diffusion_runtime, relative_error);
    print("mfem_combined_smem", n, q, num_elements, mass_runtime + diffusion_runtime, relative_error);
  }

}

int main() {

  mfem::Device device("cuda");

  initialize_globals();

//  {
//    constexpr int n = 2;
//    constexpr int q = 2;
//    for (int i = 0; i < 10; i++) {
//      int num_elements = (1024 << i);
//      run_test_suite< q, n >(num_elements);
//    }
//  }
//
//  {
//    constexpr int n = 3;
//    constexpr int q = 3;
//    for (int i = 0; i < 10; i++) {
//      int num_elements = (256 << i);
//      run_test_suite< q, n >(num_elements);
//    }
//  }
//
//  {
//    constexpr int n = 4;
//    constexpr int q = 4;
//    for (int i = 0; i < 10; i++) {
//      int num_elements = (128 << i);
//      run_test_suite< q, n >(num_elements);
//    }
//  }

  run_test_suite< 4, 4 >(65536);

}
