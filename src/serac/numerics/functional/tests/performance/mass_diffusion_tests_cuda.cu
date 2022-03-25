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

template < typename lambda, typename T, int q >
__device__ auto batch_apply_mass_qf(lambda qf, T qf_input, TensorProductQuadratureRule<q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 1, q, q, q > & cache_source) {
  constexpr int dim = 3;
  auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });
  auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);
  auto source = qf(qf_input) * dv;
  cache_source(0, threadIdx.z, threadIdx.y, threadIdx.x) = source;
  __syncthreads();
}

template < typename lambda, typename T, int q >
__device__ auto batch_apply_diffusion_qf(lambda qf, T qf_input, TensorProductQuadratureRule<q> rule, mfem::DeviceTensor< 6, const double > J_q, int e, tensor< double, 3, 1, q, q, q > & cache_flux) {
  constexpr int dim = 3;
  auto J = make_tensor<dim, dim>([&](int i, int j) { return J_q(threadIdx.x, threadIdx.y, threadIdx.z, i, j, e); });
  auto invJ = inv(J);
  auto dv = det(J) * rule.weight(threadIdx.x, threadIdx.y, threadIdx.z);
  qf_input = dot(qf_input, invJ);
  auto flux = qf(qf_input) * dv;
  flux = dot(flux, transpose(invJ));

  cache_flux(0, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][0];
  cache_flux(1, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][1];
  cache_flux(2, 0, threadIdx.z, threadIdx.y, threadIdx.x) = flux[0][2];
  __syncthreads();
}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_mass_kernel(mfem::DeviceTensor< 5, const double > u, 
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
    tensor < double, n, q, q > A2;
    tensor < double, q, n, n > A4;
  } cache1;

  __shared__ union {
    tensor < double, n, n, q > A1;
    tensor < double, 1, q, q, q > source;
    tensor < double, q, q, n > A3;
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
  batch_apply_mass_qf(qf, qf_input, rule, J, e, cache2.source);

  // integrate the material response against the test-space basis functions
  test_element::extrapolate(cache2.source, rule, r, e, cache2.A3, cache1.A4);

}

template <Geometry g, typename test, typename trial, int q, typename lambda>
__global__ void batched_cuda_diffusion_kernel(mfem::DeviceTensor< 5, const double > u, 
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
    tensor < double, 3, 1, q, q, q > flux;
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
  auto qf_input = trial_element::gradient(cache1.u_elem, rule, cache2.A1, cache1.A2);

  // evalute the q-function at each quadrature point
  batch_apply_diffusion_qf(qf, qf_input, rule, J, e, cache2.flux);

  // integrate the material response against the test-space basis functions
  test_element::extrapolate(cache2.flux, rule, r, e, cache2.A3, cache1.A4);

}

} // namespace serac

namespace compiler {
  static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

struct MassQFunction {
  template < typename T >
  SERAC_HOST_DEVICE auto operator()(T u) { return rho * u; }
  double rho;
};

struct DiffusionQFunction {
  template < typename T >
  SERAC_HOST_DEVICE auto operator()(T du_dx) { return k * du_dx; }
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

  mfem::Vector mass_answer;
  mfem::Vector diffusion_answer;

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
        serac::batched_cuda_mass_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, rule, num_elements, MassQFunction{rho});
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    mass_answer = R1D;
    double relative_error = 0.0; // this is the reference answer
    print("serac_mass_kernel", n, q, num_elements, runtime, relative_error);
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
        serac::batched_cuda_diffusion_kernel<Geometry::Hexahedron, test, trial, q><<<gridsize, blocksize>>>(u_d, r_d, J_d, rule, num_elements, DiffusionQFunction{k});
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    diffusion_answer = R1D;
    double relative_error = 0.0; // this is the reference answer
    print("serac_diffusion_kernel", n, q, num_elements, runtime, relative_error);
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

    R1D = 0.0;
    double mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    auto error = R1D;
    error -= mass_answer;
    auto relative_error = error.Norml2() / mass_answer.Norml2();
    print("mfem_mass", n, q, num_elements, mass_runtime, relative_error);

    R1D = 0.0;
    double diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, bt_, gt_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    error = R1D;
    error -= diffusion_answer;
    relative_error = error.Norml2() / diffusion_answer.Norml2();
    print("mfem_diffusion", n, q, num_elements, diffusion_runtime, relative_error);


    R1D = 0.0;
    mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPAMassApply3D<n,q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    error = R1D;
    error -= mass_answer;
    relative_error = error.Norml2() / mass_answer.Norml2();
    print("mfem_mass_smem", n, q, num_elements, mass_runtime, relative_error);

    R1D = 0.0;
    diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPADiffusionApply3D<n,q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
      cudaDeviceSynchronize();
    }) / num_runs;
    error = R1D;
    error -= diffusion_answer;
    relative_error = error.Norml2() / diffusion_answer.Norml2();
    print("mfem_diffusion_smem", n, q, num_elements, diffusion_runtime, relative_error);
  }

}

int main() {

  mfem::Device device("cuda");

  initialize_globals();

#if 0
  {
    constexpr int n = 2;
    constexpr int q = 2;
    for (int i = 0; i < 10; i++) {
      int num_elements = (1024 << i);
      run_test_suite< q, n >(num_elements);
    }
  }

  {
    constexpr int n = 3;
    constexpr int q = 3;
    for (int i = 0; i < 10; i++) {
      int num_elements = (256 << i);
      run_test_suite< q, n >(num_elements);
    }
  }

  {
    constexpr int n = 4;
    constexpr int q = 4;
    for (int i = 0; i < 10; i++) {
      int num_elements = (128 << i);
      run_test_suite< q, n >(num_elements);
    }
  }
#endif

  run_test_suite< 4, 4 >(65536);
}
