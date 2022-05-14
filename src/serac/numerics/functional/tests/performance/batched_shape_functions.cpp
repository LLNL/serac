#include "mfem.hpp"

#include "mfem_PA_kernels_h1.hpp"
#include "mfem_PA_kernels_hcurl.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

#include <vector>
#include "immintrin.h"

template <int n>
struct simd;

template <>
struct simd<4> {
  using type = __m256d;
};

template <>
struct simd<8> {
  using type = __m512d;
};

namespace serac {

auto& operator<<(std::ostream& out, const __m256d& A)
{
  out << '{' << A[0];
  for (int i = 1; i < 4; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

auto& operator<<(std::ostream& out, const __m512d& A)
{
  out << '{' << A[0];
  for (int i = 1; i < 8; i++) {
    out << ", " << A[i];
  }
  out << '}';
  return out;
}

template <Geometry g, int Q>
struct GaussLegendreRule;

template <int Q>
struct GaussLegendreRule<Geometry::Quadrilateral, Q> {
  static constexpr auto points_1D  = GaussLegendreNodes<Q>();
  static constexpr auto weights_1D = GaussLegendreWeights<Q>();

  static constexpr double weight(int qx, int qy) { return weights_1D[qx] * weights_1D[qy]; }

  static constexpr int size() { return Q * Q; }
};

template <int Q>
struct GaussLegendreRule<Geometry::Hexahedron, Q> {
  static constexpr auto points_1D  = GaussLegendreNodes<Q>();
  static constexpr auto weights_1D = GaussLegendreWeights<Q>();

  static constexpr double weight(int qx, int qy, int qz) { return weights_1D[qx] * weights_1D[qy] * weights_1D[qz]; }

  static constexpr int size() { return Q * Q * Q; }
};

template <Geometry geom, typename test, typename trial, int Q, typename lambda>
void reference_kernel(const mfem::Vector& U_, mfem::Vector& R_, const mfem::Vector& J_, size_t num_elements_,
                      lambda qf_)
{
  using trial_element              = finite_element<geom, trial>;
  using test_element               = finite_element<geom, test>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(geom);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<geom, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements_);
  auto r = detail::Reshape<test>(R_.ReadWrite(), test_ndof, int(num_elements_));
  auto u = detail::Reshape<test>(U_.Read(), trial_ndof, int(num_elements_));

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements_; e++) {
    // get the DOF values for this particular element
    auto u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = domain_integral::Preprocess<trial_element>(u_elem, xi, J_q);

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += domain_integral::Postprocess<test_element>(qf_(arg), xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, int(e));
  }
}

template <typename dof_type>
void load(const dof_type& source, dof_type& destination)
{
#ifdef __CUDA_ARCH__
  constexpr int ndof    = sizeof(dof_type) / sizeof(double);
  double*       src_ptr = reinterpret_cast<double*>(&element_values);
  for (int i = 0; i < ndof; i++) {
    element_values_ptr[i] = ptr[i];
  }
#else
  destination = source;
#endif
}

#include <type_traits>

template <typename lambda, typename value_t, typename derivative_t, int q>
auto batch_apply_qf(lambda qf, const tensor<value_t, q, q>& values, const tensor<derivative_t, q, q>& derivatives)
{
  using return_type = decltype(qf(serac::tuple{value_t{}, derivative_t{}}));
  using source_type = std::remove_const_t<std::remove_reference_t<decltype(serac::get<0>(return_type{}))> >;
  using flux_type   = std::remove_const_t<std::remove_reference_t<decltype(serac::get<1>(return_type{}))> >;

  tensor<source_type, q, q> sources{};
  tensor<flux_type, q, q>   fluxes{};

  for (int qy = 0; qy < q; ++qy) {
    for (int qx = 0; qx < q; ++qx) {
      auto [source, flux] = qf(serac::tuple{values(qy, qx), derivatives(qy, qx)});
      sources(qy, qx)     = source;
      fluxes(qy, qx)      = flux;
    }
  }

  return serac::tuple{sources, fluxes};
}

template <typename lambda, typename value_t, typename derivative_t, int q>
auto batch_apply_qf(lambda qf, const tensor<value_t, q, q, q>& values, const tensor<derivative_t, q, q, q>& derivatives)
{
  using return_type = decltype(qf(serac::tuple{value_t{}, derivative_t{}}));
  using source_type = std::remove_const_t<std::remove_reference_t<decltype(serac::get<0>(return_type{}))> >;
  using flux_type   = std::remove_const_t<std::remove_reference_t<decltype(serac::get<1>(return_type{}))> >;

  tensor<source_type, q, q, q> sources{};
  tensor<flux_type, q, q, q>   fluxes{};

  for (int qz = 0; qz < q; ++qz) {
    for (int qy = 0; qy < q; ++qy) {
      for (int qx = 0; qx < q; ++qx) {
        auto [source, flux] = qf(serac::tuple{values(qz, qy, qx), derivatives(qz, qy, qx)});
        sources(qz, qy, qx) = source;
        fluxes(qz, qy, qx)  = flux;
      }
    }
  }

  return serac::tuple{sources, fluxes};
}

template <typename T, typename value_t, typename derivative_t, int q>
auto batch_apply_PA_simd(const tensor<value_t, q, q, q>& values, const tensor<derivative_t, q, q, q>& derivatives,
                         const tensor<T, q, q, q>& D1, const tensor<tensor<T, 3, 3>, q, q, q>& D2)
{
  tensor<value_t, q, q, q>      sources{};
  tensor<derivative_t, q, q, q> fluxes{};

  for (int qz = 0; qz < q; ++qz) {
    for (int qy = 0; qy < q; ++qy) {
      for (int qx = 0; qx < q; ++qx) {
        sources(qz, qy, qx)[0] = D1(qz, qy, qx) * values(qz, qy, qx)[0];
        fluxes(qz, qy, qx)[0]  = dot(D2(qz, qy, qx), derivatives(qz, qy, qx)[0]);
      }
    }
  }

  return serac::tuple{sources, fluxes};
}

#if 0
template <Geometry g, typename test, typename trial, int q, typename lambda>
void cpu_batched_kernel(const double * inputs,
                        double * outputs,
                        const double * jacobians,
                        TensorProductQuadratureRule<q> rule,
                        size_t num_elements, 
                        lambda qf) {

  using test_element = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  auto u = reinterpret_cast< const typename trial_element::dof_type * >(inputs);
  auto r = reinterpret_cast< typename test_element::dof_type * >(outputs);

  auto J = reinterpret_cast< const typename trial_element::dof_type * >(inputs);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {

    // load the element values and jacobians for this element
    typename trial_element::dof_type u_elem;
    load(u[e], u_elem);

    // interpolate each quadrature point's value
    typename trial_element::cache_type<q> trial_cache;
    typename trial_element::batched_values_type<q> values;
    typename trial_element::batched_derivatives_type<q> derivatives;
    trial_element::interpolate(u_elem, rule, trial_cache, values, derivatives);

    // evalute the q-function at each quadrature point
    typename test_element::batched_values_type<q> sources;
    typename test_element::batched_derivatives_type<q> fluxes;
    batch_apply_qf(qf, rule, jacobians + e, values, derivatives, sources, fluxes);

    // integrate the material response against the test-space basis functions
    typename test_element::cache_type<q> test_cache;
    test_element::integrate(sources, fluxes, rule, test_cache, r[e]);

  }

}
#endif

template <Geometry g, typename test, typename trial, int q, typename lambda>
void cpu_batched_kernel(const double* inputs, double* outputs, const double* jacobians, size_t num_elements, lambda qf)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  static constexpr TensorProductQuadratureRule<q> rule{};

  auto u = reinterpret_cast<const typename trial_element::dof_type*>(inputs);
  auto r = reinterpret_cast<typename test_element::dof_type*>(outputs);
  auto J = reinterpret_cast<const typename batched_jacobian<g, q>::type*>(jacobians);

  // for each element in the domain
  for (size_t e = 0; e < num_elements; e++) {
    // load the element values and jacobians for this element
    auto u_e = u[e];
    auto J_e = J[e];

    // (batch) interpolate each quadrature point's value
    auto [values, derivatives] = trial_element::interpolate(u_e, J_e, rule);

    // (batch) evalute the q-function at each quadrature point
    auto [sources, fluxes] = batch_apply_qf(qf, values, derivatives);

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(sources, fluxes, J_e, rule, r[e]);
  }
}

template <typename simd_type, int q, int n>
void cpu_batched_kernel_simd(const tensor<tensor<simd_type, 1>, n, n, n>* u, tensor<tensor<simd_type, 1>, n, n, n>* r,
                             const size_t num_element_blocks, tensor<simd_type, q, q, q>* D1,
                             tensor<tensor<simd_type, 3, 3>, q, q, q>* D2)
{
  constexpr Geometry g = Geometry::Hexahedron;
  using test           = H1<n - 1>;
  using trial          = H1<n - 1>;
  using test_element   = finite_element<g, test>;
  using trial_element  = finite_element<g, trial>;

  static constexpr TensorProductQuadratureRule<q> rule{};

  // for each element in the domain
  for (size_t e = 0; e < num_element_blocks; e++) {
    // (batch) interpolate each quadrature point's value
    auto [values, derivatives] = trial_element::interpolate(u[e], rule);

    // (batch) evalute the q-function at each quadrature point
    auto [sources, fluxes] = batch_apply_PA_simd(values, derivatives, D1[e], D2[e]);

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(sources, fluxes, rule, r[e]);
  }
}

}  // namespace serac

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}  // namespace compiler

template <typename lambda>
auto time(lambda&& f)
{
  axom::utilities::Timer stopwatch;
  stopwatch.start();
  f();
  stopwatch.stop();
  return stopwatch.elapsed();
}

template <int p, int q>
void h1_h1_test_2D(int num_elements, int num_runs)
{
  using serac::Geometry;
  using serac::H1;

  constexpr int n   = p + 1;
  constexpr int dim = 2;

  double rho = 1.0;
  double k   = 1.0;

  using test  = H1<p>;
  using trial = H1<p>;

  auto mass_plus_diffusion = [=](auto input) {
    auto [u, du_dx] = input;
    auto source     = rho * u;
    auto flux       = k * du_dx;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * n * n);
  mfem::Vector R1D(num_elements * n * n);
  mfem::Vector J1D(num_elements * dim * dim * q * q);
  mfem::Vector rho_dv_1D(num_elements * q * q);
  mfem::Vector k_invJ_invJT_dv_1D(num_elements * dim * dim * q * q);

  auto U               = mfem::Reshape(U1D.ReadWrite(), n, n, num_elements);
  auto J               = mfem::Reshape(J1D.ReadWrite(), q * q, dim, dim, num_elements);
  auto rho_dv          = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Quadrilateral, q> rule;

  for (int e = 0; e < num_elements; e++) {
    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        U(iy, ix, e) = 0.1 * distribution(generator);
      }
    }

    for (int i = 0; i < q * q; i++) {
      serac::tensor<double, dim, dim> J_q{};

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = J_q[r][c] = (r == c) + 0.1 * distribution(generator);
        }
      }

      int qx = i % q;
      int qy = i / q;

      double qweight    = rule.weight(qx, qy);
      auto   invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      double dv         = det(J_q) * qweight;

      rho_dv(i, e) = rho * dv;
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          k_invJ_invJT_dv(i, r, c, e) = k * invJ_invJT[r][c] * dv;
        }
      }
    }
  }

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_kernel<Geometry::Quadrilateral, test, trial, q>(U1D, R1D, J1D, num_elements,
                                                                         mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::cpu_batched_kernel<Geometry::Quadrilateral, test, trial, q>(U1D.Read(), R1D.ReadWrite(), J1D.Read(),
                                                                           num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average cpu batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_cpu_batched_kernel = R1D;
  auto error                     = answer_reference;
  error -= answer_cpu_batched_kernel;
  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

#if 0
  {
    R1D                           = 0.0;
    bool                symmetric = false;
    mfem::Array<double> b_(n * q);
    mfem::Array<double> bt_(n * q);
    mfem::Array<double> g_(n * q);
    mfem::Array<double> gt_(n * q);
    auto                B  = mfem::Reshape(b_.ReadWrite(), q, n);
    auto                Bt = mfem::Reshape(bt_.ReadWrite(), n, q);

    auto G  = mfem::Reshape(g_.ReadWrite(), q, n);
    auto Gt = mfem::Reshape(gt_.ReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto value      = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bt(j, i) = B(i, j) = value[j];
        Gt(j, i) = G(i, j) = derivative[j];
      }
    }

    double mass_runtime = time([&]() {
                            for (int i = 0; i < num_runs; i++) {
                              mfem::SmemPAMassApply3D<n, q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
                              compiler::please_do_not_optimize_away(&R1D);
                            }
                          });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double diffusion_runtime =
        time([&]() {
          for (int i = 0; i < num_runs; i++) {
            mfem::SmemPADiffusionApply3D<n, q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
            compiler::please_do_not_optimize_away(&R1D);
          }
        });
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs << std::endl;
  }
  auto answer_mfem = R1D;
  error            = answer_reference;
  error -= answer_mfem;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
#endif
}

template <int p, int q>
void h1_h1_test_3D(int num_elements, int num_runs)
{
  using serac::Geometry;
  using serac::H1;
  using serac::tensor;

  constexpr int SIMD_WIDTH = 8;
  using simd_type          = typename simd<SIMD_WIDTH>::type;

  constexpr int n   = p + 1;
  constexpr int dim = 3;

  double rho = 1.0;
  double k   = 1.0;

  using test  = H1<p>;
  using trial = H1<p>;

  auto mass_plus_diffusion = [=](auto input) {
    auto [u, du_dx] = input;
    auto source     = rho * u;
    auto flux       = k * du_dx;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * n * n * n);
  mfem::Vector R1D(num_elements * n * n * n);
  mfem::Vector J1D(num_elements * dim * dim * q * q * q);
  mfem::Vector rho_dv_1D(num_elements * q * q * q);
  mfem::Vector k_invJ_invJT_dv_1D(num_elements * dim * dim * q * q * q);

  auto U               = mfem::Reshape(U1D.ReadWrite(), n, n, n, num_elements);
  auto J               = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_dv          = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Hexahedron, q> rule;

  for (int e = 0; e < num_elements; e++) {
    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          U(iz, iy, ix, e) = 0.1 * distribution(generator);
        }
      }
    }

    for (int i = 0; i < q * q * q; i++) {
      serac::tensor<double, dim, dim> J_q{};

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = J_q[r][c] = (r == c) + 0.1 * distribution(generator);
        }
      }

      int qx = i % q;
      int qy = (i % (q * q)) / q;
      int qz = i / (q * q);

      double qweight    = rule.weight(qx, qy, qz);
      auto   invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      double dv         = det(J_q) * qweight;

      rho_dv(i, e) = rho * dv;
      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          k_invJ_invJT_dv(i, r, c, e) = k * invJ_invJT[r][c] * dv;
        }
      }
    }
  }

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::cpu_batched_kernel<Geometry::Hexahedron, test, trial, q>(U1D.Read(), R1D.ReadWrite(), J1D.Read(),
                                                                        num_elements, mass_plus_diffusion);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average cpu batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_cpu_batched_kernel = R1D;
  auto error                     = answer_reference;
  error -= answer_cpu_batched_kernel;
  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  int                                                 element_blocks = (num_elements + SIMD_WIDTH - 1) / SIMD_WIDTH;
  std::vector<tensor<tensor<simd_type, 1>, n, n, n> > U_SIMD(element_blocks);
  std::vector<tensor<tensor<simd_type, 1>, n, n, n> > R_SIMD(element_blocks, tensor<tensor<simd_type, 1>, n, n, n>{});
  std::vector<tensor<simd_type, q, q, q> >            D1_SIMD(element_blocks);
  std::vector<tensor<tensor<simd_type, 3, 3>, q, q, q> > D2_SIMD(element_blocks);

  for (int b = 0; b < element_blocks; b++) {
    for (int ix = 0; ix < n; ix++) {
      for (int iy = 0; iy < n; iy++) {
        for (int iz = 0; iz < n; iz++) {
          for (int s = 0; s < SIMD_WIDTH; s++) {
            int e = b * SIMD_WIDTH + s;
            if (e < num_elements) {
              U_SIMD[b](ix, iy, iz)[0][s] = U(iz, iy, ix, e);
            } else {
              U_SIMD[b](ix, iy, iz)[0][s] = 0.0;
            }
          }
        }
      }
    }

    for (int i = 0; i < q * q * q; i++) {
      int qx = i % q;
      int qy = (i % (q * q)) / q;
      int qz = i / (q * q);

      for (int s = 0; s < SIMD_WIDTH; s++) {
        int e = b * SIMD_WIDTH + s;
        if (e < num_elements) {
          D1_SIMD[b](qz, qy, qx)[s] = rho_dv(i, e);
        }
      }

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          for (int s = 0; s < SIMD_WIDTH; s++) {
            int e = b * SIMD_WIDTH + s;
            if (e < num_elements) {
              D2_SIMD[b](qz, qy, qx)[r][c][s] = k_invJ_invJT_dv(i, r, c, e);
            }
          }
        }
      }
    }
  }

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::cpu_batched_kernel_simd(U_SIMD.data(), R_SIMD.data(), element_blocks, D1_SIMD.data(), D2_SIMD.data());
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average simd batched kernel time: " << runtime / num_runs << std::endl;

    auto R = mfem::Reshape(R1D.ReadWrite(), n, n, n, num_elements);
    for (int b = 0; b < element_blocks; b++) {
      for (int ix = 0; ix < n; ix++) {
        for (int iy = 0; iy < n; iy++) {
          for (int iz = 0; iz < n; iz++) {
            for (int s = 0; s < SIMD_WIDTH; s++) {
              int e = b * SIMD_WIDTH + s;
              if (e < num_elements) {
                R(ix, iy, iz, e) = R_SIMD[b](iz, iy, ix)[0][s];
              }
            }
          }
        }
      }
    }
  }
  auto answer_cpu_batched_kernel_simd = R1D;
  error                               = answer_reference;
  error -= answer_cpu_batched_kernel_simd;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D                           = 0.0;
    bool                symmetric = false;
    mfem::Array<double> b_(n * q);
    mfem::Array<double> bt_(n * q);
    mfem::Array<double> g_(n * q);
    mfem::Array<double> gt_(n * q);
    auto                B  = mfem::Reshape(b_.ReadWrite(), q, n);
    auto                Bt = mfem::Reshape(bt_.ReadWrite(), n, q);

    auto G  = mfem::Reshape(g_.ReadWrite(), q, n);
    auto Gt = mfem::Reshape(gt_.ReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto value      = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bt(j, i) = B(i, j) = value[j];
        Gt(j, i) = G(i, j) = derivative[j];
      }
    }

    double mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPAMassApply3D<n, q>(num_elements, b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double diffusion_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::SmemPADiffusionApply3D<n, q>(num_elements, symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs << std::endl;
  }
  auto answer_mfem = R1D;
  error            = answer_reference;
  error -= answer_mfem;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

template <int p, int q>
void hcurl_hcurl_test_2D(int num_elements, int num_runs)
{
  using serac::Geometry;
  using serac::Hcurl;

  constexpr int dim = 2;

  double rho = 1.0;
  double k   = 1.0;

  using test  = Hcurl<p>;
  using trial = Hcurl<p>;

  using trial_element = serac::finite_element<Geometry::Quadrilateral, trial>;
  using test_element  = serac::finite_element<Geometry::Quadrilateral, test>;

  auto mass_plus_curlcurl = [=](auto input) {
    auto [u, curl_u] = input;
    auto source      = rho * u;
    auto flux        = k * curl_u;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * trial_element::ndof);
  mfem::Vector R1D(num_elements * test_element::ndof);
  mfem::Vector J1D(num_elements * dim * dim * q * q);

  auto U = mfem::Reshape(U1D.ReadWrite(), trial_element::ndof, num_elements);
  auto J = mfem::Reshape(J1D.ReadWrite(), q * q, dim, dim, num_elements);

  for (int e = 0; e < num_elements; e++) {
    for (int i = 0; i < trial_element::ndof; i++) {
      U(i, e) = 0.1 * distribution(generator);
    }

    for (int i = 0; i < q * q; i++) {
      serac::tensor<double, dim, dim> J_q{};

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = J_q[r][c] = (r == c) + 0.1 * distribution(generator);
        }
      }
    }
  }

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_kernel<Geometry::Quadrilateral, test, trial, q>(U1D, R1D, J1D, num_elements,
                                                                         mass_plus_curlcurl);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::cpu_batched_kernel<Geometry::Quadrilateral, test, trial, q>(U1D.Read(), R1D.ReadWrite(), J1D.Read(),
                                                                           num_elements, mass_plus_curlcurl);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average cpu batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_cpu_batched_kernel = R1D;
  auto error                     = answer_reference;
  error -= answer_cpu_batched_kernel;
  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

#if 0
  {
    R1D            = 0.0;
    bool symmetric = false;

    // I think the "o" and "c" are supposed to be short for
    // "open" and "closed", referring to placing interpolation
    // nodes at the gauss-legendre, and gauss-lobatto points (respectively)
    mfem::Array<double> bo_((n - 1) * q);
    mfem::Array<double> bc_(n * q);
    mfem::Array<double> bot_((n - 1) * q);
    mfem::Array<double> bct_(n * q);
    mfem::Array<double> gc_(n * q);
    mfem::Array<double> gct_(n * q);

    auto Bo  = mfem::Reshape(bo_.ReadWrite(), q, n - 1);
    auto Bc  = mfem::Reshape(bc_.ReadWrite(), q, n);
    auto Bot = mfem::Reshape(bot_.ReadWrite(), n - 1, q);
    auto Bct = mfem::Reshape(bct_.ReadWrite(), n, q);
    auto Gc  = mfem::Reshape(gc_.ReadWrite(), q, n);
    auto Gct = mfem::Reshape(gct_.ReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto lobatto_value      = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto lobatto_derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bct(j, i) = Bc(i, j) = lobatto_value[j];
        Gct(j, i) = Gc(i, j) = lobatto_derivative[j];
      }

      auto legendre_value = serac::GaussLegendreInterpolation<n - 1>(rule.points_1D[i]);
      for (int j = 0; j < n - 1; j++) {
        Bot(j, i) = Bo(i, j) = legendre_value[j];
      }
    }

    double mass_runtime = time([&]() {
                            for (int i = 0; i < num_runs; i++) {
                              mfem::PAHcurlMassApply3D(n, q, num_elements, symmetric = false, bo_, bc_, bot_, bct_,
                                                       rho_invJ_invJT_dv_1D, U1D, R1D);
                              compiler::please_do_not_optimize_away(&R1D);
                            }
                          });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double curlcurl_runtime = time([&]() {
                                 for (int i = 0; i < num_runs; i++) {
                                   mfem::PACurlCurlApply3D<n, q>(n, q, symmetric = false, num_elements, bo_, bc_, bot_,
                                                                 bct_, gc_, gct_, k_JTJ_dv_over_detJsq_1D, U1D, R1D);
                                   compiler::please_do_not_optimize_away(&R1D);
                                 }
                               });
    std::cout << "average mfem curlcurl kernel time: " << curlcurl_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + curlcurl_runtime) / num_runs << std::endl;
  }
  auto answer_mfem = R1D;
  error            = answer_reference;
  error -= answer_mfem;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
#endif
}

template <int p, int q>
void hcurl_hcurl_test_3D(int num_elements, int num_runs)
{
  using serac::Geometry;
  using serac::Hcurl;

  constexpr int n   = p + 1;
  constexpr int dim = 3;

  double rho = 1.0;
  double k   = 1.0;

  using test  = Hcurl<p>;
  using trial = Hcurl<p>;

  using trial_element = serac::finite_element<Geometry::Hexahedron, trial>;
  using test_element  = serac::finite_element<Geometry::Hexahedron, test>;

  auto mass_plus_curlcurl = [=](auto input) {
    auto [u, curl_u] = input;
    auto source      = rho * u;
    auto flux        = k * curl_u;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(num_elements * trial_element::ndof);
  mfem::Vector R1D(num_elements * test_element::ndof);
  mfem::Vector J1D(num_elements * dim * dim * q * q * q);
  mfem::Vector rho_invJ_invJT_dv_1D(num_elements * dim * dim * q * q * q);
  mfem::Vector k_JTJ_dv_over_detJsq_1D(num_elements * dim * dim * q * q * q);

  auto U                    = mfem::Reshape(U1D.ReadWrite(), trial_element::ndof, num_elements);
  auto J                    = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_invJ_invJT_dv    = mfem::Reshape(rho_invJ_invJT_dv_1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto k_JTJ_dv_over_detJsq = mfem::Reshape(k_JTJ_dv_over_detJsq_1D.ReadWrite(), q * q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Hexahedron, q> rule;

  for (int e = 0; e < num_elements; e++) {
    for (int i = 0; i < trial_element::ndof; i++) {
      U(i, e) = 0.1 * distribution(generator);
    }

    for (int i = 0; i < q * q * q; i++) {
      serac::tensor<double, dim, dim> J_q{};

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          J(i, r, c, e) = J_q[r][c] = (r == c) + 0.1 * distribution(generator);
        }
      }

      int qx = i % q;
      int qy = (i % (q * q)) / q;
      int qz = i / (q * q);

      double qweight    = rule.weight(qx, qy, qz);
      auto   JTJ        = dot(transpose(J_q), J_q);
      auto   invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      auto   detJ       = det(J_q);
      double dv         = det(J_q) * qweight;

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          k_JTJ_dv_over_detJsq(i, r, c, e) = k * (JTJ[r][c] / (detJ * detJ)) * dv;
          rho_invJ_invJT_dv(i, r, c, e)    = rho * invJ_invJT[r][c] * dv;
        }
      }
    }
  }

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::reference_kernel<Geometry::Hexahedron, test, trial, q>(U1D, R1D, J1D, num_elements, mass_plus_curlcurl);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_reference = R1D;

  {
    R1D            = 0.0;
    double runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        serac::cpu_batched_kernel<Geometry::Hexahedron, test, trial, q>(U1D.Read(), R1D.ReadWrite(), J1D.Read(),
                                                                        num_elements, mass_plus_curlcurl);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average cpu batched kernel time: " << runtime / num_runs << std::endl;
  }
  auto answer_cpu_batched_kernel = R1D;
  auto error                     = answer_reference;
  error -= answer_cpu_batched_kernel;
  double relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;

  {
    R1D            = 0.0;
    bool symmetric = false;

    // I think the "o" and "c" are supposed to be short for
    // "open" and "closed", referring to placing interpolation
    // nodes at the gauss-legendre, and gauss-lobatto points (respectively)
    mfem::Array<double> bo_((n - 1) * q);
    mfem::Array<double> bc_(n * q);
    mfem::Array<double> bot_((n - 1) * q);
    mfem::Array<double> bct_(n * q);
    mfem::Array<double> gc_(n * q);
    mfem::Array<double> gct_(n * q);

    auto Bo  = mfem::Reshape(bo_.ReadWrite(), q, n - 1);
    auto Bc  = mfem::Reshape(bc_.ReadWrite(), q, n);
    auto Bot = mfem::Reshape(bot_.ReadWrite(), n - 1, q);
    auto Bct = mfem::Reshape(bct_.ReadWrite(), n, q);
    auto Gc  = mfem::Reshape(gc_.ReadWrite(), q, n);
    auto Gct = mfem::Reshape(gct_.ReadWrite(), n, q);

    for (int i = 0; i < q; i++) {
      auto lobatto_value      = serac::GaussLobattoInterpolation<n>(rule.points_1D[i]);
      auto lobatto_derivative = serac::GaussLobattoInterpolationDerivative<n>(rule.points_1D[i]);

      for (int j = 0; j < n; j++) {
        Bct(j, i) = Bc(i, j) = lobatto_value[j];
        Gct(j, i) = Gc(i, j) = lobatto_derivative[j];
      }

      auto legendre_value = serac::GaussLegendreInterpolation<n - 1>(rule.points_1D[i]);
      for (int j = 0; j < n - 1; j++) {
        Bot(j, i) = Bo(i, j) = legendre_value[j];
      }
    }

    double mass_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PAHcurlMassApply3D(n, q, num_elements, symmetric = false, bo_, bc_, bot_, bct_, rho_invJ_invJT_dv_1D, U1D,
                                 R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs << std::endl;

    double curlcurl_runtime = time([&]() {
      for (int i = 0; i < num_runs; i++) {
        mfem::PACurlCurlApply3D<n, q>(n, q, symmetric = false, num_elements, bo_, bc_, bot_, bct_, gc_, gct_,
                                      k_JTJ_dv_over_detJsq_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem curlcurl kernel time: " << curlcurl_runtime / num_runs << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + curlcurl_runtime) / num_runs << std::endl;
  }
  auto answer_mfem = R1D;
  error            = answer_reference;
  error -= answer_mfem;
  relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

int main()
{
  int num_runs     = 10;
  int num_elements = 10000;
  h1_h1_test_2D<1 /* polynomial order */, 2 /* quadrature points / dim */>(num_elements, num_runs);
  h1_h1_test_2D<2 /* polynomial order */, 3 /* quadrature points / dim */>(num_elements, num_runs);
  h1_h1_test_2D<3 /* polynomial order */, 4 /* quadrature points / dim */>(num_elements, num_runs);

  h1_h1_test_3D<1 /* polynomial order */, 2 /* quadrature points / dim */>(num_elements, num_runs);
  h1_h1_test_3D<2 /* polynomial order */, 3 /* quadrature points / dim */>(num_elements, num_runs);
  h1_h1_test_3D<3 /* polynomial order */, 4 /* quadrature points / dim */>(num_elements, num_runs);

  hcurl_hcurl_test_2D<1 /* polynomial order */, 2 /* quadrature points / dim */>(num_elements, num_runs);
  hcurl_hcurl_test_2D<2 /* polynomial order */, 3 /* quadrature points / dim */>(num_elements, num_runs);
  hcurl_hcurl_test_2D<3 /* polynomial order */, 4 /* quadrature points / dim */>(num_elements, num_runs);

  hcurl_hcurl_test_3D<1 /* polynomial order */, 2 /* quadrature points / dim */>(num_elements, num_runs);
  hcurl_hcurl_test_3D<2 /* polynomial order */, 3 /* quadrature points / dim */>(num_elements, num_runs);
  hcurl_hcurl_test_3D<3 /* polynomial order */, 4 /* quadrature points / dim */>(num_elements, num_runs);
}
