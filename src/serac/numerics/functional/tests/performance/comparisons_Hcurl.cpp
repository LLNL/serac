#include "mfem.hpp"

#include "mfem_PA_kernels_hcurl.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/quadrature_data.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

#include "serac/numerics/functional/domain_integral_kernels.hpp"

#include <vector>

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

namespace serac {

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

}  // namespace serac

template <int p, int q>
void hcurl_hcurl_test_2D(size_t num_elements, size_t num_runs)
{
  using serac::Geometry;
  using serac::Hcurl;

  constexpr int dim = 2;
  constexpr int n   = p + 1;

  const double num_runs_d = static_cast<double>(num_runs);

  double rho = 1.0;
  double k   = 1.0;

  using test  = Hcurl<p>;
  using trial = Hcurl<p>;

  using trial_element = serac::finite_element<Geometry::Quadrilateral, trial>;
  using test_element  = serac::finite_element<Geometry::Quadrilateral, test>;

  // a mass + curlcurl qfunction
  auto qf = [=](auto /*x*/, auto input) {
    auto [u, curl_u] = input;
    auto source      = rho * u;
    auto flux        = k * curl_u;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector X1D(static_cast<int>(num_elements * dim * q * q));
  mfem::Vector U1D(static_cast<int>(num_elements * trial_element::ndof));
  mfem::Vector R1D(static_cast<int>(num_elements * test_element::ndof));
  mfem::Vector J1D(static_cast<int>(num_elements * dim * dim * q * q));
  mfem::Vector rho_invJ_invJT_dv_1D(static_cast<int>(num_elements * dim * dim * q * q));
  mfem::Vector k_dv_over_detJsq_1D(static_cast<int>(num_elements * q * q));

  X1D                    = 0.0;
  auto U                 = mfem::Reshape(U1D.ReadWrite(), trial_element::ndof, num_elements);
  auto J                 = mfem::Reshape(J1D.ReadWrite(), q * q, dim, dim, num_elements);
  auto rho_invJ_invJT_dv = mfem::Reshape(rho_invJ_invJT_dv_1D.ReadWrite(), q * q, dim, dim, num_elements);
  auto k_dv_over_detJsq  = mfem::Reshape(k_dv_over_detJsq_1D.ReadWrite(), q * q, num_elements);

  serac::GaussLegendreRule<Geometry::Quadrilateral, q> rule;

  for (size_t e = 0; e < num_elements; e++) {
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

      int qx = i % q;
      int qy = i / q;

      double qweight    = rule.weight(qx, qy);
      auto   invJ_invJT = dot(inv(J_q), transpose(inv(J_q)));
      auto   detJ       = det(J_q);
      double dv         = det(J_q) * qweight;

      for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
          rho_invJ_invJT_dv(i, r, c, e) = rho * invJ_invJT[r][c] * dv;
        }
      }
      k_dv_over_detJsq(i, e) = (k * dv) / (detJ * detJ);
    }
  }

  {
    R1D = 0.0;

    serac::domain_integral::KernelConfig<q, Geometry::Quadrilateral, test, trial> eval_config;

    serac::domain_integral::EvaluationKernel element_residual{eval_config, J1D, X1D, num_elements, qf, serac::NoQData};

    // unused anyway, since there is no material state
    bool update_state = false;

    double runtime = time([&]() {
      for (size_t i = 0; i < num_runs; i++) {
        element_residual({&U1D}, R1D, update_state);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs_d << std::endl;
  }
  auto answer_reference = R1D;

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
      for (size_t i = 0; i < num_runs; i++) {
        mfem::PAHcurlMassApply2D(n, q, static_cast<int>(num_elements), symmetric, bo_, bc_, bot_, bct_,
                                 rho_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs_d << std::endl;

    double curlcurl_runtime = time([&]() {
      for (size_t i = 0; i < num_runs; i++) {
        mfem::PACurlCurlApply2D(n, q, static_cast<int>(num_elements), bo_, bot_, gc_, gct_, k_dv_over_detJsq_1D, U1D,
                                R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem curlcurl kernel time: " << curlcurl_runtime / num_runs_d << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + curlcurl_runtime) / num_runs_d << std::endl;
  }
  auto answer_mfem = R1D;
  auto error       = answer_reference;
  error -= answer_mfem;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

template <int p, int q>
void mfem_test_3D(size_t num_elements, size_t num_runs, mfem::Vector& U1D, mfem::Vector& R1D,
                  mfem::Vector& rho_invJ_invJT_dv_1D, mfem::Vector& k_JTJ_dv_over_detJsq_1D)
{
  constexpr int n          = p + 1;
  double        num_runs_d = static_cast<double>(num_runs);

  R1D            = 0.0;
  bool symmetric = false;

  serac::GaussLegendreRule<serac::Geometry::Hexahedron, q> rule;

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
    for (size_t i = 0; i < num_runs; i++) {
      mfem::PAHcurlMassApply3D(n, q, static_cast<int>(num_elements), symmetric = false, bo_, bc_, bot_, bct_,
                               rho_invJ_invJT_dv_1D, U1D, R1D);
      compiler::please_do_not_optimize_away(&R1D);
    }
  });
  std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs_d << std::endl;

  double curlcurl_runtime = time([&]() {
    for (size_t i = 0; i < num_runs; i++) {
      mfem::PACurlCurlApply3D<n, q>(n, q, symmetric = false, static_cast<int>(num_elements), bo_, bc_, bot_, bct_, gc_,
                                    gct_, k_JTJ_dv_over_detJsq_1D, U1D, R1D);
      compiler::please_do_not_optimize_away(&R1D);
    }
  });
  std::cout << "average mfem curlcurl kernel time: " << curlcurl_runtime / num_runs_d << std::endl;

  std::cout << "average mfem combined kernel time: " << (mass_runtime + curlcurl_runtime) / num_runs_d << std::endl;
}

template <int p, int q>
void hcurl_hcurl_test_3D(size_t num_elements, size_t num_runs)
{
  using serac::Geometry;
  using serac::Hcurl;

  [[maybe_unused]] constexpr int n   = p + 1;
  constexpr int                  dim = 3;

  const double num_runs_d = static_cast<double>(num_runs);

  double rho = 1.0;
  double k   = 1.0;

  using test  = Hcurl<p>;
  using trial = Hcurl<p>;

  using trial_element = serac::finite_element<Geometry::Hexahedron, trial>;
  using test_element  = serac::finite_element<Geometry::Hexahedron, test>;

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(static_cast<int>(num_elements * trial_element::ndof));
  mfem::Vector R1D(static_cast<int>(num_elements * test_element::ndof));
  mfem::Vector X1D(static_cast<int>(num_elements * dim * q * q * q));
  mfem::Vector J1D(static_cast<int>(num_elements * dim * dim * q * q * q));
  mfem::Vector rho_invJ_invJT_dv_1D(static_cast<int>(num_elements * dim * dim * q * q * q));
  mfem::Vector k_JTJ_dv_over_detJsq_1D(static_cast<int>(num_elements * dim * dim * q * q * q));

  X1D                       = 0.0;
  auto U                    = mfem::Reshape(U1D.ReadWrite(), trial_element::ndof, num_elements);
  auto J                    = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_invJ_invJT_dv    = mfem::Reshape(rho_invJ_invJT_dv_1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto k_JTJ_dv_over_detJsq = mfem::Reshape(k_JTJ_dv_over_detJsq_1D.ReadWrite(), q * q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Hexahedron, q> rule;

  for (size_t e = 0; e < num_elements; e++) {
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

  // a mass + curlcurl qfunction
  auto qf = [=](auto /*x*/, auto input) {
    auto [u, curl_u] = input;
    auto source      = rho * u;
    auto flux        = k * curl_u;
    return serac::tuple{source, flux};
  };

  {
    R1D = 0.0;

    serac::domain_integral::KernelConfig<q, Geometry::Hexahedron, test, trial> eval_config;

    serac::domain_integral::EvaluationKernel element_residual{eval_config, J1D, X1D, num_elements, qf, serac::NoQData};

    // unused anyway, since there is no material state
    bool update_state = false;

    double runtime = time([&]() {
      for (size_t i = 0; i < num_runs; i++) {
        element_residual({&U1D}, R1D, update_state);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs_d << std::endl;
  }
  auto answer_reference = R1D;

  mfem_test_3D<p, q>(num_elements, num_runs, U1D, R1D, rho_invJ_invJT_dv_1D, k_JTJ_dv_over_detJsq_1D);
  auto answer_mfem = R1D;
  auto error       = answer_reference;
  error -= answer_mfem;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

int main()
{
  size_t num_runs     = 10;
  size_t num_elements = 10000;

  // hcurl_hcurl_test_2D<1 /* polynomial order */, 2 /* quadrature points / dim */>(8 * num_elements, num_runs);
  // hcurl_hcurl_test_2D<2 /* polynomial order */, 3 /* quadrature points / dim */>(4 * num_elements, num_runs);
  // hcurl_hcurl_test_2D<3 /* polynomial order */, 4 /* quadrature points / dim */>(1 * num_elements, num_runs);

  // hcurl_hcurl_test_3D<1 /* polynomial order */, 2 /* quadrature points / dim */>(8 * num_elements, num_runs);
  // hcurl_hcurl_test_3D<2 /* polynomial order */, 3 /* quadrature points / dim */>(4 * num_elements, num_runs);
  hcurl_hcurl_test_3D<3 /* polynomial order */, 4 /* quadrature points / dim */>(1 * num_elements, num_runs);
}
