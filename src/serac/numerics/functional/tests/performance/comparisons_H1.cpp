#include "mfem.hpp"

#include "mfem_EA_kernels_h1.hpp"
#include "mfem_PA_kernels_h1.hpp"

#include "axom/core/utilities/Timer.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/quadrature_data.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"

#ifdef USE_OLD_IMPLEMENTATION
#include "serac/numerics/functional/old/domain_integral_kernels.hpp"
#else
#include "serac/numerics/functional/domain_integral_kernels.hpp"
#endif

#include <vector>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
} // namespace compiler

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

}

template <int p, int q>
void h1_h1_test_2D(size_t num_elements, size_t num_runs)
{
  using serac::Geometry;
  using serac::H1;

  constexpr int dim = 2;
  constexpr int n   = p + 1;

  const double num_runs_d = static_cast<double>(num_runs);

  double rho = 1.0;
  double k   = 1.0;

  using test  = H1<p>;
  using trial = H1<p>;

  // a mass + diffusion qfunction
  auto qf = [=](auto /*x*/, auto temperature) {
    auto [u, du_dx] = temperature;
    auto source     = rho * u;
    auto flux       = k * du_dx;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(static_cast<int>(num_elements * n * n));
  mfem::Vector R1D(static_cast<int>(num_elements * n * n));
  mfem::Vector J1D(static_cast<int>(num_elements * dim * dim * q * q));
  mfem::Vector rho_dv_1D(static_cast<int>(num_elements * q * q));
  mfem::Vector k_invJ_invJT_dv_1D(static_cast<int>(num_elements * dim * dim * q * q));

  auto U               = mfem::Reshape(U1D.ReadWrite(), n, n, num_elements);
  auto J               = mfem::Reshape(J1D.ReadWrite(), q * q, dim, dim, num_elements);
  auto rho_dv          = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Quadrilateral, q> rule;

  for (size_t e = 0; e < num_elements; e++) {
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
    R1D = 0.0;

    auto X1D = U1D;

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
                            for (size_t i = 0; i < num_runs; i++) {
                              mfem::SmemPAMassApply2D<n, q>(static_cast<int>(num_elements), b_, bt_, rho_dv_1D, U1D, R1D);
                              compiler::please_do_not_optimize_away(&R1D);
                            }
                          });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs_d << std::endl;

    double diffusion_runtime =
        time([&]() {
          for (size_t i = 0; i < num_runs; i++) {
            mfem::SmemPADiffusionApply2D<n, q>(static_cast<int>(num_elements), symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
            compiler::please_do_not_optimize_away(&R1D);
          }
        });
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs_d << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs_d << std::endl;
  }
  auto answer_mfem = R1D;
  auto error            = answer_reference;
  error -= answer_mfem;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

template <int p, int q>
void h1_h1_test_EA_2D(size_t num_elements, size_t num_runs)
{
  using serac::Geometry;
  using serac::H1;

  constexpr int dim = 2;
  constexpr int n   = p + 1;

  const double num_runs_d = static_cast<double>(num_runs);

  double rho = 0.0;
  double k   = 1.0;

  using test  = H1<p>;
  using trial = H1<p>;

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector KE_1D(static_cast<int>(num_elements * n * n * n * n));
  mfem::Vector J1D(static_cast<int>(num_elements * dim * dim * q * q));
  mfem::Vector rho_dv_1D(static_cast<int>(num_elements * q * q));
  mfem::Vector k_invJ_invJT_dv_1D(static_cast<int>(num_elements * dim * dim * q * q));

  auto J               = mfem::Reshape(J1D.ReadWrite(), q * q, dim, dim, num_elements);
  auto rho_dv          = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q, 3, num_elements);

  using derivative_type = serac::tuple <
    serac::tuple < double, serac::zero >,
    serac::tuple < serac::zero, serac::tensor< double, dim, dim > >
  >;

  serac::CPUArray<derivative_type, 2> derivatives(num_elements, q * q);

  serac::GaussLegendreRule<Geometry::Quadrilateral, q> rule;

  for (size_t e = 0; e < num_elements; e++) {

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
      k_invJ_invJT_dv(i, 0, e) = k * invJ_invJT[0][0] * dv;
      k_invJ_invJT_dv(i, 1, e) = k * invJ_invJT[0][1] * dv;
      k_invJ_invJT_dv(i, 2, e) = k * invJ_invJT[1][1] * dv;

      derivatives(e, i) = {{rho, serac::zero{}}, {serac::zero{}, k * serac::DenseIdentity<dim>()}};
    }
  }

  {
    KE_1D = 0.0;

    double runtime = time([&]() {
      for (size_t i = 0; i < num_runs; i++) {
        serac::domain_integral::element_gradient_kernel<Geometry::Quadrilateral, test, trial, q>(
          serac::CPUArrayView<double, 3>(KE_1D.GetData(), num_elements, n * n, n * n),
          view(derivatives), J1D, num_elements);
        compiler::please_do_not_optimize_away(&KE_1D);
      }
    });
    std::cout << "average reference kernel time: " << runtime / num_runs_d << std::endl;
  }
  auto answer_reference = KE_1D;

  {
    KE_1D               = 0.0;
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

    //double mass_runtime = time([&]() {
    //                        for (size_t i = 0; i < num_runs; i++) {
    //                          mfem::SmemPAMassApply2D<n, q>(static_cast<int>(num_elements), b_, bt_, rho_dv_1D, U1D, R1D);
    //                          compiler::please_do_not_optimize_away(&R1D);
    //                        }
    //                      });
    //std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs_d << std::endl;

    double diffusion_runtime =
        time([&]() {
          for (size_t i = 0; i < num_runs; i++) {
            mfem::EADiffusionAssemble2D<n, q>(static_cast<int>(num_elements), b_, g_, k_invJ_invJT_dv_1D, KE_1D, true);
            compiler::please_do_not_optimize_away(&KE_1D);
          }
        });
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs_d << std::endl;

    //std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs_d << std::endl;
  }
  auto answer_mfem = KE_1D;
  auto error       = answer_reference;
  error -= answer_mfem;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

template <int p, int q>
void h1_h1_test_3D(size_t num_elements, size_t num_runs)
{
  using serac::Geometry;
  using serac::H1;
  using serac::tensor;

  constexpr int n   = p + 1;
  constexpr int dim = 3;

  const double num_runs_d = static_cast<double>(num_runs);

  double rho = 1.0;
  double k   = 1.0;

  using test  = H1<p>;
  using trial = H1<p>;

  // a mass + diffusion qfunction
  auto qf = [=](auto /*x*/, auto temperature) {
    auto [u, du_dx] = temperature;
    auto source     = rho * u;
    auto flux       = k * du_dx;
    return serac::tuple{source, flux};
  };

  std::default_random_engine             generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  mfem::Vector U1D(static_cast<int>(num_elements * n * n * n));
  mfem::Vector R1D(static_cast<int>(num_elements * n * n * n));
  mfem::Vector J1D(static_cast<int>(num_elements * dim * dim * q * q * q));
  mfem::Vector rho_dv_1D(static_cast<int>(num_elements * q * q * q));
  mfem::Vector k_invJ_invJT_dv_1D(static_cast<int>(num_elements * dim * dim * q * q * q));

  auto U               = mfem::Reshape(U1D.ReadWrite(), n, n, n, num_elements);
  auto J               = mfem::Reshape(J1D.ReadWrite(), q * q * q, dim, dim, num_elements);
  auto rho_dv          = mfem::Reshape(rho_dv_1D.ReadWrite(), q * q * q, num_elements);
  auto k_invJ_invJT_dv = mfem::Reshape(k_invJ_invJT_dv_1D.ReadWrite(), q * q * q, dim, dim, num_elements);

  serac::GaussLegendreRule<Geometry::Hexahedron, q> rule;

  for (size_t e = 0; e < num_elements; e++) {
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

    auto X1D = U1D;

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
      for (size_t i = 0; i < num_runs; i++) {
        mfem::SmemPAMassApply3D<n, q>(static_cast<int>(num_elements), b_, bt_, rho_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem mass kernel time: " << mass_runtime / num_runs_d << std::endl;

    double diffusion_runtime = time([&]() {
      for (size_t i = 0; i < num_runs; i++) {
        mfem::SmemPADiffusionApply3D<n, q>(static_cast<int>(num_elements), symmetric = false, b_, g_, k_invJ_invJT_dv_1D, U1D, R1D);
        compiler::please_do_not_optimize_away(&R1D);
      }
    });
    std::cout << "average mfem diffusion kernel time: " << diffusion_runtime / num_runs_d << std::endl;

    std::cout << "average mfem combined kernel time: " << (mass_runtime + diffusion_runtime) / num_runs_d << std::endl;
  }
  auto answer_mfem = R1D;
  auto error            = answer_reference;
  error -= answer_mfem;
  auto relative_error = error.Norml2() / answer_reference.Norml2();
  std::cout << "error: " << relative_error << std::endl;
}

int main()
{
  size_t num_runs     = 10;
  size_t num_elements = 1000;

  //h1_h1_test_2D<1 /* polynomial order */, 2 /* quadrature points / dim */>(8 * num_elements, num_runs);
  //h1_h1_test_2D<2 /* polynomial order */, 3 /* quadrature points / dim */>(4 * num_elements, num_runs);
  //h1_h1_test_2D<3 /* polynomial order */, 4 /* quadrature points / dim */>(1 * num_elements, num_runs);

  //h1_h1_test_3D<1 /* polynomial order */, 2 /* quadrature points / dim */>(8 * num_elements, num_runs);
  //h1_h1_test_3D<2 /* polynomial order */, 3 /* quadrature points / dim */>(4 * num_elements, num_runs);
  //h1_h1_test_3D<3 /* polynomial order */, 4 /* quadrature points / dim */>(1 * num_elements, num_runs);

  h1_h1_test_EA_2D< 3, 4 >(num_elements, num_runs);
}
