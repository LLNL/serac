// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <random>
#include <chrono>

#include "axom/slic.hpp"
#include "axom/core/utilities/Timer.hpp"

#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

static constexpr auto I = Identity<3>();

/**
 * @brief a 3D constitutive model for a J2 material with linear isotropic and kinematic hardening.
 */
struct J2 {
  double E;        ///< Young's modulus
  double nu;       ///< Poisson's ratio
  double Hi;       ///< isotropic hardening constant
  double Hk;       ///< kinematic hardening constant
  double sigma_y;  ///< yield stress

  /** @brief variables describing the stress state, yield surface, and some information about the most recent stress
   * increment */
  struct State {
    tensor<double, 3, 3> beta;           ///< back-stress tensor
    tensor<double, 3, 3> el_strain;      ///< elastic strain
    double               pl_strain;      ///< plastic strain
    double               pl_strain_inc;  ///< incremental plastic strain
    double               q;              ///< (trial) J2 stress
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  auto calculate_stress(const tensor<double, 3, 3> grad_u, State& state) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    state.el_strain          = sym(grad_u);
    double               p   = K * tr(state.el_strain);
    tensor<double, 3, 3> s   = 2.0 * G * dev(state.el_strain);
    tensor<double, 3, 3> eta = s - state.beta;
    state.q                  = sqrt(3.0 / 2.0) * norm(eta);
    double phi               = state.q - (sigma_y + Hi * state.pl_strain);

    // see (7.207) on pg. 261
    state.pl_strain_inc = fmax(0.0, phi / (3 * G + Hk + Hi));

    // (ii) admissibility
    if (state.pl_strain_inc > 0.0) {
      // (iii) return mapping
      s = s - sqrt(6.0) * G * state.pl_strain_inc * normalize(eta);

      state.pl_strain = state.pl_strain + state.pl_strain_inc;
      state.el_strain = (s / (2.0 * G)) + ((p / K) * I);

      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * state.pl_strain_inc * normalize(eta);
    }

    return s + p * I;
  }

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto calculate_stress_AD(const T grad_u, State& state) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    auto el_strain = sym(grad_u);
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto eta       = s - state.beta;
    auto q         = sqrt(3.0 / 2.0) * norm(eta);
    auto phi       = q - (sigma_y + Hi * state.pl_strain);

    // (ii) admissibility
    if (phi > 0.0) {
      // see (7.207) on pg. 261
      auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

      // (iii) return mapping
      s = s - sqrt(6.0) * G * plastic_strain_inc * normalize(eta);

      state.pl_strain = state.pl_strain + get_value(plastic_strain_inc);

      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * get_value(plastic_strain_inc) * normalize(get_value(eta));
    }

    return s + p * I;
  }

  /** @brief calculate the gradient of Cauchy stress w.r.t. grad_u */
  auto calculate_gradient(const State& state) const
  {
    double K = E / (3.0 * (1.0 - 2.0 * nu));
    double G = 0.5 * E / (1.0 + nu);

    double A1 = 2.0 * G;
    double A2 = 0.0;

    tensor<double, 3, 3> N{};

    if (state.pl_strain_inc > 0.0) {
      tensor<double, 3, 3> s = 2.0 * G * dev(state.el_strain);
      N                      = normalize(s - state.beta);

      A1 -= 6 * G * G * state.pl_strain_inc / state.q;
      A2 = 6 * G * G * ((state.pl_strain_inc / state.q) - (1.0 / (3.0 * G + Hi + Hk)));
    }

    return make_tensor<3, 3, 3, 3>([&](auto i, auto j, auto k, auto l) {
      double I4    = (i == j) * (k == l);
      double I4sym = 0.5 * ((i == k) * (j == l) + (i == l) * (j == k));
      double I4dev = I4sym - (i == j) * (k == l) / 3.0;
      return K * I4 + A1 * I4dev + A2 * N(i, j) * N(k, l);
    });
  }
};

// impose some arbitrary time-dependent deformation
auto displacement_gradient(double t)
{
  /** CUDA WORKAROUND deduction guide failed **/
  return 10 * tensor<double, 3, 3>{{
                  {sin(t), 0.0, 0.0},
                  {0.0, t, exp(t) - 1},
                  {0.0, 0.0, t * t},
              }};
}

int main()
{
  // create & initialize test logger, finalized when exiting scope
  axom::slic::SimpleLogger logger;

  axom::utilities::Timer stopwatch;
  double                 J2_evaluation_time = 0.0;
  double                 J2_gradient_time   = 0.0;
  double                 J2_AD_time         = 0.0;

  double t  = 0.0;
  double dt = 0.0001;

  J2 material{
      100,   // Young's modulus
      0.25,  // Poisson's ratio
      1.0,   // isotropic hardening constant
      2.3,   // kinematic hardening constant
      300.0  // yield stress
  };

  J2::State state{};

  while (t < 1.0) {
    auto grad_u = displacement_gradient(t);

    auto backup = state;

    stopwatch.start();
    tensor<double, 3, 3> stress = material.calculate_stress(grad_u, state);
    stopwatch.stop();
    J2_evaluation_time += stopwatch.elapsed();

    stopwatch.start();
    tensor<double, 3, 3, 3, 3> C = material.calculate_gradient(state);
    stopwatch.stop();
    J2_gradient_time += stopwatch.elapsed();

    stopwatch.start();
    auto stress_and_C = material.calculate_stress_AD(make_dual(grad_u), backup);
    stopwatch.stop();
    J2_AD_time += stopwatch.elapsed();

    bool error_too_big =
        (norm(stress - get_value(stress_and_C)) > 1.0e-12) || (norm(C - get_gradient(stress_and_C)) > 1.0e-12) ||
        (norm(state.beta - backup.beta) > 1.0e-12) || (fabs(state.pl_strain - backup.pl_strain) > 1.0e-12);

    if (error_too_big) {
      SLIC_ERROR("Significant difference between expected and actual results. Exiting...");
      exit(1);
    }

    t += dt;
  }

  std::cout << "total J2 evaluation time (no AD): " << J2_evaluation_time << std::endl;
  std::cout << "total J2 gradient time (no AD): " << J2_gradient_time << std::endl;
  std::cout << "total J2 evaluation+gradient time (AD): " << J2_AD_time << std::endl;
  std::cout << "(AD time) / (manual gradient time): " << J2_AD_time / (J2_evaluation_time + J2_gradient_time)
            << std::endl;
}

// timings on my local machine:
//
// total J2 evaluation time (no AD):       0.0196884
// total J2 gradient time (no AD):         0.0439456
// total J2 evaluation+gradient time (AD): 0.256943
// (AD time) / (manual gradient time):     4.03782
//
// so, manually implementing derivatives is faster (as expected)
