#include "tensor.hpp"
#include "tuple_tensor_dual_functions.hpp"

#pragma once

namespace serac {

struct Empty {
};

/**
 * @brief Power-law isotropic hardening law
 */
struct PowerLawHardening {
  double sigma_y;  ///< yield strength
  double n;        ///< hardening index in reciprocal form
  double eps0;     ///< reference value of accumulated plastic strain

  /**
   * @brief Computes the flow stress
   *
   * @tparam T Number-like type for the argument
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T>
  auto operator()(const T accumulated_plastic_strain) const
  {
    using std::pow;
    return sigma_y * pow(1.0 + accumulated_plastic_strain / eps0, 1.0 / n);
  };
};

/**
 * @brief Voce's isotropic hardening law
 *
 * This form has an exponential saturation character.
 */
struct VoceHardening {
  double sigma_y;          ///< yield strength
  double sigma_sat;        ///< saturation value of flow strength
  double strain_constant;  ///< The constant dictating how fast the exponential decays

  /**
   * @brief Computes the flow stress
   *
   * @tparam T Number-like type for the argument
   * @param accumulated_plastic_strain The uniaxial equivalent accumulated plastic strain
   * @return Flow stress value
   */
  template <typename T>
  auto operator()(const T accumulated_plastic_strain) const
  {
    using std::exp;
    return sigma_sat - (sigma_sat - sigma_y) * exp(-accumulated_plastic_strain / strain_constant);
  };
};

/// @brief J2 material with nonlinear isotropic hardening.
template <typename HardeningType>
struct J2Nonlinear {
  static constexpr int    dim = 3;      ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double        E;          ///< Young's modulus
  double        nu;         ///< Poisson's ratio
  HardeningType hardening;  ///< Flow stress hardening model
  double        density;    ///< mass density

  /// @brief variables required to characterize the hysteresis response
  using State = tensor<double, 10>;
  using UnpackedState = tuple<tensor<double, 3, 3>, double>;

  State pack(tensor<double, 3, 3> plastic_strain, double accumulated_plastic_strain) const
  {
    State internal_state{};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        internal_state[3*i + j] = plastic_strain[i][j];
      }
    }
    internal_state[9] = accumulated_plastic_strain;
    return internal_state;
  }

  UnpackedState unpack(State internal_state) const
  {
    auto plastic_strain = make_tensor<3, 3>([=](int i, int j) { return internal_state[3*i + j];});
    double accumulated_plastic_strain = internal_state[9];
    return tuple{plastic_strain, accumulated_plastic_strain};
  }

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, const T du_dX) const
  {
    using std::sqrt;
    constexpr auto I = DenseIdentity<dim>();
    const double   K = E / (3.0 * (1.0 - 2.0 * nu));
    const double   G = 0.5 * E / (1.0 + nu);

    // (i) elastic predictor
    auto [plastic_strain, accumulated_plastic_strain] = unpack(state);
    auto el_strain = sym(du_dX) - plastic_strain;
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto q         = sqrt(1.5) * norm(s);

    // (ii) admissibility
    const double eqps_old = accumulated_plastic_strain;
    auto         residual = [eqps_old, G, *this](auto delta_eqps, auto trial_mises) {
      return trial_mises - 3.0 * G * delta_eqps - this->hardening(eqps_old + delta_eqps);
    };
    if (residual(0.0, get_value(q)) > tol * hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double              lower_bound = 0.0;
      double              upper_bound = (get_value(q) - hardening(eqps_old)) / (3.0 * G);
      auto [delta_eqps, status]       = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q);

      auto Np = 1.5 * s / q;

      s = s - 2.0 * G * delta_eqps * Np;
      accumulated_plastic_strain += get_value(delta_eqps);
      plastic_strain += get_value(delta_eqps) * get_value(Np);
      state = pack(plastic_strain, accumulated_plastic_strain);
    }

    return s + p * I;
  }
};

/**
 * @brief Neo-Hookean material model
 *
 */
struct NeoHookean {
  using State = Empty;  ///< this material has no internal variables

  static constexpr int dim = 3;
  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * @tparam T Number-like type for the displacement gradient components
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The Cauchy stress
   */
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const tensor<double, dim, dim>& du_dX, double K, double G) const
  {
    using std::log;
    constexpr auto I         = DenseIdentity<dim>();
    auto           lambda    = K - (2.0 / 3.0) * G;
    auto           B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto           J         = det(I + du_dX);
    return (lambda * log(J) * I + G * B_minus_I) / J;
  }

  double density;  ///< mass density
};


/// @brief a 3D constitutive model for a J2 material with linear isotropic and kinematic hardening.
struct J2 {
  /// this material is written for 3D
  static constexpr int dim = 3;

  double density;  ///< mass density

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> beta;                        ///< back-stress tensor
    tensor<double, dim, dim> plastic_strain;              ///< plastic strain
    double                   accumulated_plastic_strain;  ///< incremental plastic strain
  };

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T>
  auto operator()(State& state, const T du_dX, double E, double nu, double sigma_y, double Hi, double Hk) const
  {
    using std::sqrt;
    constexpr auto I = DenseIdentity<3>();
    double   K = E / (3.0 * (1.0 - 2.0 * nu));
    double   G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    auto el_strain = sym(du_dX) - state.plastic_strain;
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto eta       = s - state.beta;
    auto q         = sqrt(3.0 / 2.0) * norm(eta);
    auto phi       = q - (sigma_y + Hi * state.accumulated_plastic_strain);

    // (ii) admissibility
    if (phi > 0.0) {
      // see (7.207) on pg. 261
      auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

      // from here on, only normalize(eta) is required
      // so we overwrite eta with its normalized version
      eta = normalize(eta);

      // (iii) return mapping
      s = s - sqrt(6.0) * G * plastic_strain_inc * eta;
      state.accumulated_plastic_strain += get_value(plastic_strain_inc);
      state.plastic_strain += sqrt(3.0 / 2.0) * get_value(plastic_strain_inc) * get_value(eta);
      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * get_value(plastic_strain_inc) * get_value(eta);
    }

    return s + p * I;
  }
};

/**
 * @brief Drive the material model thorugh a uniaxial tension experiment
 *
 * Drives material model through specified axial displacement gradient history.
 * The time elaspses from 0 up to t_max.
 * Currently only implemented for isotropic materials (or orthotropic materials with the
 * principal axes aligned with the coordinate directions).
 *
 * @param t_max upper limit of the time interval.
 * @param num_steps The number of discrete time points at which the response is sampled (uniformly spaced).
 *        This is inclusive of the point at time zero.
 * @param material The material model to use
 * @param initial_state The state variable collection for this material, set to the desired initial
 *        condition.
 * @param epsilon_xx A function describing the desired axial displacement gradient as a function of time.
 *        (NB axial displacement gradient is equivalent to engineering strain).
 * @param parameter_functions Pack of functions that return each parameter as a function of time. Leave
 *        empty if the material has no parameters.
 */
template <typename MaterialType, typename StateType, typename... parameter_types>
auto uniaxial_stress_test(double t_max, size_t num_steps, const MaterialType material, const StateType initial_state,
                          std::function<double(double)> epsilon_xx, const parameter_types... parameter_functions)
{
  double t = 0;

  auto state = initial_state;

  auto sigma_yy_and_zz = [&](auto x) {
    auto epsilon_yy = x[0];
    auto epsilon_zz = x[1];
    using T         = decltype(epsilon_yy);
    tensor<T, 3, 3> du_dx{};
    du_dx[0][0]     = epsilon_xx(t);
    du_dx[1][1]     = epsilon_yy;
    du_dx[2][2]     = epsilon_zz;
    auto state_copy = state;
    auto stress     = material(state_copy, du_dx, parameter_functions(t)...);
    return tensor{{stress[1][1], stress[2][2]}};
  };

  std::vector<tuple<double, tensor<double, 3, 3>, tensor<double, 3, 3>, StateType> > output_history;
  output_history.reserve(num_steps);

  tensor<double, 3, 3> dudx{};
  const double         dt = t_max / double(num_steps - 1);
  for (size_t i = 0; i < num_steps; i++) {
    auto initial_guess     = tensor<double, 2>{dudx[1][1], dudx[2][2]};
    auto epsilon_yy_and_zz = find_root(sigma_yy_and_zz, initial_guess);
    dudx[0][0]             = epsilon_xx(t);
    dudx[1][1]             = epsilon_yy_and_zz[0];
    dudx[2][2]             = epsilon_yy_and_zz[1];

    auto stress = material(state, dudx, parameter_functions(t)...);
    output_history.push_back(tuple{t, dudx, stress, state});

    t += dt;
  }

  return output_history;
}

} // namespace serac
