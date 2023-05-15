#include "tensor.hpp"
#include "tuple_tensor_dual_functions.hpp"

#pragma once

namespace serac {

struct Empty {
};

/**
 * @brief Neo-Hookean material model
 *
 */
struct NeoHookean {
  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The Cauchy stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& /* state */, const tensor<T, dim, dim>& du_dX, double K, double G) const
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

} // namespace serac