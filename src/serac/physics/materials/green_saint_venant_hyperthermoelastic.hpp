#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
//#include "serac/numerics/functional/tuple_arithmetic.hpp"

namespace serac {

/**
 * @brief Compute Green's strain from the displacement gradient
 */
template <typename T>
auto greenStrain(const tensor<T, 3, 3>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

/// @brief Green-Saint Venant isotropic thermoelastic model
struct ThermoelasticMaterial {
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C;          ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature
  double k;          ///< thermal conductivity

  struct State { /* this material has no internal variables */
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  template <typename T1, typename T2, typename T3>
  auto calculateConstitutiveOutputs(const tensor<T1, 3, 3>& grad_u, T2 theta, const tensor<T3, 3>& grad_theta,
                                    State& /*state*/, const tensor<double, 3, 3>& grad_u_old, double /*theta_old*/,
                                    double dt) const
  {
    const double K    = E / (3.0 * (1.0 - 2.0 * nu));
    const double G    = 0.5 * E / (1.0 + nu);
    static constexpr auto I = Identity<3>();
    auto         F    = grad_u + I;
    const auto   Eg   = greenStrain(grad_u);
    const auto   trEg = tr(Eg);

    // stress
    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto P = F * S;

    // internal heat source
    // use backward difference to estimate rate of green strain
    const auto Eg_old = greenStrain(grad_u_old);
    const auto egdot  = (trEg - tr(Eg_old)) / dt;
    const auto s0     = -3 * K * alpha * theta * egdot;

    // heat flux
    const auto q0 = -k * grad_theta;

    return serac::tuple{P, C, s0, q0};
  }

  /**
   * @brief Return constitutive output for thermal operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] cv0 volumetric heat capacity in ref config
   * @param[out] s0 internal heat supply in ref config
   * @param[out] q0 Piola heat flux
   */
  auto calculateThermalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                           const tensor<double, 3>& grad_theta, State& state,
                                           const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt) const
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return serac::tuple{cv0, s0, q0};
  }

  /**
   * @brief Return constitutive output for mechanics operator
   *
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   * @param[in] grad_theta temperature gradient
   * @param[in] grad_u_old displacement gradient at previous time step
   * @param[in] theta_old temperature at previous time step
   * @param[in] dt time increment
   * @param[out] P Piola stress
   */
  auto calculateMechanicalConstitutiveOutputs(const tensor<double, 3, 3>& grad_u, double theta,
                                              const tensor<double, 3>& grad_theta, State& state,
                                              const tensor<double, 3, 3>& grad_u_old, double theta_old, double dt) const
  {
    auto [P, cv0, s0, q0] = calculateConstitutiveOutputs(grad_u, theta, grad_theta, state, grad_u_old, theta_old, dt);
    return P;
  }

  /**
   * @brief evaluate free energy density
   * @param[in] grad_u displacement gradient
   * @param[in] theta temperature
   */
  template <typename T1, typename T2>
  auto calculateFreeEnergy(const tensor<T1, 3, 3>& grad_u, T2 theta) const
  {
    const double K      = E / (3.0 * (1.0 - 2.0 * nu));
    const double G      = 0.5 * E / (1.0 + nu);
    auto         strain = greenStrain(grad_u);
    auto         trE    = tr(strain);
    auto         psi_1  = G * squared_norm(dev(strain)) + 0.5 * K * trE * trE;
    using std::log;
    auto logT  = log(theta / theta_ref);
    auto psi_2 = C * (theta - theta_ref + theta * logT);
    auto psi_3 = -3.0 * K * alpha * (theta - theta_ref) * trE;
    return psi_1 + psi_2 + psi_3;
  }
};

} // namespace serac
