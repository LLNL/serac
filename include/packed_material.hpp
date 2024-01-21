#include "materials.hpp"
#include "tensor.hpp"
#include "tuple_tensor_dual_functions.hpp"

namespace serac {

/// @brief J2 material with nonlinear isotropic hardening.
template <typename HardeningType>
struct J2Packed {
  static constexpr int    dim = 3;      ///< spatial dimension
  static constexpr double tol = 1e-10;  ///< relative tolerance on residual mag to judge convergence of return map

  double        E;          ///< Young's modulus
  double        nu;         ///< Poisson's ratio
  HardeningType hardening;  ///< Flow stress hardening model
  double        density;    ///< mass density

  /// @brief variables required to characterize the hysteresis response
  template <typename T>
  using State = tensor<T, 10>;

  template <typename T>
  using UnpackedState = tuple<tensor<T, 3, 3>, T>;

  template <typename T>
  State<T> pack(tensor<T, 3, 3> plastic_strain, T accumulated_plastic_strain) const
  {
    State<T> internal_state{};
    for (int i = 0, ij = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++, ij++) {
        internal_state[ij] = plastic_strain[i][j];
      }
    }
    internal_state[9] = accumulated_plastic_strain;
    return internal_state;
  }

  template <typename T>
  UnpackedState<T> unpack(State<T> internal_state) const
  {
    tensor<T, 3, 3> plastic_strain = make_tensor<3, 3>([=](int i, int j) { return internal_state[3*i + j];});
    T accumulated_plastic_strain = internal_state[9];
    return tuple{plastic_strain, accumulated_plastic_strain};
  }

  /** @brief calculate the Cauchy stress, given the displacement gradient and previous material state */
  template <typename T1, typename T2>
  auto operator()(State<T1>& state, tensor<T2, 3, 3> du_dX) const
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
    const auto eqps_old = accumulated_plastic_strain;
    auto         residual = [G, *this](auto delta_eqps, auto trial_mises, auto eq_old) {
      return trial_mises - 3.0 * G * delta_eqps - this->hardening(eq_old + delta_eqps);
    };
    if (residual(0.0, get_value(q), get_value(eqps_old)) > tol * hardening.sigma_y) {
      // (iii) return mapping

      // Note the tolerance for convergence is the same as the tolerance for entering the return map.
      // This ensures that if the constitutive update is called again with the updated internal
      // variables, the return map won't be repeated.
      ScalarSolverOptions opts{.xtol = 0, .rtol = tol * hardening.sigma_y, .max_iter = 25};
      double              lower_bound = 0.0;
      double              upper_bound = get_value( (q - hardening(eqps_old)) / (3.0 * G) );
      auto [delta_eqps, status]       = solve_scalar_equation(residual, 0.0, lower_bound, upper_bound, opts, q, eqps_old);

      auto Np = 1.5 * s / q;

      s = s - 2.0 * G * delta_eqps * Np;
      constexpr bool internal_state_is_dual = is_tensor_of_dual_number<T1>::value;
      if constexpr (internal_state_is_dual) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain += delta_eqps * Np;
      } else {
        accumulated_plastic_strain += get_value(delta_eqps);
        plastic_strain += get_value(delta_eqps) * get_value(Np);
      }
      state = pack(plastic_strain, accumulated_plastic_strain);
    }

    return s + p * I;
  }
};

} // namespace serac