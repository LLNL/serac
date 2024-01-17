// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file shape_aware_functional.hpp
 *
 * @brief Wrapper of serac::Functional for evaluating integrals and derivatives of quantities with shape displacement
 * fields
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

namespace serac {

/// @cond
constexpr int SOURCE     = 0;
constexpr int FLUX       = 1;
constexpr int VALUE      = 0;
constexpr int DERIVATIVE = 1;
/// @endcond

namespace detail {

/**
 * @brief Compute the Jacobian (dx'/dx, x' = x + p) for shape-displaced integrals
 *
 * @tparam dim Dimension of the integral
 * @tparam shape_type The variable type for the shape displacement
 * @param shape The shape displacement
 * @return The computed Jacobian of the shape-displacement transformation (dx'/dx)
 *
 * @note This is currently only implemented for H1 shape displacement fields
 */
template <int dim, typename shape_type>
SERAC_HOST_DEVICE auto compute_jacobian(Dimension<dim>, const shape_type& shape)
{
  auto dp_dX = get<DERIVATIVE>(shape);

  // x' = x + p
  // J  = dx'/dx
  //    = I + dp_dx
  return Identity<dim>() + dp_dX;
}

/**
 * @brief Compute the boundary area correction term for boundary integrals with a shape displacement field
 *
 * @tparam position_type The position input argument type
 * @tparam shape_type The shape displacement input argument type
 * @param X The input position (value and gradient)
 * @param shape The input shape displacement (value and gradient)
 * @return The area correction factor transforming the boundary integral from the reference domain area into the
 * shape-displaced area
 */
template <typename position_type, typename shape_type>
SERAC_HOST_DEVICE auto compute_boundary_area_correction(const position_type& X, const shape_type& shape)
{
  auto x_prime = X + shape;

  auto n = cross(get<DERIVATIVE>(x_prime));

  // serac::Functional's boundary integrals multiply the q-function output by
  // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
  // then that weight needs to be corrected. The new weight should be
  // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
  // to get
  //   q * area_correction * w_old
  // = q * (w_new / w_old) * w_old
  // = q * w_new

  auto area_correction = norm(n) / norm(cross(get<DERIVATIVE>(X)));

  return area_correction;
}

/**
 * @brief Modify the value and flux of a q-function integrand according to the correct pullback mappings
 *
 * @tparam dim The dimension of the integral
 * @tparam test_space The finite element space for the test function
 * @tparam shape_type The type of the shape displacement argument
 * @tparam q_func_type The type of the unmodified q function return value (value and flux)
 *
 * @param d The dimension of the integral
 * @param shape The input shape displacement (value and gradient)
 * @param v The unmodified q function return value (value and flux)
 *
 * @return The corrected q function return accounting for shape displacements
 *
 * @note This is currently only implemented for H1 and L2 test function spaces
 */
template <int dim, typename test_space, typename shape_type, typename q_func_type,
          typename = std::enable_if_t<test_space{}.family == Family::H1 || test_space{}.family == Family::L2>>
SERAC_HOST_DEVICE auto modify_shape_aware_qf_return(Dimension<dim> d, test_space /*test*/, const shape_type& shape,
                                                    const q_func_type& v)
{
  auto J = compute_jacobian(d, shape);

  auto dv = det(J);

  auto modified_flux   = dot(get<FLUX>(v), transpose(inv(J))) * dv;
  auto modified_source = get<SOURCE>(v) * dv;

  return serac::tuple{modified_source, modified_flux};
}

/**
 * @brief Modify the value and flux of a q-function integrand according to the pullback map for QOI integrals
 *
 * @tparam dim The dimension of the integral
 * @tparam shape_type The type of the shape displacement argument
 * @tparam q_func_type The type of the unmodified q function return value (value only)
 *
 * @param d The dimension of the integral
 * @param shape The input shape displacement (value and gradient)
 * @param v The unmodified q function return value (value only)
 *
 * @return The corrected q function return accounting for shape displacements
 *
 * @note This is a specialization for QOI test function spaces
 */
template <int dim, typename shape_type, typename q_func_type>
SERAC_HOST_DEVICE auto modify_shape_aware_qf_return(Dimension<dim> d, double, const shape_type& shape,
                                                    const q_func_type& v)
{
  auto J = compute_jacobian(d, shape);

  auto dv = det(J);

  return dv * v;
}

/**
 * @brief Modify the gradient of a trial function according to the shape displacement Jacobian
 *
 * @tparam jacobian_type The type of the Jacobian argument
 * @tparam space_type The finite element space for the trial function
 * @tparam trial_type The type of the unmodified trial function (value and gradient)
 *
 * @param J The Jacobian (dx'/dx) for the shape displaced coordinate map
 * @param u The unmodified trial function argument (value and gradient)
 *
 * @return The modified trial q function argument value adjusted for the current shape displacement field
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces
 */
template <typename jacobian_type, typename trial_type, typename space_type,
          typename = std::enable_if_t<space_type{}.family == Family::H1 || space_type{}.family == Family::L2>>
SERAC_HOST_DEVICE auto modify_trial_argument(const jacobian_type& J, space_type /* space */, const trial_type& u)
{
  auto du_dx = get<DERIVATIVE>(u);

  // Our updated spatial coordinate is x' = x + p. We want to calculate
  // du/dx' = du/dx * dx/dx'
  //        = du/dx * (dx'/dx)^-1
  //        = du_dx * (J)^-1
  auto trial_derivative = dot(du_dx, inv(J));

  return serac::tuple{get<VALUE>(u), trial_derivative};
}

/**
 * @brief A helper function to modify all of the trial function input derivatives according to the given shape
 * displacement for integrands without state variables
 *
 * @tparam dim The dimension of the integral
 * @tparam lambda The q-function type
 * @tparam coord_type The input position type
 * @tparam shape_type The type of the shape displacement argument
 * @tparam space_types The type of the input finite element space tuple for the trial functions
 * @tparam trial_types The type of the input finite element argument tuple (values and derivatives)
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param d The dimension of the integral
 * @param qf The q-function integrand with expects shape-adjusted arguments
 * @param t The time at which to evaluate the integrand
 * @param x The spatial coordinate at which to evaluate the integrand
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces.
 */
template <int dim, typename lambda, typename coord_type, typename shape_type, typename space_types,
          typename trial_types, int... i>
SERAC_HOST_DEVICE auto apply_shape_aware_qf_helper(Dimension<dim> d, lambda&& qf, double t, const coord_type& x,
                                                   const shape_type& shape, const space_types& space_tuple,
                                                   const trial_types& arg_tuple, std::integer_sequence<int, i...>)
{
  [[maybe_unused]] auto J = compute_jacobian(d, shape);

  return qf(t, x + get<VALUE>(shape),
            modify_trial_argument(J, serac::get<i>(space_tuple), serac::get<i>(arg_tuple))...);
}

/**
 * @brief A helper function to modify all of the trial function input derivatives according to the given shape
 * displacement for integrands without state variables
 *
 * @tparam dim The dimension of the integral
 * @tparam lambda The q-function type
 * @tparam coord_type The input position type
 * @tparam state_type The quadrature data container type
 * @tparam shape_type The type of the shape displacement argument
 * @tparam space_types The type of the input finite element space tuple for the trial functions
 * @tparam trial_types The type of the input finite element argument tuple (values and derivatives)
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param d The dimension of the integral
 * @param qf The q-function integrand with expects shape-adjusted arguments
 * @param t The time at which to evaluate the integrand
 * @param x The spatial coordinate at which to evaluate the integrand
 * @param state The quadrature data at which to evaluate the integrand
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces.
 */
template <int dim, typename lambda, typename coord_type, typename state_type, typename shape_type, typename space_types,
          typename trial_types, int... i>
SERAC_HOST_DEVICE auto apply_shape_aware_qf_helper_with_state(Dimension<dim> d, lambda&& qf, double t,
                                                              const coord_type& x, state_type& state,
                                                              const shape_type& shape, const space_types& space_tuple,
                                                              const trial_types& arg_tuple,
                                                              std::integer_sequence<int, i...>)
{
  [[maybe_unused]] auto J = compute_jacobian(d, shape);

  return qf(t, x + get<VALUE>(shape), state,
            modify_trial_argument(J, serac::get<i>(space_tuple), serac::get<i>(arg_tuple))...);
}

}  // namespace detail

/// @cond
template <typename T1, typename T2, ExecutionSpace exec = serac::default_execution_space>
class ShapeAwareFunctional;
/// @endcond

/**
 * @brief This is a small wrapper around serac::Functional for shape-displaced domains of integration
 *
 * If a finite element kernel is defined on a domain \f$x = X + p\f$ where \f$X\f$ is the reference
 * configuration and \f$p\f$ is a shape displacement field. This shape displacement is typically a input
 * parameter used in shape optimization problems. This wrapper correctly modifies domain and boundary
 * integrals for L2 and H1 trial functions and H1, L2, and QOI (double) test functions for vector-valued
 * H1 shape displacement fields.
 *
 * See serac::Functional for more details about the underlying finite element abstraction and serac::HeatTransfer
 * for an example of how to use this wrapper.
 *
 * @tparam test The space of test function to use
 * @tparam shape The space of the shape displacement function to use
 * @tparam trials The space of trial functions to use
 * @tparam exec whether to carry out calculations on CPU or GPU
 */
template <typename test, typename shape, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<shape, test(trials...), exec> {
  /// @brief The compile-time trial function finite element spaces
  static constexpr tuple<trials...> trial_spaces{};

  /// @brief The compile-time test function finite element space
  static constexpr test test_space{};

  /// @brief The compile-time shape displacement finite element space
  static constexpr shape shape_space{};

  /// @brief The number of input trial functions
  static constexpr uint32_t num_trial_spaces = sizeof...(trials);

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   *
   * @tparam test_type The finite element space of the test functions (used to disable this method if called with test =
   * double)
   *
   * @param[in] shape_fes The shape displacement finite element space
   * @param[in] test_fes The (non-qoi) test finite element space
   * @param[in] trial_fes The trial finite element spaces
   */
  template <typename test_type = test, typename = std::enable_if_t<!std::is_same_v<double, test_type>>>
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace* shape_fes, const mfem::ParFiniteElementSpace* test_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    static_assert(test_space.family != Family::HDIV, "Shape-aware functional not implemented for HDiv test functions");
    static_assert(test_space.family != Family::HCURL,
                  "Shape-aware functional not implemented for HCurl test functions");
    static_assert(shape_space.family == Family::H1, "Only H1 spaces allowed for shape displacements");

    for_constexpr<num_trial_spaces>([](auto i) {
      static_assert(get<i>(trial_spaces).family != Family::HDIV,
                    "Shape-aware functional not implemented for HDiv trial functions");
      static_assert(get<i>(trial_spaces).family != Family::HCURL,
                    "Shape-aware functional not implemented for HCurl trial functions");
      static_assert(get<i>(trial_spaces).family != Family::QOI,
                    "Shape-aware functional not implemented for QOI trial functions");
    });

    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<test(shape, trials...), exec>>(test_fes, prepended_spaces);
  }

  /**
   * @brief Constructs a QOI functional using @p mfem::ParFiniteElementSpace objects corresponding to the trial spaces
   *
   * @tparam test_type The finite element space of the test functions (used to disable this method if called with test
   * != double)
   *
   * @param[in] shape_fes The shape displacement finite element space
   * @param[in] trial_fes The trial finite element spaces
   */
  template <typename test_type = test, typename = std::enable_if_t<std::is_same_v<double, test_type>>>
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace*                               shape_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    static_assert(shape_space.family == Family::H1, "Only H1 spaces allowed for shape displacements");

    for_constexpr<num_trial_spaces>([](auto i) {
      static_assert(get<i>(trial_spaces).family != Family::HDIV,
                    "Shape-aware functional not implemented for HDiv trial functions");
      static_assert(get<i>(trial_spaces).family != Family::HCURL,
                    "Shape-aware functional not implemented for HCurl trial functions");
      static_assert(get<i>(trial_spaces).family != Family::QOI,
                    "Shape-aware functional not implemented for QOI trial functions");
    });

    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<double(shape, trials...), exec>>(prepended_spaces);
  }

  /**
   * @brief Adds a domain integral term to the weak formulation of the PDE
   *
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam args The type of the trial function input arguments
   * @tparam lambda The type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam domain_type The type of the integration domain (either serac::Domain or mfem::Mesh)
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   *
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   * @param[inout] qdata The data for each quadrature point
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda, typename domain_type, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, domain_type& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    if constexpr (std::is_same_v<qpt_data_type, Nothing>) {
      functional_->AddDomainIntegral(
          Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
          [integrand](double time, auto x, auto shape_val, auto... qfunc_args) {
            auto qfunc_tuple = make_tuple(qfunc_args...);

            auto unmodified_qf_return = detail::apply_shape_aware_qf_helper(
                Dimension<dim>{}, integrand, time, x, shape_val, trial_spaces, qfunc_tuple,
                std::make_integer_sequence<int, sizeof...(qfunc_args)>{});
            return detail::modify_shape_aware_qf_return(Dimension<dim>{}, test_space, shape_val, unmodified_qf_return);
          },
          domain, qdata);
    } else {
      functional_->AddDomainIntegral(
          Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
          [integrand](double time, auto x, auto& state, auto shape_val, auto... qfunc_args) {
            auto qfunc_tuple = make_tuple(qfunc_args...);

            auto unmodified_qf_return = detail::apply_shape_aware_qf_helper_with_state(
                Dimension<dim>{}, integrand, time, x, state, shape_val, trial_spaces, qfunc_tuple,
                std::make_integer_sequence<int, sizeof...(qfunc_args)>{});
            return detail::modify_shape_aware_qf_return(Dimension<dim>{}, test_space, shape_val, unmodified_qf_return);
          },
          domain, qdata);
    }
  }

  /**
   * @brief Adds a boundary integral term to the weak formulation of the PDE
   *
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam args The type of the trial function input arguments
   * @tparam lambda The type of the integrand functor: must implement operator() with an appropriate function signature
   * @tparam domain_type The type of the integration domain (either serac::Domain or mfem::Mesh)
   *
   * @param[in] integrand The user-provided quadrature function, see @p Integral
   * @param[in] domain The domain on which to evaluate the integral
   *
   * @note The @p Dimension parameters are used to assist in the deduction of the @a geometry_dim
   * and @a spatial_dim template parameter
   */
  template <int dim, int... args, typename lambda, typename domain_type>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, domain_type& domain)
  {
    functional_->AddBoundaryIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape_val, auto... qfunc_args) {
          auto unmodified_qf_return = integrand(time, x + shape_val, qfunc_args...);

          return unmodified_qf_return * detail::compute_boundary_area_correction(x, shape_val);
        },
        domain);
  }

  /**
   * @brief This function lets the user evaluate the serac::ShapeAwareFunctional with the given trial space values
   *
   * @note The first argument after time in the argument list is always the shape displacement field.
   *
   * @tparam T The types of the arguments passed in
   * @tparam wrt The index of the argument to take the gradient with respect to. Index 0 is always the shape
   * displacement
   * @param t The time
   * @param args The trial space dofs used to carry out the calculation. The first argument is always the shape
   * displacement.
   *
   * @return Either the evaluated integral value or a tuple of the integral value and the requested derivative.
   */
  template <uint32_t wrt, typename... T>
  auto operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    return (*functional_)(DifferentiateWRT<wrt>{}, t, args...);
  }

  /**
   * @brief This function lets the user evaluate the serac::ShapeAwareFunctional with the given trial space values
   *
   * @note The first argument after time in the argument list is always the shape displacement field.
   *
   * @tparam T The types of the arguments passed in
   * @param t The time
   * @param args The trial space dofs used to carry out the calculation. The first argument is always the shape
   * displacement. To compute a derivative, at most one argument can be of type `differentiate_wrt_this(mfem::Vector)`.
   *
   * @return Either the evaluated integral value or a tuple of the integral value and the requested derivative.
   */
  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    return (*functional_)(t, args...);
  }

  /**
   * @brief A flag to update the quadrature data for this operator following the computation
   *
   * Typically this is set to false during nonlinear solution iterations and is set to true for the
   * final pass once equilibrium is found.
   *
   * @param update_flag A flag to update the related quadrature data
   */
  void updateQdata(bool update_flag) { functional_->updateQdata(update_flag); }

private:
  /// @brief The underlying pure Functional object
  std::unique_ptr<Functional<test(shape, trials...), exec>> functional_;
};

}  // namespace serac
