// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file functional.hpp
 *
 * @brief Implementation of the quadrature-function-based functional enabling rapid development of FEM formulations
 */

#pragma once

#include "serac/numerics/functional/functional.hpp"

namespace serac {

constexpr int SOURCE     = 0;
constexpr int FLUX       = 1;
constexpr int VALUE      = 0;
constexpr int DERIVATIVE = 1;

namespace detail {

/**
 * @brief Compute the Jacobian (dx'/dx, x' = x + p) for shape-displaced integrals
 *
 * @tparam dim Dimension of the integral
 * @tparam shape_type The type for the shape displacement
 * @param shape The shape displacement
 * @return The computed Jacobian of the shape-displacement transformation
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
 * @tparam T The type of the unmodified q function return value (value and flux)
 *
 * @param d The dimension of the integral
 * @param shape The input shape displacement (value and gradient)
 * @param v The unmodified q function return value (value and flux)
 *
 * @return The corrected q function return accounting for shape displacements
 *
 * @note This is currently only implemented for H1 and L2 test function spaces
 */
template <int dim, typename test_space, typename shape_type, typename T,
          typename = std::enable_if_t<test_space{}.family == Family::H1 || test_space{}.family == Family::L2>>
SERAC_HOST_DEVICE auto modify_shape_aware_qf_return(Dimension<dim> d, test_space /*test*/, const shape_type& shape,
                                                    const T& v)
{
  auto J = compute_jacobian(d, shape);

  auto dv = det(J);

  auto modified_flux   = dot(get<FLUX>(v), transpose(inv(J))) * dv;
  auto modified_source = get<SOURCE>(v) * dv;

  return serac::tuple{modified_source, modified_flux};
}

template <int dim, typename shape_type, typename T>
SERAC_HOST_DEVICE auto modify_shape_aware_qf_return(Dimension<dim> d, double, const shape_type& shape, const T& v)
{
  auto J = compute_jacobian(d, shape);

  auto dv = det(J);

  return dv * v;
}

/**
 * @brief Modify the gradient of a trial function according to the correct shape displacement map
 *
 * @tparam shape_type The type of the shape displacement argument
 * @tparam space_type The finite element space for the trial function
 * @tparam T The type of the unmodified trial function (value and gradient)
 *
 * @param d The dimension of the integral
 * @param shape The input shape displacement (value and gradient)
 * @param v The unmodified q function return value (value and flux)
 *
 * @return The modified q function argument value adjusted for the current shape displacement field
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces
 */
template <typename jacobian_type, typename T, typename space_type,
          typename = std::enable_if_t<space_type{}.family == Family::H1 || space_type{}.family == Family::L2>>
SERAC_HOST_DEVICE auto modify_trial_argument(const jacobian_type& J, space_type /* space */, const T& u)
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
 * @tparam S The type of the input finite element space tuple for the trial functions
 * @tparam T The type of the input finite element argument tuple (values and derivatives)
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param d The dimension of the integral
 * @param qf The q-function integrand
 * @param t The time at which to evaluate the integrand
 * @param x The spatial coordinate at which to evaluate the integrand
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces
 */
template <int dim, typename lambda, typename coord_type, typename shape_type, typename S, typename T, int... i>
SERAC_HOST_DEVICE auto apply_shape_aware_qf_helper(Dimension<dim> d, lambda&& qf, double t, const coord_type& x,
                                                   const shape_type& shape, const S& space_tuple, const T& arg_tuple,
                                                   std::integer_sequence<int, i...>)
{
  [[maybe_unused]] auto J = compute_jacobian(d, shape);

  return qf(t, x + get<VALUE>(shape),
            modify_trial_argument(J, serac::get<i>(space_tuple), serac::get<i>(arg_tuple))...);
}

/**
 * @brief A helper function to modify all of the trial function input derivatives according to the given shape
 * displacement for integrands with state variables
 *
 * @tparam dim The dimension of the integral
 * @tparam lambda The q-function type
 * @tparam coord_type The input position type
 * @tparam state_type The type of the quadrature state container
 * @tparam shape_type The type of the shape displacement argument
 * @tparam S The type of the input finite element space tuple for the trial functions
 * @tparam T The type of the input finite element argument tuple (values and derivatives)
 * @tparam i Indices for accessing the individual arguments for the underlying q-function
 *
 * @param d The dimension of the integral
 * @param qf The q-function integrand
 * @param t The time at which to evaluate the integrand
 * @param x The spatial coordinate at which to evaluate the integrand
 * @param state The quadrature data state at the requested point
 * @param shape The space displacement at which to evaluate the integrand
 * @param space_tuple The tuple of finite element spaces used by the input trial functions
 * @param arg_tuple The tuple of input arguments for the trial functions (value and gradient)
 *
 * @return The q-function value using the shape-modified input arguments. Note that the returned value and flux have not
 * been modified to reflect the shape displacement.
 *
 * @note This is currently only implemented for H1 and L2 trial function spaces
 */
template <int dim, typename lambda, typename coord_type, typename state_type, typename shape_type, typename S,
          typename T, int... i>
SERAC_HOST_DEVICE auto apply_shape_aware_qf_helper_with_state(Dimension<dim> d, lambda&& qf, double t,
                                                              const coord_type& x, state_type& state,
                                                              const shape_type& shape, const S& space_tuple,
                                                              const T& arg_tuple, std::integer_sequence<int, i...>)
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

template <typename test, typename shape, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<shape, test(trials...), exec> {
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr test             test_space{};
  static constexpr shape            shape_space{};
  static constexpr uint32_t         num_trial_spaces = sizeof...(trials);

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  // template <typename test_space = test, std::enable_if_t<!std::is_same_v<double, test_space>>>
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
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] trial_fes The trial space
   */
  // template <typename test_space = test, std::enable_if_t<std::is_same_v<double, test_space>>>
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

  template <int dim, int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain,
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

  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    functional_->AddBoundaryIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape_val, auto... qfunc_args) {
          auto unmodified_qf_return = integrand(time, x + shape_val, qfunc_args...);

          return unmodified_qf_return * detail::compute_boundary_area_correction(x, shape_val);
        },
        domain);
  }

  template <uint32_t wrt, typename... T>
  auto operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    return (*functional_)(DifferentiateWRT<wrt>{}, t, args...);
  }

  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    return (*functional_)(t, args...);
  }

  void updateQdata(bool update_flag) { functional_->updateQdata(update_flag); }

private:
  std::unique_ptr<Functional<test(shape, trials...), exec>> functional_;
};

}  // namespace serac
