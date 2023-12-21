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
#include "serac/numerics/functional/detail/metaprogramming.hpp"

#include <type_traits>

namespace serac {

namespace detail {

template <int dim, typename test_space, typename shape_type, typename T>
SERAC_HOST_DEVICE auto modify_qf_return(Dimension<dim>, test_space test, shape_type shape, T v)
{
  [[maybe_unused]] constexpr int SOURCE     = 0;
  [[maybe_unused]] constexpr int FLUX       = 1;
  [[maybe_unused]] constexpr int DERIVATIVE = 1;

  auto dp_dX = get<DERIVATIVE>(shape);

  // x' = x + p
  // J  = dx'/dx
  //    = I + dp_dx
  auto J = Identity<dim>() + dp_dX;

  auto dv = det(J);

  serac::tuple modified_test_return{get<SOURCE>(v) + 0.0 * get<SOURCE>(v) * dv,
                                    get<FLUX>(v) + 0.0 * inv(J) * get<FLUX>(v)};

  if constexpr (test.family == Family::H1) {
    get<FLUX>(modified_test_return) = dot(inv(J), get<FLUX>(v));
  }

  if constexpr (test.family == Family::H1 || test.family == Family::L2) {
    get<SOURCE>(modified_test_return) = get<SOURCE>(v) * dv;
  }

  return modified_test_return;
}

template <int dim, typename shape_type, typename S, typename T>
SERAC_HOST_DEVICE auto modify_trial_argument(Dimension<dim>, shape_type shape, S space, T u)
{
  [[maybe_unused]] constexpr int VALUE      = 0;
  [[maybe_unused]] constexpr int DERIVATIVE = 1;

  serac::tuple modified_trial{get<VALUE>(u),
                              get<DERIVATIVE>(u) + 0.0 * dot(get<DERIVATIVE>(u), get<DERIVATIVE>(shape))};

  // For H1 and L2 fields, we must correct the gradient value to reflect
  // the shape-perturbed coordinates
  if constexpr (space.family == Family::H1 || space.family == Family::L2) {
    auto du_dx = get<DERIVATIVE>(u);
    auto dp_dX = get<DERIVATIVE>(shape);

    // x' = x + p
    // J  = dx'/dx
    //    = I + dp_dx
    auto J = Identity<dim>() + dp_dX;

    // Our updated spatial coordinate is x' = x + p. We want to calculate
    // du/dx' = du/dx * dx/dx'
    //        = du/dx * (dx'/dx)^-1
    //        = du_dx * (J)^-1
    get<DERIVATIVE>(modified_trial) = dot(du_dx, inv(J));
  }

  return modified_trial;
}

template <int dim, typename lambda, typename coord_type, typename shape_type, typename S, typename T, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(Dimension<dim> d, lambda&& qf, double t, coord_type coords, shape_type shape,
                                       const S& space_tuple, const T& arg_tuple, std::integer_sequence<int, i...>)
{
  constexpr int VALUE = 0;

  auto x_prime = coords + get<VALUE>(shape);

  return qf(t, x_prime, modify_trial_argument(d, shape, serac::get<i>(space_tuple), serac::get<i>(arg_tuple))...);
}

template <int dim, typename lambda, typename coord_type, typename shape_type, typename... S, typename... T>
SERAC_HOST_DEVICE auto apply_qf(Dimension<dim> d, lambda&& qf, double t, coord_type coords, const shape_type shape,
                                const serac::tuple<S...>& space_tuple, const serac::tuple<T...>& arg_tuple)
{
  static_assert(sizeof...(S) == sizeof...(T),
                "Size of trial space types and q function arguments not equal in ShapeAwareFunctional.");

  return serac::detail::apply_qf_helper(d, qf, t, coords, shape, space_tuple, arg_tuple,
                                        std::make_integer_sequence<int, sizeof...(T)>{});
}

}  // namespace detail

/// @cond
template <typename T1, typename T2, ExecutionSpace exec = serac::default_execution_space>
class ShapeAwareFunctional;
/// @endcond

template <typename test, typename shape_space, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<shape_space, test(trials...), exec> {
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t         num_trial_spaces = sizeof...(trials);

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace*                                   test_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> trial_fes)
      : functional_(test_fes, trial_fes)
  {
    test test_space{};

    SLIC_ERROR_ROOT_IF(test_space.family == Family::HDIV,
                       "Shape-aware functional not implemented for HDiv test functions");
    SLIC_ERROR_ROOT_IF(test_space.family == Family::HCURL,
                       "Shape-aware functional not implemented for HCurl test functions");

    SLIC_ERROR_ROOT_IF(get<0>(trial_spaces).family != Family::H1, "Only H1 spaces allowed for shape displacements");

    for_constexpr<num_trial_spaces>([](auto i) {
      auto space = get<i>(trial_spaces);

      SLIC_ERROR_ROOT_IF(space.family == Family::HDIV,
                         "Shape-aware functional not implemented for HDiv trial functions");
      SLIC_ERROR_ROOT_IF(space.family == Family::HCURL,
                         "Shape-aware functional not implemented for HCurl trial functions");
      SLIC_ERROR_ROOT_IF(space.family == Family::QOI, "Shape-aware functional not implemented for QOI trial functions");
    });
  }

  template <int dim, int... args, typename lambda>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    functional_.AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape, auto... qfunc_args) {
          auto qfunc_tuple = make_tuple(qfunc_args...);

          auto unmodified_qf_return =
              detail::apply_qf(Dimension<dim>{}, integrand, time, x, shape, trial_spaces, qfunc_tuple);

          return detail::modify_qf_return(Dimension<dim>{}, test{}, shape, unmodified_qf_return);
        },
        domain);
  }

  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    functional_.AddBoundaryIntegral(
        Dimension<dim>{}, DependsOn<0, (1 + args)...>{},
        [integrand, spaces = trial_spaces](double time, auto x, auto shape, auto... qfunc_args) {
          auto qfunc_tuple = make_tuple(qfunc_args...);

          [[maybe_unused]] constexpr int VALUE      = 0;
          [[maybe_unused]] constexpr int DERIVATIVE = 1;

          auto dp_dX = get<DERIVATIVE>(shape);
          auto p     = get<VALUE>(shape);

          return detail::apply_qf(integrand, time, x, shape, trial_spaces, qfunc_tuple);

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
          // to get
          //   q * area_correction * w_old
          // = q * (w_new / w_old) * w_old
          // = q * w_new
          // auto area_correction = cross(get<DERIVATIVE>(x_prime)) / norm(cross(get<DERIVATIVE>(x)));

          // return qf_return * area_correction;
        },
        domain);
  }

  template <uint32_t wrt, typename... T>
  auto operator()(DifferentiateWRT<wrt>, double t, const T&... args)
  {
    return functional_(DifferentiateWRT<wrt>{}, t, args...);
  }

  template <typename... T>
  auto operator()(double t, const T&... args)
  {
    return functional_(t, args...);
  }

private:
  Functional<test(shape_space, trials...), exec> functional_;
};

}  // namespace serac
