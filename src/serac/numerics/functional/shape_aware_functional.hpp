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

#include <type_traits>

namespace serac {

constexpr int SOURCE     = 0;
constexpr int FLUX       = 1;
constexpr int VALUE      = 0;
constexpr int DERIVATIVE = 1;

namespace detail {

template <int dim, typename test_space, typename shape_type, typename T,
          typename = std::enable_if_t<test_space{}.family == Family::H1 || test_space{}.family == Family::L2 ||
                                      std::is_same_v<double, test_space>>>
SERAC_HOST_DEVICE auto modify_shape_aware_qf_return(Dimension<dim>, test_space /*test*/, shape_type shape, T v)
{
  auto dp_dX = get<DERIVATIVE>(shape);

  // x' = x + p
  // J  = dx'/dx
  //    = I + dp_dx
  auto J = Identity<dim>() + dp_dX;

  auto dv = det(J);

  auto modified_flux   = dot(inv(J), get<FLUX>(v)) * dv;
  auto modified_source = get<SOURCE>(v) * dv;

  return serac::tuple{modified_source, modified_flux};
}

template <int dim, typename shape_type, typename T, typename space_type,
          typename = std::enable_if_t<space_type{}.family == Family::H1 || space_type{}.family == Family::L2>>
SERAC_HOST_DEVICE auto modify_trial_argument(Dimension<dim>, shape_type shape, space_type /* space */, T u)
{
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
  auto trial_derivative = dot(du_dx, inv(J));

  return serac::tuple{get<VALUE>(u), trial_derivative};
}

template <int dim, typename lambda, typename coord_type, typename shape_type, typename S, typename T, int... i>
SERAC_HOST_DEVICE auto apply_shape_aware_qf_helper([[maybe_unused]] Dimension<dim> d, lambda&& qf, double t,
                                                   coord_type x, shape_type shape, const S& space_tuple,
                                                   const T& arg_tuple, std::integer_sequence<int, i...>)
{
  return qf(t, x + get<VALUE>(shape),
            modify_trial_argument(d, shape, serac::get<i>(space_tuple), serac::get<i>(arg_tuple))...);
}

template <int dim, typename lambda, typename coord_type, typename shape_type, typename... S, typename... T>
SERAC_HOST_DEVICE auto apply_shape_aware_qf(Dimension<dim> d, lambda&& qf, double t, coord_type x,
                                            const shape_type shape, const serac::tuple<S...>& space_tuple,
                                            const serac::tuple<T...>& arg_tuple)
{
  static_assert(sizeof...(S) == sizeof...(T),
                "Size of trial space types and q function arguments not equal in ShapeAwareFunctional.");

  return serac::detail::apply_shape_aware_qf_helper(d, qf, t, x, shape, space_tuple, arg_tuple,
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
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace* shape_fes, const mfem::ParFiniteElementSpace* test_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<test(shape_space, trials...), exec>>(test_fes, prepended_spaces);

    test test_space{};

    SLIC_ERROR_ROOT_IF(test_space.family == Family::HDIV,
                       "Shape-aware functional not implemented for HDiv test functions");
    SLIC_ERROR_ROOT_IF(test_space.family == Family::HCURL,
                       "Shape-aware functional not implemented for HCurl test functions");

    SLIC_ERROR_ROOT_IF(shape_space{}.family != Family::H1, "Only H1 spaces allowed for shape displacements");

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
    functional_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape, auto... qfunc_args) {
          auto qfunc_tuple = make_tuple(qfunc_args...);

          auto space_tuple = make_tuple(get<args>(trial_spaces)...);

          auto unmodified_qf_return =
              detail::apply_shape_aware_qf(Dimension<dim>{}, integrand, time, x, shape, space_tuple, qfunc_tuple);

          return detail::modify_shape_aware_qf_return(Dimension<dim>{}, test{}, shape, unmodified_qf_return);
        },
        domain);
  }

  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    functional_->AddBoundaryIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape, auto... qfunc_args) {
          auto x_prime = x + shape;

          auto unmodified_qf_return = integrand(time, x_prime, qfunc_args...);

          auto n = cross(get<DERIVATIVE>(x_prime));

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
          // to get
          //   q * area_correction * w_old
          // = q * (w_new / w_old) * w_old
          // = q * w_new

          auto area_correction = norm(n) / norm(cross(get<DERIVATIVE>(x)));

          return unmodified_qf_return * area_correction;
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

private:
  std::unique_ptr<Functional<test(shape_space, trials...), exec>> functional_;
};

/**
 * @brief a partial template specialization of ShapeAwareFunctional with test == double, implying "quantity of interest"
 */
template <typename shape_space, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<shape_space, double(trials...), exec> {
  using test = QOI;
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t         num_trial_spaces = sizeof...(trials);

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] trial_fes The trial space
   */
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace*                               shape_fes,
                       std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces> trial_fes)
  {
    std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> prepended_spaces;

    prepended_spaces[0] = shape_fes;

    for (uint32_t i = 0; i < num_trial_spaces; ++i) {
      prepended_spaces[1 + i] = trial_fes[i];
    }

    functional_ = std::make_unique<Functional<double(shape_space, trials...), exec>>(prepended_spaces);

    SLIC_ERROR_ROOT_IF(shape_space{}.family != Family::H1, "Only H1 spaces allowed for shape displacements");

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
    functional_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape, auto... qfunc_args) {
          auto qfunc_tuple = make_tuple(qfunc_args...);

          auto space_tuple = make_tuple(get<args>(trial_spaces)...);

          auto unmodified_qf_return =
              detail::apply_shape_aware_qf(Dimension<dim>{}, integrand, time, x, shape, space_tuple, qfunc_tuple);

          auto dp_dX = get<DERIVATIVE>(shape);

          // x' = x + p
          // J  = dx'/dx
          //    = I + dp_dx
          auto J = Identity<dim>() + dp_dX;

          auto dv = det(J);

          return unmodified_qf_return * dv;
        },
        domain);
  }

  template <int dim, int... args, typename lambda>
  void AddBoundaryIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain)
  {
    functional_->AddBoundaryIntegral(
        Dimension<dim>{}, DependsOn<0, (args + 1)...>{},
        [integrand](double time, auto x, auto shape, auto... qfunc_args) {
          auto x_prime = x + shape;

          auto unmodified_qf_return = integrand(time, x_prime, qfunc_args...);

          auto n = cross(get<DERIVATIVE>(x_prime));

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
          // to get
          //   q * area_correction * w_old
          // = q * (w_new / w_old) * w_old
          // = q * w_new

          auto area_correction = norm(n) / norm(cross(get<DERIVATIVE>(x)));

          return unmodified_qf_return * area_correction;
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

private:
  std::unique_ptr<Functional<double(shape_space, trials...), exec>> functional_;
};

}  // namespace serac
