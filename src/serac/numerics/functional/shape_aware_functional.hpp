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

namespace serac {

/// @cond
template <typename T, ExecutionSpace exec = serac::default_execution_space>
class ShapeAwareFunctional;
/// @endcond

/**
 * @brief Intended to be like @p std::function for finite element kernels
 *
 * That is: you tell it the inputs (trial spaces) for a kernel, and the outputs (test space) like @p std::function.
 *
 * For example, this code represents a function that takes an integer argument and returns a double:
 * @code{.cpp}
 * std::function< double(int) > my_func;
 * @endcode
 * And this represents a function that takes values from an Hcurl field and returns a
 * residual vector associated with an H1 field:
 * @code{.cpp}
 * Functional< H1(Hcurl) > my_residual;
 * @endcode
 *
 * @tparam test The space of test functions to use
 * @tparam trial The space of trial functions to use
 * @tparam exec whether to carry out calculations on CPU or GPU
 *
 * To use this class, you use the methods @p Functional::Add****Integral(integrand,domain_of_integration)
 * where @p integrand is a q-function lambda or functor and @p domain_of_integration is an @p mfem::mesh
 *
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 *
 * @code{.cpp}
 * // for domains made up of quadrilaterals embedded in R^2
 * my_residual.AddAreaIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<2>{}, integrand, domain_of_integration);
 *
 * // for domains made up of quadrilaterals embedded in R^3
 * my_residual.AddSurfaceIntegral(integrand, domain_of_integration);
 *
 * // for domains made up of hexahedra embedded in R^3
 * my_residual.AddVolumeIntegral(integrand, domain_of_integration);
 * // alternatively...
 * my_residual.AddDomainIntegral(Dimension<3>{}, integrand, domain_of_integration);
 * @endcode
 */
template <typename test, typename shape_trial, typename... trials, ExecutionSpace exec>
class ShapeAwareFunctional<test(shape_trial, trials...), exec> {
  static constexpr tuple<trials...> trial_spaces{};
  static constexpr uint32_t         num_trial_spaces = sizeof...(trials);

public:
  /**
   * @brief Constructs using @p mfem::ParFiniteElementSpace objects corresponding to the test/trial spaces
   * @param[in] test_fes The (non-qoi) test space
   * @param[in] trial_fes The trial space
   */
  ShapeAwareFunctional(const mfem::ParFiniteElementSpace*  test_fes,
                       const mfem::ParFiniteElementSpace* shape_fes,
             std::array<const mfem::ParFiniteElementSpace*, num_trial_spaces + 1> trial_fes)
      : functional_(test_fes, prepend(shape_fes, trial_fes))
  {
  }

  template <int dim, int... args, typename lambda, typename qpt_data_type = Nothing>
  void AddDomainIntegral(Dimension<dim>, DependsOn<args...>, lambda&& integrand, mfem::Mesh& domain,
                         std::shared_ptr<QuadratureData<qpt_data_type>> qdata = NoQData)
  {
    functional_.AddDomainIntegral(Dimension<dim>{}, DependsOn<1, (1 + args)...>{},
     [integrand, spaces = trial_spaces] (double time, auto x, auto& state, auto shape, auto... qfunc_args)
     {
      auto adjusted_position = x + shape;
      auto qfunc_tuple = make_tuple(qfunc_args...);

      // adjust the 1-forms of the input states to account for the shape modifications
      for (size_t i = 0; i < spaces.size(); ++i) {
        
        auto fe_space       = get<i>(spaces);
        using fe_space_type = decltype(fe_space);

        [[maybe_unused]] constexpr int VALUE      = 0;
        [[maybe_unused]] constexpr int DERIVATIVE = 1;

        auto dp_dX   = get<DERIVATIVE>(shape);
        auto p       = get<VALUE>(shape);

        auto u = get<i>(qfunc_tuple);

        if constexpr (fe_space_type::family == Family::H1) {
          
          auto du_dX = get<DERIVATIVE>(get<i>(qfunc_tuple));



        }
      }

      return integrand(time, adjusted_position, state, qfunc_args...);
     },
     domain, qdata);
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
  Functional<test(shape_trial, trials...), exec> functional_;
};

}  // namespace serac

