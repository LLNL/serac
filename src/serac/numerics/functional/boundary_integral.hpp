// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_integral.hpp
 *
 * @brief This file defines a class, BoundaryIntegral, for integrating q-functions against finite element
 * basis functions on a mesh boundary region.
 */
#pragma once

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/quadrature.hpp"
#include "serac/numerics/functional/tuple_arithmetic.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"
#include "serac/numerics/functional/boundary_integral_kernels.hpp"
#if defined(__CUDACC__)
#include "serac/numerics/functional/boundary_integral_kernels.cuh"
#endif

namespace serac {

/**
 * @brief Describes a single boundary integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 * @tparam exec whether or not the calculation and memory will be on the CPU or GPU
 */
template <int num_trial_spaces, int Q, ExecutionSpace exec>
class BoundaryIntegral{
public:

  /**
   * @brief Constructs an @p BoundaryIntegral from a user-provided quadrature function
   * @tparam dim The dimension of the element (2 for quad, 3 for hex, etc)
   * @tparam test the test function space 
   * @tparam trials the trial function space(s)
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] num_elements The number of elements in the mesh
   * @param[in] J The Jacobians of the element transformations at all quadrature points
   * @param[in] X The actual (not reference) coordinates of all quadrature points
   * @param[in] N The unit normals of all quadrature points
   * @param[in] qf The user-provided quadrature function
   * @param[in] arg_indices indices used to select which trail spaces to use in evaluation kernels
   * 
   * @see mfem::GeometricFactors
   * @note The @p Dimension parameters are used to assist in the deduction of the dim template parameter
   */
  template <int dim, typename test, typename ... trials, typename lambda_type, typename qpt_data_type = void>
  BoundaryIntegral(test, serac::tuple< trials ... >, size_t num_elements, const mfem::Vector& J, const mfem::Vector& X, const mfem::Vector& N,
                   Dimension<dim>, lambda_type&& qf, std::vector<int> arg_indices)
  {
    static_assert(((trials::family == Family::H1) && ...) , "Error: boundary integrals currently only support H1 trial spaces");

    argument_indices = arg_indices;

    constexpr size_t num_active_trial_spaces = sizeof ... (trials);

    SLIC_ERROR_ROOT_IF(num_active_trial_spaces != arg_indices.size(), "Error: argument indices inconsistent with provided number of arguments");

    using namespace boundary_integral;

    constexpr auto geometry                      = supported_geometries[dim];
    constexpr auto quadrature_points_per_element = detail::pow(Q, dim);

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    if constexpr (exec == ExecutionSpace::CPU) {
      KernelConfig<Q, geometry, test, trials...> eval_config;

      evaluation_ = EvaluationKernel{eval_config, J, X, N, num_elements, qf};

      for_constexpr<num_active_trial_spaces>([this, num_elements, quadrature_points_per_element, &J, &X, &N, &qf,
                                       eval_config](auto i) {
        // allocate memory for the derivatives of the q-function at each quadrature point
        //
        // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
        // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
        // that of the DomainIntegral that allocated it.
        using which_trial_space = typename serac::tuple_element<i, serac::tuple<trials...> >::type;
        using derivative_type   = decltype(get_derivative_type<i, dim, trials...>(qf));
        auto ptr = accelerator::make_shared_array<exec, derivative_type>(num_elements * quadrature_points_per_element);
        ExecArrayView<derivative_type, 2, exec> qf_derivatives(ptr.get(), num_elements, quadrature_points_per_element);

        evaluation_with_AD_[i] =
            EvaluationKernel{DerivativeWRT<i>{}, eval_config, qf_derivatives, J, X, N, num_elements, qf};

        // note: this lambda function captures ptr by-value to extend its lifetime
        //                        vvv
        action_of_gradient_[i] = [ptr, qf_derivatives, num_elements, J](const mfem::Vector& dU, mfem::Vector& dR) {
          action_of_gradient_kernel<geometry, test, which_trial_space, Q>(dU, dR, qf_derivatives, J, num_elements);
        };

        element_gradient_[i] = [qf_derivatives, num_elements, J](CPUArrayView<double, 3> K_e) {
          element_gradient_kernel<geometry, test, which_trial_space, Q>(K_e, qf_derivatives, J, num_elements);
        };
      });
    }
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] which_trial_space the index of the argument being differentiated,
   *    which == -1 corresponds to direct evaluation without any differentiation
   */
  void Mult(const std::array<mfem::Vector, num_trial_spaces>& input_E, mfem::Vector& output_E, int which_trial_space) const
  {
    int which_local_trial_space = -1;
    std::vector < const mfem::Vector * > selected(argument_indices.size());
    for (size_t i = 0; i < argument_indices.size(); i++) {
      selected[i] = &input_E[size_t(argument_indices[i])];

      // now that these integrals don't depend on all of the arguments,
      // we have to figure out which of our local arguments correspond to
      // `which_trial_space` 
      //
      // if this calculation doesn't depend on that argument at all, then
      // we just call the evaluation kernel without any differentiation
      // 
      if (which_trial_space == argument_indices[i]) {
        which_local_trial_space = int(i);
      }
    }

    if (which_local_trial_space == -1) {
      evaluation_(selected, output_E);
    } else {
      evaluation_with_AD_[which_local_trial_space](selected, output_E);
    }
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] which_trial_space the index of the argument being differentiated
   * @see action_of_gradient_kernel
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E, std::size_t which_trial_space) const
  {
    int which_local_trial_space = -1;
    for (size_t i = 0; i < argument_indices.size(); i++) {
      if (which_trial_space == std::size_t(argument_indices[i])) {
        which_local_trial_space = int(i);
      }
    }
    if (which_local_trial_space != -1) {
      action_of_gradient_[which_local_trial_space](input_E, output_E);
    }
  }

  /**
   * @brief Computes the derivative of each element's residual with respect to the element values
   * @param[inout] K_b The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * nelems)
   * @param[in] which_trial_space the index of the argument being differentiated
   */
  void ComputeElementGradients(ExecArrayView<double, 3, ExecutionSpace::CPU> K_b, std::size_t which_trial_space) const
  {
    int which_local_trial_space = -1;
    for (size_t i = 0; i < argument_indices.size(); i++) {
      if (which_trial_space == std::size_t(argument_indices[i])) {
        which_local_trial_space = int(i);
      }
    }
    if (which_local_trial_space != -1) { 
      element_gradient_[which_local_trial_space](K_b);
    }
  }

private:
  /// @brief kernel for integrating the q-function over the domain
  std::function<void(const std::vector< const mfem::Vector * >, mfem::Vector&)> evaluation_;

  /// @brief kernels for integrating the q-function over the domain, and caching some data about its derivatives
  std::function<void(const std::vector< const mfem::Vector * >, mfem::Vector&)>
      evaluation_with_AD_[num_trial_spaces];

  /// @brief kernels for computing directional derivatives, using the most recently cached q-function derivatives
  std::function<void(const mfem::Vector&, mfem::Vector&)> action_of_gradient_[num_trial_spaces];

  /// @brief kernels for computing consistent "element stiffness" matrices
  std::function<void(ExecArrayView<double, 3, exec>)> element_gradient_[num_trial_spaces];

  std::vector < int > argument_indices;
};

}  // namespace serac
