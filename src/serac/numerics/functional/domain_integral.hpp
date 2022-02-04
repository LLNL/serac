// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file domain_integral.hpp
 *
 * @brief This file contains the implementations of finite element kernels with matching geometric and
 *   spatial dimensions (e.g. quadrilaterals in 2D, hexahedra in 3D), and the class template DomainIntegral
 *   for encapsulating those kernels.
 */
#pragma once

#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/domain_integral_kernels.hpp"
#if defined(__CUDACC__)
#include "serac/numerics/functional/domain_integral_kernels.cuh"
#endif

namespace serac {
template <typename spaces, ExecutionSpace exec>
class DomainIntegral;

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <typename test, typename... trials, ExecutionSpace exec>
class DomainIntegral<test(trials...), exec> {
public:
  static constexpr tuple<trials...> trial_spaces{};                        ///< a tuple of the different trial spaces
  static constexpr int              num_trial_spaces = sizeof...(trials);  ///< how many trial spaces were specified

  /**
   * @brief Constructs a @p DomainIntegral from a user-provided quadrature function
   * @tparam dim The dimension of the mesh
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] num_elements The number of elements in the mesh
   * @param[in] J The Jacobians of the element transformations at all quadrature points
   * @param[in] X The actual (not reference) coordinates of all quadrature points
   * @see mfem::GeometricFactors
   * @param[in] qf The user-provided quadrature function
   * @param[inout] data The data for each quadrature point
   * @note The @p Dimension parameters are used to assist in the deduction of the @a dim
   * and @a dim template parameters
   */
  template <int dim, typename lambda_type, typename qpt_data_type = void>
  DomainIntegral(size_t num_elements, const mfem::Vector& J, const mfem::Vector& X, Dimension<dim>, lambda_type&& qf,
                 QuadratureData<qpt_data_type>& data = dummy_qdata)
  {
    using namespace domain_integral;

    [[maybe_unused]] constexpr auto geometry                      = supported_geometries[dim];
    constexpr auto                  Q                             = std::max({test::order, trials::order...}) + 1;
    constexpr auto                  quadrature_points_per_element = (dim == 2) ? Q * Q : Q * Q * Q;

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    if constexpr (exec == ExecutionSpace::CPU) {
      KernelConfig<Q, geometry, test, trials...> eval_config;

      evaluation_ = EvaluationKernel{eval_config, J, X, num_elements, qf, data};

      for_constexpr<num_trial_spaces>([this, num_elements, quadrature_points_per_element, &J, &X, &qf, &data,
                                       eval_config](auto i) {
        // allocate memory for the derivatives of the q-function at each quadrature point
        //
        // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
        // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
        // that of the DomainIntegral that allocated it.
        using which_trial_space = typename serac::tuple_element<i, serac::tuple<trials...> >::type;
        using derivative_type   = decltype(get_derivative_type<i, dim, trials...>(qf, data(0, 0)));
        auto ptr = accelerator::make_shared_array<exec, derivative_type>(num_elements * quadrature_points_per_element);
        ExecArrayView<derivative_type, 2, exec> qf_derivatives(ptr.get(), num_elements, quadrature_points_per_element);

        evaluation_with_AD_[i] =
            EvaluationKernel{DerivativeWRT<i>{}, eval_config, qf_derivatives, J, X, num_elements, qf, data};

        // note: this lambda function captures ptr by-value to extend its lifetime
        //                        vvv
        action_of_gradient_[i] = [ptr, qf_derivatives, num_elements, J](const mfem::Vector& dU, mfem::Vector& dR) {
          domain_integral::action_of_gradient_kernel<geometry, test, which_trial_space, Q>(dU, dR, qf_derivatives, J,
                                                                                           num_elements);
        };

        element_gradient_[i] = [qf_derivatives, num_elements, J](CPUArrayView<double, 3> K_e) {
          domain_integral::element_gradient_kernel<geometry, test, which_trial_space, Q>(K_e, qf_derivatives, J,
                                                                                         num_elements);
        };
      });
    }

#if 0
    // TEMPORARY: Add temporary guard so ExecutionSpace::GPU cannot be used when there is no GPU.
    // The proposed future solution is to template the calls on policy (evaluation_kernel<policy>)
#if defined(__CUDACC__)
    if constexpr (exec == ExecutionSpace::GPU) {
      // note: this lambda function captures ptr by-value to extend its lifetime
      //                   vvv
      evaluation_ = [this, ptr, qf_derivatives, num_elements, qf, &data, &J, &X](const mfem::Vector& U,
                                                                                 mfem::Vector&       R) {
        // TODO: Refactor execution configuration. Blocksize of 128 chosen as a good starting point. Has not been
        // optimized
        serac::detail::GPULaunchConfiguration exec_config{.blocksize = 128};

        domain_integral::evaluation_kernel_cuda<
            geometry, test_space, trial_space, Q,
            serac::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT>(
            exec_config, U, R, qf_derivatives, J, X, num_elements, qf, data);
      };

      action_of_gradient_ = [this, qf_derivatives, num_elements, &J](const mfem::Vector& dU, mfem::Vector& dR) {
        // TODO: Refactor execution configuration. Blocksize of 128 chosen as a good starting point. Has not been
        // optimized
        serac::detail::GPULaunchConfiguration exec_config{.blocksize = 128};

        domain_integral::action_of_gradient_kernel<
            geometry, test_space, trial_space, Q,
            serac::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT>(exec_config, dU, dR,
                                                                                       qf_derivatives, J, num_elements);
      };
    }
#endif

#endif
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] which_trial_space specifies which trial space to compute derivatives with respect to (if any)
   *
   * @note which_trial_space == -1 implies that this function will call the evaluation kernel that performs no
   * differentiation
   */
  void Mult(const std::array<mfem::Vector, num_trial_spaces>& input_E, mfem::Vector& output_E,
            int which_trial_space) const
  {
    if (which_trial_space == -1) {
      evaluation_(input_E, output_E);
    } else {
      evaluation_with_AD_[which_trial_space](input_E, output_E);
    }
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] which_trial_space specifies which trial space input_E correpsonds to
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E, size_t which_trial_space) const
  {
    action_of_gradient_[which_trial_space](input_E, output_E);
  }

  /**
   * @brief Computes the element stiffness matrices, storing them in an `mfem::Vector` that has been reshaped into a
   * multidimensional array
   * @param[inout] K_e The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * elem)
   * @param[in] which_trial_space specifies which trial space K_e correpsonds to
   */
  void ComputeElementGradients(ExecArrayView<double, 3, ExecutionSpace::CPU> K_e, size_t which_trial_space) const
  {
    element_gradient_[which_trial_space](K_e);
  }

private:
  /// @brief Type-erased handle to evaluation kernel
  std::function<void(const std::array<mfem::Vector, num_trial_spaces>&, mfem::Vector&)> evaluation_;

  /// @brief Type-erased handle to evaluation+differentiation kernels
  std::function<void(const std::array<mfem::Vector, num_trial_spaces>&, mfem::Vector&)>
      evaluation_with_AD_[num_trial_spaces];

  /// @brief Type-erased handle to action of gradient kernels
  std::function<void(const mfem::Vector&, mfem::Vector&)> action_of_gradient_[num_trial_spaces];

  /// @brief Type-erased handle to gradient matrix assembly kernels
  std::function<void(ExecArrayView<double, 3, exec>)> element_gradient_[num_trial_spaces];
};

}  // namespace serac
