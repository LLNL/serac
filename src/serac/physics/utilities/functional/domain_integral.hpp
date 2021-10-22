// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/functional/array.hpp"
#include "serac/physics/utilities/functional/domain_integral_kernels.hpp"
#if defined(__CUDACC__)
#include "serac/physics/utilities/functional/domain_integral_kernels.cuh"
#endif

namespace serac {

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <typename spaces, ExecutionSpace exec>
class DomainIntegral {
public:
  using test_space  = test_space_t<spaces>;   ///< the test function space
  using trial_space = trial_space_t<spaces>;  ///< the trial function space

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
  DomainIntegral(int num_elements, const mfem::Vector& J, const mfem::Vector& X, Dimension<dim>, lambda_type&& qf,
                 QuadratureData<qpt_data_type>& data = dummy_qdata)
      : J_(J), X_(X)
  {
    constexpr auto geometry                      = supported_geometries[dim];
    constexpr auto Q                             = std::max(test_space::order, trial_space::order) + 1;
    constexpr auto quadrature_points_per_element = (dim == 2) ? Q * Q : Q * Q * Q;

    uint32_t num_quadrature_points = quadrature_points_per_element * uint32_t(num_elements);

    // these lines of code figure out the argument types that will be passed
    // into the quadrature function in the finite element kernel.
    //
    // we use them to observe the output type and allocate memory to store
    // the derivative information at each quadrature point
    using x_t             = tensor<double, dim>;
    using u_du_t          = typename detail::lambda_argument<trial_space, dim, dim>::type;
    using qf_result_type  = typename detail::qf_result<lambda_type, x_t, u_du_t, qpt_data_type>::type;
    using derivative_type = decltype(get_gradient(std::declval<qf_result_type>()));

    // allocate memory for the derivatives of the q-function at each quadrature point
    //
    // Note: ptr's lifetime is managed in an unusual way! It is captured by-value in one of the
    // lambda functions below to augment the reference count, and extend its lifetime to match
    // that of the DomainIntegral that allocated it.
    auto ptr = accelerator::make_shared_array<derivative_type, exec>(num_quadrature_points);

    size_t                              n1 = static_cast<size_t>(num_elements);
    size_t                              n2 = static_cast<size_t>(quadrature_points_per_element);
    ArrayView<derivative_type, 2, exec> qf_derivatives{ptr.get(), n1, n2};

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    //
    // note: qf_derivatives is captured by each lambda function below,
    //       to allow the evaluation kernel to pass derivative values to the gradient kernel
    if constexpr (exec == ExecutionSpace::CPU) {
      // note: this lambda function captures ptr by-value to extend its lifetime
      //                   vvv
      evaluation_ = [this, ptr, qf_derivatives, num_elements, qf, &data](const mfem::Vector& U, mfem::Vector& R) {
        domain_integral::evaluation_kernel<geometry, test_space, trial_space, Q>(U, R, qf_derivatives, J_, X_,
                                                                                 num_elements, qf, data);
      };

      action_of_gradient_ = [this, qf_derivatives, num_elements](const mfem::Vector& dU, mfem::Vector& dR) {
        domain_integral::action_of_gradient_kernel<geometry, test_space, trial_space, Q>(dU, dR, qf_derivatives, J_,
                                                                                         num_elements);
      };

      element_gradient_ = [this, qf_derivatives, num_elements](CPUView<double, 3> K_e) {
        domain_integral::element_gradient_kernel<geometry, test_space, trial_space, Q>(K_e, qf_derivatives, J_,
                                                                                       num_elements);
      };
    }

    // TEMPORARY: Add temporary guard so ExecutionSpace::GPU cannot be used when there is no GPU.
    // The proposed future solution is to template the calls on policy (evaluation_kernel<policy>)
#if defined(__CUDACC__)
    if constexpr (exec == ExecutionSpace::GPU) {
      // note: this lambda function captures ptr by-value to extend its lifetime
      //                   vvv
      evaluation_ = [this, ptr, qf_derivatives, num_elements, qf, &data](const mfem::Vector& U, mfem::Vector& R) {
        // TODO: Refactor execution configuration. Blocksize of 128 chosen as a good starting point. Has not been
        // optimized
        serac::detail::GPULaunchConfiguration exec_config{.blocksize = 128};

        domain_integral::evaluation_kernel_cuda<
            geometry, test_space, trial_space, Q,
            serac::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT>(
            exec_config, U, R, qf_derivatives, J_, X_, num_elements, qf, data);
      };

      action_of_gradient_ = [this, qf_derivatives, num_elements](const mfem::Vector& dU, mfem::Vector& dR) {
        // TODO: Refactor execution configuration. Blocksize of 128 chosen as a good starting point. Has not been
        // optimized
        serac::detail::GPULaunchConfiguration exec_config{.blocksize = 128};

        domain_integral::action_of_gradient_kernel<
            geometry, test_space, trial_space, Q,
            serac::detail::ThreadParallelizationStrategy::THREAD_PER_QUADRATURE_POINT>(
            exec_config, dU, dR, qf_derivatives, J_, num_elements);
      };
    }
#endif
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see evaluation_kernel
   */
  void Mult(const mfem::Vector& input_E, mfem::Vector& output_E) const { evaluation_(input_E, output_E); }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @see gradient_kernel
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E) const
  {
    action_of_gradient_(input_E, output_E);
  }

  /**
   * @brief Computes the element stiffness matrices, storing them in an `mfem::Vector` that has been reshaped into a
   * multidimensional array
   * @param[inout] K_e The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * elem)
   */
  void ComputeElementGradients(ArrayView<double, 3, ExecutionSpace::CPU> K_e) const { element_gradient_(K_e); }

private:
  /**
   * @brief Jacobians of the element transformations at all quadrature points
   */
  const mfem::Vector J_;

  /**
   * @brief Mapped (physical) coordinates of all quadrature points
   */
  const mfem::Vector X_;

  /**
   * @brief Type-erased handle to evaluation kernel
   * @see evaluation_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation_;

  /**
   * @brief Type-erased handle to gradient kernel
   * @see gradient_kernel
   */
  std::function<void(const mfem::Vector&, mfem::Vector&)> action_of_gradient_;

  /**
   * @brief Type-erased handle to gradient matrix assembly kernel
   * @see gradient_matrix_kernel
   */
  std::function<void(ArrayView<double, 3, exec>)> element_gradient_;
};

}  // namespace serac
