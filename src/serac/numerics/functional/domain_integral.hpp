// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

#include <array>
#include <memory>

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/geometric_factors.hpp"
#include "serac/numerics/functional/domain_integral_kernels.hpp"
#if defined(__CUDACC__)
#include "serac/numerics/functional/domain_integral_kernels.cuh"
#endif

#include "serac/numerics/functional/debug_print.hpp"

namespace serac {

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <int num_trial_spaces, int Q, ExecutionSpace exec>
class DomainIntegral {
public:
  /**
   * @brief Constructs a @p DomainIntegral from a user-provided quadrature function
   * @tparam dim The dimension of the mesh
   * @tparam test the test function space
   * @tparam trials the trial function space(s)
   * @tparam qpt_data_type The type of the data to store for each quadrature point
   * @param[in] num_elements The number of elements in the mesh
   * @param[in] J The Jacobians of the element transformations at all quadrature points
   * @param[in] X The actual (not reference) coordinates of all quadrature points
   * @param[in] qf The user-provided quadrature function
   * @param[in] active_arguments indices used to select which trial spaces to use in evaluation kernels
   *
   * @see mfem::GeometricFactors
   * @param[inout] qdata The data for each quadrature point
   * @note The @p Dimension parameters are used to assist in the deduction of the @a dim
   * and @a dim template parameters
   */
  template <typename test, typename... trials, auto geom, typename lambda_type, typename qpt_data_type = Nothing>
  DomainIntegral(test, serac::tuple<trials...>, mfem::Mesh& domain, CompileTimeValue<geom>, lambda_type&& qf,
                 std::shared_ptr<QuadratureData<qpt_data_type> > qdata, std::vector<int> active_arguments)
  {
    // GCC 12.0 is incorrectly emitting an error that `dim` is unused,
    // so the attribute is there to suppress that warning/error
    [[maybe_unused]] static constexpr int dim = dimension_of(geom);

    // this is a temporary measure to check correctness for the replacement "GeometricFactor"
    // kernels, must be fixed before merging!
    auto* geom_factors = new serac::GeometricFactors(&domain, Q, to_mfem(geom));
    auto& X            = geom_factors->X;
    auto& J            = geom_factors->J;

    write_to_file(X, "X.txt");
    write_to_file(J, "J.txt");

    size_t num_elements = geom_factors->num_elements;

    constexpr size_t num_active_trial_spaces = sizeof...(trials);
    SLIC_ERROR_ROOT_IF(num_active_trial_spaces != active_arguments.size(),
                       "Error: argument indices inconsistent with provided number of arguments");

    integral_to_functional_ = active_arguments;
    functional_to_integral_ = std::vector<int>(num_trial_spaces, -1);
    for (size_t i = 0; i < active_arguments.size(); i++) {
      functional_to_integral_[static_cast<size_t>(active_arguments[i])] = static_cast<int>(i);
    }

    SERAC_MARK_BEGIN("Domain Integral Set Up");
    using namespace domain_integral;

    constexpr auto geometry                      = geom;
    constexpr auto quadrature_points_per_element = num_quadrature_points(geom, Q);

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    if constexpr (exec == ExecutionSpace::CPU) {
      KernelConfig<Q, geometry, test, trials...> eval_config;

      evaluation_ = EvaluationKernel{NoDifferentiation{}, eval_config, J, X, num_elements, qf, qdata};

      for_constexpr<num_active_trial_spaces>([this, num_elements, quadrature_points_per_element, &J, &X, &qf, qdata,
                                              eval_config](auto index) {
        // allocate memory for the derivatives of the q-function at each quadrature point
        //
        // Note: ptrs' lifetime is managed in an unusual way! It is captured by-value in the
        // action_of_gradient functor below to augment the reference count, and extend its lifetime to match
        // that of the DomainIntegral that allocated it.
        using which_trial_space = typename serac::tuple_element<index, serac::tuple<trials...> >::type;
        using derivative_type   = decltype(get_derivative_type<index, dim, trials...>(qf, (*qdata)(0, 0)));
        auto ptr = accelerator::make_shared_array<exec, derivative_type>(num_elements * quadrature_points_per_element);
        ExecArrayView<derivative_type, 2, exec> qf_derivatives(ptr.get(), num_elements, quadrature_points_per_element);

        evaluation_with_AD_[index] =
            EvaluationKernel{DerivativeWRT<index>{}, eval_config, qf_derivatives, J, X, num_elements, qf, qdata};

        // note: this lambda function captures ptr by-value to extend its lifetime
        //                        vvv
        action_of_gradient_[index] = [ptr, qf_derivatives, num_elements, J](const mfem::Vector& dU, mfem::Vector& dR) {
          domain_integral::action_of_gradient_kernel<geometry, test, which_trial_space, Q>(dU, dR, qf_derivatives, J,
                                                                                           num_elements);
        };

        element_gradient_[index] = [qf_derivatives, num_elements, J](CPUArrayView<double, 3> K_e) {
          domain_integral::element_gradient_kernel<geometry, test, which_trial_space, Q>(K_e, qf_derivatives, J,
                                                                                         num_elements);
        };
      });
    }

// some of the GPU functionality is temporarily disabled to
// help incrementally roll-out the variadic implementation of Functional
// TODO: re-enable GPU kernels in a follow-up PR
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
    SERAC_MARK_END("Domain Integral Set Up");
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = evaluate( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] functional_index specifies which trial space to compute derivatives with respect to (if any)
   * @param[in] update_state a boolean flag for specifying if the material state data should be overwritten or not.
   *                         Typically, this flag is kept false while looking for a solution to the
   *                         residual equations.
   *
   * @note functional_index == -1 implies that this function will call the evaluation kernel that performs no
   * differentiation
   */
  void Mult(const std::array<mfem::Vector, num_trial_spaces>& input_E, mfem::Vector& output_E, int functional_index,
            bool update_state) const
  {
    std::vector<const mfem::Vector*> selected(integral_to_functional_.size());
    for (size_t i = 0; i < integral_to_functional_.size(); i++) {
      selected[i] = &input_E[size_t(integral_to_functional_[i])];
    }

    int index = (functional_index == -1) ? -1 : functional_to_integral_[static_cast<size_t>(functional_index)];
    if (index == -1) {
      SERAC_MARK_BEGIN("Domain Integral Evaluation");
      evaluation_(selected, output_E, update_state);
      SERAC_MARK_END("Domain Integral Evaluation");
    } else {
      SERAC_MARK_BEGIN("Domain Integral Evaluation with AD");
      evaluation_with_AD_[index](selected, output_E, update_state);
      SERAC_MARK_END("Domain Integral Evaluation with AD");
    }
  }

  /**
   * @brief Applies the integral, i.e., @a output_E = gradient( @a input_E )
   * @param[in] input_E The input to the evaluation; per-element DOF values
   * @param[out] output_E The output of the evalution; per-element DOF residuals
   * @param[in] functional_index specifies which trial space input_E correpsonds to
   */
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E, std::size_t functional_index) const
  {
    SERAC_MARK_BEGIN("Domain Integral Action of Gradient");
    int index = functional_to_integral_[functional_index];
    if (index != -1) {
      action_of_gradient_[index](input_E, output_E);
    }
    SERAC_MARK_END("Domain Integral Action of Gradient");
  }

  /**
   * @brief Computes the element stiffness matrices, storing them in an `mfem::Vector` that has been reshaped into a
   * multidimensional array
   * @param[inout] K_e The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * elem)
   * @param[in] functional_index specifies which trial space K_e correpsonds to
   */
  void ComputeElementGradients(ExecArrayView<double, 3, ExecutionSpace::CPU> K_e, std::size_t functional_index) const
  {
    SERAC_MARK_BEGIN("Domain Integral Element Gradient");
    int index = functional_to_integral_[functional_index];
    if (index != -1) {
      element_gradient_[index](K_e);
    }
    SERAC_MARK_END("Domain Integral Element Gradient");
  }

private:
  /// @brief Type-erased handle to evaluation kernel
  std::function<void(const std::vector<const mfem::Vector*>, mfem::Vector&, bool)> evaluation_;

  /// @brief Type-erased handle to evaluation+differentiation kernels
  std::function<void(const std::vector<const mfem::Vector*>, mfem::Vector&, bool)>
      evaluation_with_AD_[num_trial_spaces];

  /// @brief Type-erased handle to action of gradient kernels
  std::function<void(const mfem::Vector&, mfem::Vector&)> action_of_gradient_[num_trial_spaces];

  /// @brief Type-erased handle to gradient matrix assembly kernels
  std::function<void(ExecArrayView<double, 3, exec>)> element_gradient_[num_trial_spaces];

  /**
   * @brief an array for mapping `DomainIntegral` argument indices to `Functional` argument indices
   * e.g. `integral_to_functional_[0] == 2` means that the
   * argument 0 of this integral corresponds to argument 2 in the associated `Functional`
   */
  std::vector<int> integral_to_functional_;

  /**
   * @brief an array for mapping `Functional` argument indices to `DomainIntegral` argument indices
   * e.g. `functional_to_integral[2] == 0` means that the
   * argument 2 of the associated functional corresponds to argument 0 of this integral
   */
  std::vector<int> functional_to_integral_;
};

}  // namespace serac
