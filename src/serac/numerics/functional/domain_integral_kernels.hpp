// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/quadrature_data.hpp"
#include "serac/numerics/functional/function_signature.hpp"
#include "serac/numerics/functional/differentiate_wrt.hpp"

#include <RAJA/index/RangeSegment.hpp>
#include <RAJA/RAJA.hpp>
#include <array>
#include <cstdint>

namespace serac {

namespace domain_integral {

/**
 *  @tparam space the user-specified trial space
 *  @tparam dimension describes whether the problem is 1D, 2D, or 3D
 *
 *  @brief a struct used to encode what type of arguments will be passed to a domain integral q-function, for the given
 * trial space
 */
template <typename space, typename dimension>
SERAC_HOST_DEVICE struct QFunctionArgument;

/// @overload
template <int p, int dim>
SERAC_HOST_DEVICE struct QFunctionArgument<H1<p, 1>, Dimension<dim> > {
  using type = tuple<double, tensor<double, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
SERAC_HOST_DEVICE struct QFunctionArgument<H1<p, c>, Dimension<dim> > {
  using type = tuple<tensor<double, c>, tensor<double, c, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int dim>
SERAC_HOST_DEVICE struct QFunctionArgument<L2<p, 1>, Dimension<dim> > {
  using type = tuple<double, tensor<double, dim> >;  ///< what will be passed to the q-function
};
/// @overload
template <int p, int c, int dim>
SERAC_HOST_DEVICE struct QFunctionArgument<L2<p, c>, Dimension<dim> > {
  using type = tuple<tensor<double, c>, tensor<double, c, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p>
SERAC_HOST_DEVICE struct QFunctionArgument<Hcurl<p>, Dimension<2> > {
  using type = tuple<tensor<double, 2>, double>;  ///< what will be passed to the q-function
};

/// @overload
template <int p>
SERAC_HOST_DEVICE struct QFunctionArgument<Hcurl<p>, Dimension<3> > {
  using type = tuple<tensor<double, 3>, tensor<double, 3> >;  ///< what will be passed to the q-function
};

/// @brief layer of indirection needed to unpack the entries of the argument tuple
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, typename coords_type, typename T, typename qpt_data_type, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(lambda&& qf, coords_type&& x_q, qpt_data_type&& qpt_data, const T& arg_tuple,
                                       std::integer_sequence<int, i...>)
{
  if constexpr (std::is_same<typename std::decay<qpt_data_type>::type, Nothing>::value) {
    return qf(x_q, serac::get<i>(arg_tuple)...);
  } else {
    return qf(x_q, qpt_data, serac::get<i>(arg_tuple)...);
  }
}

/**
 * @brief Actually calls the q-function
 * This is an indirection layer to provide a transparent call site usage regardless of whether
 * quadrature point (state) information is required
 * @param[in] qf The quadrature function functor object
 * @param[in] x_q The physical coordinates of the quadrature point
 * @param[in] arg_tuple The values and derivatives at the quadrature point, as a dual
 * @param[inout] qpt_data The state information at the quadrature point
 */
template <typename lambda, typename coords_type, typename... T, typename qpt_data_type>
SERAC_HOST_DEVICE auto apply_qf(lambda&& qf, coords_type&& x_q, qpt_data_type&& qpt_data,
                                const serac::tuple<T...>& arg_tuple)
{
  return apply_qf_helper(qf, x_q, qpt_data, arg_tuple,
                         std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

template <int i, int dim, typename... trials, typename lambda, typename qpt_data_type>
auto get_derivative_type(lambda qf, qpt_data_type&& qpt_data)
{
  using qf_arguments = serac::tuple<typename QFunctionArgument<trials, serac::Dimension<dim> >::type...>;
  return get_gradient(apply_qf(qf, tensor<double, dim>{}, qpt_data, make_dual_wrt<i>(qf_arguments{})));
};

template <typename lambda, int dim, int n, typename... T>
SERAC_HOST_DEVICE auto batch_apply_qf_no_qdata(lambda qf, const tensor<double, dim, n> x, const T&... inputs)
{
  using return_type = decltype(qf(tensor<double, dim>{}, T{}[0]...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim> x_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = x(j, i);
    }
    outputs[i] = qf(x_q, inputs[i]...);
  }
  return outputs;
}

template <typename lambda, int dim, int n, typename qpt_data_type, typename... T>
SERAC_HOST_DEVICE auto batch_apply_qf(lambda qf, const tensor<double, dim, n> x, qpt_data_type* qpt_data,
                                      bool update_state, const T&... inputs)
{
  using return_type = decltype(qf(tensor<double, dim>{}, qpt_data[0], T{}[0]...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim> x_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = x(j, i);
    }

    auto qdata = qpt_data[i];
    outputs[i] = qf(x_q, qdata, inputs[i]...);
    if (update_state) {
      qpt_data[i] = qdata;
    }
  }
  return outputs;
}

template <uint32_t differentiation_index, int Q, mfem::Geometry::Type geom, typename test_element_type,
          typename trial_element_tuple_type, typename lambda_type, typename state_type, typename derivative_type,
          int... indices>
void evaluation_kernel_impl(trial_element_tuple_type          trial_elements, test_element_type,
                            const std::vector<const double*>& inputs, double* outputs, const double* positions,
                            const double* jacobians, lambda_type qf,
                            [[maybe_unused]] axom::ArrayView<state_type, 2> qf_state,
                            [[maybe_unused]] derivative_type* qf_derivatives, uint32_t num_elements, bool update_state,
                            camp::int_seq<int, indices...>)
{
  // mfem provides this information as opaque arrays of doubles,
  // so we reinterpret the pointer with
  auto                           r = reinterpret_cast<typename test_element_type::dof_type*>(outputs);
  auto                           x = reinterpret_cast<const typename batched_position<geom, Q>::type*>(positions);
  auto                           J = reinterpret_cast<const typename batched_jacobian<geom, Q>::type*>(jacobians);
  TensorProductQuadratureRule<Q> rule{};

  [[maybe_unused]] auto qpts_per_elem = num_quadrature_points(geom, Q);

  [[maybe_unused]] tuple u = {
      reinterpret_cast<const typename decltype(type<indices>(trial_elements))::dof_type*>(inputs[indices])...};

#if defined(USE_CUDA)
  std::cout << "USING CUDA :)\n";
  using policy = RAJA::cuda_exec<512>;
#else
  std::cout << "USING SIMD :)\n";
  using policy = RAJA::simd_exec;
#endif

  // for each element in the domain
  RAJA::forall<policy>(
      RAJA::TypedRangeSegment<uint32_t>(0, num_elements),
      [J, x, qf, u, qpts_per_elem, rule, r, qf_state, qf_derivatives, update_state] RAJA_HOST_DEVICE(uint32_t e) {
        // load the jacobians and positions for each quadrature point in this element
        // printf("HERE1\n");
        auto J_e = J[e];
        auto x_e = x[e];
        // printf("HERE2\n");
        static constexpr trial_element_type empty_trial_element{};
        // batch-calculate values / derivatives of each trial space, at each quadrature point
        [[maybe_unused]] tuple qf_inputs = {promote_each_to_dual_when<indices == differentiation_index>(
            get<indices>(empty_trial_element).interpolate(get<indices>(u)[e], rule))...};
        // printf("HERE3\n");
        // use J_e to transform values / derivatives on the parent element
        // to the to the corresponding values / derivatives on the physical element
        (parent_to_physical<get<indices>(empty_trial_element).family>(get<indices>(qf_inputs), J_e), ...);
        // printf("HERE4\n");
        // (batch) evalute the q-function at each quadrature point
        //
        // note: the weird immediately-invoked lambda expression is
        // a workaround for a bug in GCC(<12.0) where it fails to
        // decide which function overload to use, and crashes
        auto qf_outputs = [&]() {
          if constexpr (std::is_same_v<state_type, Nothing>) {
            return batch_apply_qf_no_qdata(qf, x_e, get<indices>(qf_inputs)...);
          } else {
            return batch_apply_qf(qf, x_e, &qf_state(e, 0), update_state, get<indices>(qf_inputs)...);
          }
        }();
        // printf("HERE5\n");
        // use J to transform sources / fluxes on the physical element
        // back to the corresponding sources / fluxes on the parent element
        physical_to_parent<test_element::family>(qf_outputs, J_e);
        // printf("HERE6\n");
        // write out the q-function derivatives after applying the
        // physical_to_parent transformation, so that those transformations
        // won't need to be applied in the action_of_gradient and element_gradient kernels
        if constexpr (differentiation_index != serac::NO_DIFFERENTIATION) {
          for (int q = 0; q < leading_dimension(qf_outputs); q++) {
            qf_derivatives[e * uint32_t(qpts_per_elem) + uint32_t(q)] = get_gradient(qf_outputs[q]);
          }
        }
        // printf("HERE7\n");
        // (batch) integrate the material response against the test-space basis functions
        test_element::integrate(get_value(qf_outputs), rule, &r[e]);
        // printf("HERE8\n");
      });

  return;
}

//clang-format off
template <bool is_QOI, typename S, typename T>
SERAC_HOST_DEVICE auto chain_rule(const S& dfdx, const T& dx)
{
  if constexpr (is_QOI) {
    return serac::chain_rule(serac::get<0>(dfdx), serac::get<0>(dx)) +
           serac::chain_rule(serac::get<1>(dfdx), serac::get<1>(dx));
  }

  if constexpr (!is_QOI) {
    return serac::tuple{serac::chain_rule(serac::get<0>(serac::get<0>(dfdx)), serac::get<0>(dx)) +
                            serac::chain_rule(serac::get<1>(serac::get<0>(dfdx)), serac::get<1>(dx)),
                        serac::chain_rule(serac::get<0>(serac::get<1>(dfdx)), serac::get<0>(dx)) +
                            serac::chain_rule(serac::get<1>(serac::get<1>(dfdx)), serac::get<1>(dx))};
  }
}
//clang-format on

template <bool is_QOI, typename derivative_type, int n, typename T>
SERAC_HOST_DEVICE auto batch_apply_chain_rule(derivative_type* qf_derivatives, const tensor<T, n>& inputs)
{
  using return_type = decltype(chain_rule<is_QOI>(derivative_type{}, T{}));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    outputs[i] = chain_rule<is_QOI>(qf_derivatives[i], inputs[i]);
  }
  return outputs;
}

/**
 * @brief The base kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO), QOI}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p DomainIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q parameter describing number of quadrature points (see num_quadrature_points() function for more details)
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 * @note lambda does not appear as a template argument, as the directional derivative is
 * inherently just a linear transformation
 *
 * @param[in] dU The full set of per-element DOF values (primary input)
 * @param[inout] dR The full set of per-element residuals (primary output)
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */

template <int Q, mfem::Geometry::Type g, typename test, typename trial, typename derivatives_type>
void action_of_gradient_kernel(const double* dU, double* dR, derivatives_type* qf_derivatives, std::size_t num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  constexpr bool is_QOI   = (test::family == Family::QOI);
  constexpr int  num_qpts = num_quadrature_points(g, Q);

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto                                 du = reinterpret_cast<const typename trial_element::dof_type*>(dU);
  auto                                 dr = reinterpret_cast<typename test_element::dof_type*>(dR);
  constexpr TensorProductQuadratureRule<Q> rule{};

#if defined(USE_CUDA)
  using policy = RAJA::cuda_exec<512>;
#else
  using policy = RAJA::simd_exec;
#endif

  // for each element in the domain
  RAJA::forall<policy>(RAJA::TypedRangeSegment<uint32_t>(0, num_elements), [=] RAJA_HOST_DEVICE(uint32_t e) {
    // (batch) interpolate each quadrature point's value
    auto qf_inputs = trial_element::interpolate(du[e], rule);

    // (batch) evalute the q-function at each quadrature point
    auto qf_outputs = batch_apply_chain_rule<is_QOI>(qf_derivatives + e * num_qpts, qf_inputs);

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(qf_outputs, rule, &dr[e]);
  }
}

/**
 * @brief The base kernel template used to compute tangent element entries that can be assembled
 * into a tangent matrix
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO), QOI}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p Integral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q parameter describing number of quadrature points (see num_quadrature_points() function for more details)
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 *
 * @param[inout] dk 3-dimensional array storing the element gradient matrices
 * @param[in] derivatives_ptr pointer to data describing the derivatives of the q-function with respect to its arguments
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <mfem::Geometry::Type g, typename test, typename trial, int Q, typename derivatives_type>
#if defined(USE_CUDA)
void element_gradient_kernel(ExecArrayView<double, 3, ExecutionSpace::GPU> dK,
#else
void element_gradient_kernel(ExecArrayView<double, 3, ExecutionSpace::CPU> dK,
#endif
                             derivatives_type* qf_derivatives, std::size_t num_elements)
{
  // quantities of interest have no flux term, so we pad the derivative
  // tuple with a "zero" type in the second position to treat it like the standard case
  constexpr bool is_QOI        = test::family == Family::QOI;
  using padded_derivative_type = std::conditional_t<is_QOI, tuple<derivatives_type, zero>, derivatives_type>;

  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  constexpr int nquad = num_quadrature_points(g, Q);

  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    auto* output_ptr = reinterpret_cast<typename test_element::dof_type*>(&dK(e, 0, 0));

    tensor<padded_derivative_type, nquad> derivatives{};
    for (int q = 0; q < nquad; q++) {
      if constexpr (is_QOI) {
        get<0>(derivatives(q)) = qf_derivatives[e * nquad + uint32_t(q)];
      } else {
        derivatives(q) = qf_derivatives[e * nquad + uint32_t(q)];
      }
    }

    for (int J = 0; J < trial_element::ndof; J++) {
      auto source_and_flux = trial_element::batch_apply_shape_fn(J, derivatives, rule);
      test_element::integrate(source_and_flux, rule, output_ptr + J, trial_element::ndof);
    }
  }
}

template <uint32_t wrt, int Q, mfem::Geometry::Type geom, typename signature, typename lambda_type, typename state_type,
          typename derivative_type>
std::function<void(const std::vector<const double*>&, double*, bool)> evaluation_kernel(
    signature s, lambda_type qf, const double* positions, const double* jacobians,
    std::shared_ptr<QuadratureData<state_type> > qf_state, std::shared_ptr<derivative_type> qf_derivatives,
    uint32_t num_elements)
{
  auto trial_elements = trial_elements_tuple<geom>(s);
  auto test           = get_test_element<geom>(s);
  return [=](const std::vector<const double*>& inputs, double* outputs, bool update_state) {
    domain_integral::evaluation_kernel_impl<wrt, Q, geom>(trial_elements, test, inputs, outputs, positions, jacobians,
                                                          qf, (*qf_state)[geom], qf_derivatives.get(), num_elements,
                                                          update_state, s.index_seq);
  };
}

template <int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type>
std::function<void(const double*, double*)> jacobian_vector_product_kernel(
    signature, std::shared_ptr<derivative_type> qf_derivatives, uint32_t num_elements)
{
  return [=](const double* du, double* dr) {
    using test_space  = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types>::type;
    action_of_gradient_kernel<Q, geom, test_space, trial_space>(du, dr, qf_derivatives.get(), num_elements);
  };
}

template <int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type>
#if defined(USE_CUDA)
std::function<void(ExecArrayView<double, 3, ExecutionSpace::GPU>)>
#else
std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)>
#endif
element_gradient_kernel(signature, std::shared_ptr<derivative_type> qf_derivatives, uint32_t num_elements)
{
#if defined(USE_CUDA)
  return [=](ExecArrayView<double, 3, ExecutionSpace::GPU> K_elem) {
#else
  return [=](ExecArrayView<double, 3, ExecutionSpace::CPU> K_elem) {
#endif
    using test_space  = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types>::type;
    element_gradient_kernel<geom, test_space, trial_space, Q>(K_elem, qf_derivatives.get(), num_elements);
  };
}

}  // namespace domain_integral

}  // namespace serac
