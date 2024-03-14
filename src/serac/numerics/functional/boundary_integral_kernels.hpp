// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <array>

#include "serac/numerics/functional/quadrature_data.hpp"
#include "serac/numerics/functional/differentiate_wrt.hpp"

namespace serac {

namespace boundary_integral {

/**
 *  @tparam space the user-specified trial space
 *  @tparam dimension describes whether the problem is 1D, 2D, or 3D
 *
 *  @brief a struct used to encode what type of arguments will be passed to a domain integral q-function, for the given
 * trial space
 */
template <typename space, typename dimension>
struct QFunctionArgument;

/// @overload
template <int p>
struct QFunctionArgument<H1<p, 1>, Dimension<1>> {
  using type = serac::tuple<double, double>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int dim>
struct QFunctionArgument<H1<p, 1>, Dimension<dim>> {
  using type = serac::tuple<double, tensor<double, dim>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c>
struct QFunctionArgument<H1<p, c>, Dimension<1>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<H1<p, c>, Dimension<dim>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c, dim>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p>
struct QFunctionArgument<L2<p, 1>, Dimension<1>> {
  using type = serac::tuple<double, double>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int dim>
struct QFunctionArgument<L2<p, 1>, Dimension<dim>> {
  using type = serac::tuple<double, tensor<double, dim>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c>
struct QFunctionArgument<L2<p, c>, Dimension<1>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<L2<p, c>, Dimension<dim>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c, dim>>;  ///< what will be passed to the q-function
};

/// @overload
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, typename T, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(lambda&& qf, double t, const tensor<double, 2>& x_q, const T& arg_tuple,
                                       std::integer_sequence<int, i...>)
{
  tensor<double, 2> J_q{};
  return qf(t, serac::tuple{x_q, J_q}, serac::get<i>(arg_tuple)...);
}

/// @overload
SERAC_SUPPRESS_NVCC_HOSTDEVICE_WARNING
template <typename lambda, typename T, int... i>
SERAC_HOST_DEVICE auto apply_qf_helper(lambda&& qf, double t, const tensor<double, 3>& x_q, const T& arg_tuple,
                                       std::integer_sequence<int, i...>)
{
  constexpr int                dim = 3;
  tensor<double, dim, dim - 1> J_q{};
  return qf(t, serac::tuple{x_q, J_q}, serac::get<i>(arg_tuple)...);
}

/// @overload
template <typename lambda, typename coords_type, typename... T>
SERAC_HOST_DEVICE auto apply_qf(lambda&& qf, double t, coords_type&& x_q, const serac::tuple<T...>& arg_tuple)
{
  return apply_qf_helper(qf, t, x_q, arg_tuple, std::make_integer_sequence<int, static_cast<int>(sizeof...(T))>{});
}

template <int i, int dim, typename... trials, typename lambda>
auto get_derivative_type(lambda qf)
{
  using qf_arguments = serac::tuple<typename QFunctionArgument<trials, serac::Dimension<dim>>::type...>;
  return tuple{get_gradient(apply_qf(qf, double{}, tensor<double, dim + 1>{}, make_dual_wrt<i>(qf_arguments{}))),
               zero{}};
};

template <typename lambda, int n, typename... T>
auto batch_apply_qf(lambda qf, double t, const tensor<double, 2, n>& positions,
                    const tensor<double, 1, 2, n>& jacobians, const T&... inputs)
{
  constexpr int dim = 2;
  using first_arg_t = serac::tuple<tensor<double, dim>, tensor<double, dim>>;
  using return_type = decltype(qf(double{}, first_arg_t{}, T{}[0]...));
  tensor<tuple<return_type, zero>, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim> x_q;
    tensor<double, dim> J_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = positions(j, i);
      J_q[j] = jacobians(0, j, i);
    }
    double scale = norm(cross(J_q));

    get<0>(outputs[i]) = qf(t, serac::tuple{x_q, J_q}, inputs[i]...) * scale;
  }
  return outputs;
}

template <typename lambda, int n, typename... T>
auto batch_apply_qf(lambda qf, double t, const tensor<double, 3, n>& positions,
                    const tensor<double, 2, 3, n>& jacobians, const T&... inputs)
{
  constexpr int dim = 3;
  using first_arg_t = serac::tuple<tensor<double, dim>, tensor<double, dim, dim - 1>>;
  using return_type = decltype(qf(double{}, first_arg_t{}, T{}[0]...));
  tensor<tuple<return_type, zero>, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor<double, dim>          x_q;
    tensor<double, dim, dim - 1> J_q;
    for (int j = 0; j < dim; j++) {
      x_q[j] = positions(j, i);
      for (int k = 0; k < dim - 1; k++) {
        J_q(j, k) = jacobians(k, j, i);
      }
    }
    double scale = norm(cross(J_q));

    get<0>(outputs[i]) = qf(t, serac::tuple{x_q, J_q}, inputs[i]...) * scale;
  }
  return outputs;
}

/// @trial_elements the element type for each trial space
template <uint32_t differentiation_index, int Q, mfem::Geometry::Type geom, typename test_element,
          typename trial_element_type, typename lambda_type, typename derivative_type, int... indices>
void evaluation_kernel_impl(trial_element_type trial_elements, test_element, double t,
                            const std::vector<const double*>& inputs, double* outputs, const double* positions,
                            const double* jacobians, lambda_type qf, [[maybe_unused]] derivative_type* qf_derivatives,
                            const int* elements, uint32_t num_elements, camp::int_seq<int, indices...>)
{
  // mfem provides this information as opaque arrays of doubles,
  // so we reinterpret the pointer with
  constexpr int dim                          = dimension_of(geom) + 1;
  constexpr int nqp                          = num_quadrature_points(geom, Q);
  using X_Type                               = const tensor<double, dim, nqp>;
  using J_Type                               = const tensor<double, dim - 1, dim, nqp>;
  auto                                     J = reinterpret_cast<J_Type*>(jacobians);
  auto                                     x = reinterpret_cast<X_Type*>(positions);
  auto                                     r = reinterpret_cast<typename test_element::dof_type*>(outputs);
  constexpr TensorProductQuadratureRule<Q> rule{};

  constexpr int qpts_per_elem = num_quadrature_points(geom, Q);

  [[maybe_unused]] tuple u = {
      reinterpret_cast<const typename decltype(type<indices>(trial_elements))::dof_type*>(inputs[indices])...};

  using interpolate_out_type = decltype(tuple{get<indices>(trial_elements).template interpolate_output_helper<Q>()...});

  using qf_inputs_type = decltype(tuple{promote_each_to_dual_when<indices == differentiation_index>(
      get<indices>(trial_elements).template interpolate_output_helper<Q>())...});

#ifdef USE_CUDA
  auto&           rm             = umpire::ResourceManager::getInstance();
  auto            dest_allocator = rm.getAllocator("DEVICE");
  qf_inputs_type* qf_inputs =
      static_cast<qf_inputs_type*>(dest_allocator.allocate(sizeof(qf_inputs_type) * num_elements));
  interpolate_out_type* interpolate_result =
      static_cast<interpolate_out_type*>(dest_allocator.allocate(sizeof(interpolate_out_type) * num_elements));

  auto device_r = copy_data(r, serac::size(*r) * sizeof(double), "DEVICE");

  printCUDAMemUsage();
#else
  auto&           rm             = umpire::ResourceManager::getInstance();
  auto            host_allocator = rm.getAllocator("HOST");
  qf_inputs_type* qf_inputs =
      static_cast<qf_inputs_type*>(host_allocator.allocate(sizeof(qf_inputs_type) * num_elements));
  interpolate_out_type* interpolate_result =
      static_cast<interpolate_out_type*>(host_allocator.allocate(sizeof(interpolate_out_type) * num_elements));

  typename test_element::dof_type* device_r = r;
#endif
  rm.memset(qf_inputs, 0);
  rm.memset(interpolate_result, 0);

  auto e_range = RAJA::TypedRangeSegment<uint32_t>(0, num_elements);
#if defined(USE_CUDA)
  using threads_x [[maybe_unused]] = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
  using teams_e                    = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
  using launch_policy              = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
#else
  using threads_x [[maybe_unused]]          = RAJA::LoopPolicy<RAJA::seq_exec>;
  using teams_e                             = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy                       = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
#endif

  // for each element in the domain
  RAJA::launch<launch_policy>(
      RAJA::LaunchParams(RAJA::Teams(num_elements), RAJA::Threads(BLOCK_SZ)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<teams_e>(
            ctx, e_range,
            [&ctx, t, J, x, u, qf, trial_elements, qpts_per_elem, rule, device_r, elements, qf_derivatives, qf_inputs,
             interpolate_result](uint32_t e) {
              // batch-calculate values / derivatives of each trial space, at each quadrature point
              (get<indices>(trial_elements)
                   .interpolate(get<indices>(u)[elements[e]], rule, &get<indices>(interpolate_result[e]), ctx),
               ...);

              ctx.teamSync();

              (promote_each_to_dual_when<indices == differentiation_index>(get<indices>(interpolate_result[e]),
                                                                           &get<indices>(qf_inputs[e]), ctx),
               ...);

              ctx.teamSync();

              // (batch) evalute the q-function at each quadrature point
              auto qf_outputs = batch_apply_qf(qf, t, x[e], J[e], get<indices>(qf_inputs[e])...);

              ctx.teamSync();

              // write out the q-function derivatives after applying the
              // physical_to_parent transformation, so that those transformations
              // won't need to be applied in the action_of_gradient and element_gradient kernels
              if constexpr (differentiation_index != serac::NO_DIFFERENTIATION) {
                RAJA::RangeSegment x_range(0, leading_dimension(qf_outputs));
                RAJA::loop<threads_x>(ctx, x_range, [&](int q) {
                  qf_derivatives[e * qpts_per_elem + uint32_t(q)] = get_gradient(qf_outputs[q]);
                });
              }

              ctx.teamSync();

              // (batch) integrate the material response against the test-space basis functions
              test_element::integrate(get_value(qf_outputs), rule, &device_r[elements[e]], ctx);
            });
      });
#ifdef USE_CUDA
  rm.copy(r, device_r);
  deallocate(qf_inputs, "DEVICE");
  deallocate(interpolate_result, "DEVICE");
  deallocate(device_r, "DEVICE");
  printCUDAMemUsage();
#else
  host_allocator.deallocate(interpolate_result);
  host_allocator.deallocate(qf_inputs);
#endif
}

//clang-format off
template <typename S, typename T>
SERAC_HOST_DEVICE auto chain_rule(const S& dfdx, const T& dx)
{
  return serac::chain_rule(serac::get<0>(serac::get<0>(dfdx)), serac::get<0>(dx)) +
         serac::chain_rule(serac::get<1>(serac::get<0>(dfdx)), serac::get<1>(dx));
}
//clang-format on

template <typename derivative_type, int n, typename T>
SERAC_HOST_DEVICE auto batch_apply_chain_rule(derivative_type* qf_derivatives, const tensor<T, n>& inputs,
                                              const RAJA::LaunchContext& ctx)
{
  using return_type = decltype(chain_rule(derivative_type{}, T{}));
  tensor<tuple<return_type, zero>, n> outputs{};
  for (int i = 0; i < n; i++) {
    get<0>(outputs[i]) = chain_rule(qf_derivatives[i], inputs[i]);
  }
  return outputs;
}

/**
 * @brief The base kernel template used to create create custom directional derivative
 * kernels associated with finite element calculations
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p BoundaryIntegral
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
template <int Q, mfem::Geometry::Type geom, typename test, typename trial, typename derivatives_type>
void action_of_gradient_kernel(const double* dU, double* dR, derivatives_type* qf_derivatives, const int* elements,
                               std::size_t num_elements)
{
  using test_element  = finite_element<geom, test>;
  using trial_element = finite_element<geom, trial>;

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  constexpr int                            nqp = num_quadrature_points(geom, Q);
  auto                                     du  = reinterpret_cast<const typename trial_element::dof_type*>(dU);
  auto                                     dr  = reinterpret_cast<typename test_element::dof_type*>(dR);
  constexpr TensorProductQuadratureRule<Q> rule{};

  auto          e_range = RAJA::RangeSegment(0, num_elements);
  trial_element empty_trial_element{};

  using qf_inputs_type = decltype(trial_element::template interpolate_output_helper<Q>());

#ifdef USE_CUDA
  auto&           rm             = umpire::ResourceManager::getInstance();
  auto            dest_allocator = rm.getAllocator("DEVICE");
  qf_inputs_type* qf_inputs =
      static_cast<qf_inputs_type*>(dest_allocator.allocate(sizeof(qf_inputs_type) * num_elements));
#else
  auto&           rm             = umpire::ResourceManager::getInstance();
  auto            host_allocator = rm.getAllocator("HOST");
  qf_inputs_type* qf_inputs =
      static_cast<qf_inputs_type*>(host_allocator.allocate(sizeof(qf_inputs_type) * num_elements));
#endif
  rm.memset(qf_inputs, 0);

#if defined(USE_CUDA)
  using teams_e       = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
#else
  using teams_e                    = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy              = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
#endif

  // for each element in the domain
  RAJA::launch<launch_policy>(
      RAJA::LaunchParams(RAJA::Teams(num_elements), RAJA::Threads(BLOCK_X, BLOCK_Y, BLOCK_Z)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<teams_e>(ctx, e_range, [du, rule, &ctx, qf_inputs, elements, qf_derivatives, dr](int e) {
          // (batch) interpolate each quadrature point's value
          trial_element::interpolate(du[elements[e]], rule, qf_inputs, ctx);

          // (batch) evalute the q-function at each quadrature point
          auto qf_outputs = batch_apply_chain_rule(qf_derivatives + e * nqp, *qf_inputs, ctx);

          // (batch) integrate the material response against the test-space basis functions
          test_element::integrate(qf_outputs, rule, &dr[elements[e]], ctx);
        });
      });
  rm.deallocate(qf_inputs);
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
void element_gradient_kernel(ExecArrayView<double, 3, ExecutionSpace::CPU> dK, derivatives_type* qf_derivatives,
                             const int* elements, std::size_t num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  constexpr bool is_QOI = test::family == Family::QOI;
  using padded_derivative_type [[maybe_unused]] =
      std::conditional_t<is_QOI, tuple<derivatives_type, zero>, derivatives_type>;

  RAJA::TypedRangeSegment<size_t>          elements_range(0, num_elements);
  constexpr int                            nquad = num_quadrature_points(g, Q);
  constexpr TensorProductQuadratureRule<Q> rule{};
#if defined(USE_CUDA)
  using teams_e       = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
  using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
#else
  using teams_e                    = RAJA::LoopPolicy<RAJA::seq_exec>;
  using launch_policy              = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
#endif
#ifdef USE_CUDA
  using threads_x [[maybe_unused]] = RAJA::LoopPolicy<RAJA::cuda_thread_x_direct>;
#else
  using threads_x [[maybe_unused]] = RAJA::LoopPolicy<RAJA::seq_exec>;
#endif

  // for each element in the domain
  RAJA::launch<launch_policy>(
      RAJA::LaunchParams(RAJA::Teams(num_elements), RAJA::Threads(BLOCK_SZ)),
      [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
        RAJA::loop<teams_e>(ctx, elements_range, [&ctx, dK, elements, qf_derivatives, nquad, rule](uint32_t e) {
          auto* output_ptr = reinterpret_cast<typename test_element::dof_type*>(&dK(elements[e], 0, 0));

          tensor<derivatives_type, nquad> derivatives{};
          RAJA::RangeSegment              x_range(0, nquad);
          RAJA::loop<threads_x>(ctx, x_range, [&](int q) { derivatives(q) = qf_derivatives[e * nquad + uint32_t(q)]; });

          ctx.teamSync();

          for (int J = 0; J < trial_element::ndof; J++) {
            RAJA_TEAM_SHARED decltype(trial_element::batch_apply_shape_fn(J, derivatives, rule, ctx)) source_and_flux;
            source_and_flux = trial_element::batch_apply_shape_fn(J, derivatives, rule, ctx);
            test_element::integrate(source_and_flux, rule, output_ptr + J, ctx, trial_element::ndof);
          }
          ctx.teamSync();
        });
      });
}

template <uint32_t wrt, int Q, mfem::Geometry::Type geom, typename signature, typename lambda_type,
          typename derivative_type>
auto evaluation_kernel(signature s, lambda_type qf, const double* positions, const double* jacobians,
                       std::shared_ptr<derivative_type> qf_derivatives, const int* elements, uint32_t num_elements)
{
  auto trial_elements = trial_elements_tuple<geom>(s);
  auto test_element   = get_test_element<geom>(s);
  return [=](double time, const std::vector<const double*>& inputs, double* outputs, bool /* update state */) {
    evaluation_kernel_impl<wrt, Q, geom>(trial_elements, test_element, time, inputs, outputs, positions, jacobians, qf,
                                         qf_derivatives.get(), elements, num_elements, s.index_seq);
  };
}

template <int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type>
std::function<void(const double*, double*)> jacobian_vector_product_kernel(
    signature, std::shared_ptr<derivative_type> qf_derivatives, const int* elements, uint32_t num_elements)
{
  return [=](const double* du, double* dr) {
    using test_space  = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types>::type;
    action_of_gradient_kernel<Q, geom, test_space, trial_space>(du, dr, qf_derivatives.get(), elements, num_elements);
  };
}

template <int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type>
std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)> element_gradient_kernel(
    signature, std::shared_ptr<derivative_type> qf_derivatives, const int* elements, uint32_t num_elements)
{
  return [=](ExecArrayView<double, 3, ExecutionSpace::CPU> K_elem) {
    using test_space  = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types>::type;
    element_gradient_kernel<geom, test_space, trial_space, Q>(K_elem, qf_derivatives.get(), elements, num_elements);
  };
}

}  // namespace boundary_integral

}  // namespace serac
