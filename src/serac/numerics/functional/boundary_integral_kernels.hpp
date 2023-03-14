// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <array>

#include "serac/numerics/quadrature_data.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"
#include "serac/numerics/functional/evector_view.hpp"

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
template <int p, int dim>
struct QFunctionArgument<H1<p, 1>, Dimension<dim>> {
  using type = serac::tuple<double, tensor<double, dim>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<H1<p, c>, Dimension<dim>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c, dim>>;  ///< what will be passed to the q-function
};

template <int p, int dim>
struct QFunctionArgument<L2<p, 1>, Dimension<dim>> {
  using type = serac::tuple<double, tensor<double, dim>>;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<L2<p, c>, Dimension<dim>> {
  using type = serac::tuple<tensor<double, c>, tensor<double, c, dim>>;  ///< what will be passed to the q-function
};

template <int i, int dim, typename... trials, typename lambda>
auto get_derivative_type(lambda qf)
{
  using qf_arguments = serac::tuple<typename QFunctionArgument<trials, serac::Dimension<dim>>::type...>;
  return tuple{get_gradient(detail::apply_qf(qf, tensor<double, dim + 1>{}, tensor<double, dim + 1>{},
                                             make_dual_wrt<i>(qf_arguments{}))),
               zero{}};
};

template <int i>
struct DerivativeWRT {
};

template <int Q, mfem::Geometry::Type g, typename test, typename... trials>
struct KernelConfig {
};

template <typename lambda, int dim, int n, typename... T>
auto batch_apply_qf(lambda qf, const tensor<double, dim, n>& positions,
                    const tensor<double, dim - 1, dim, n>& jacobians, const T & ... inputs)
{
  using return_type = decltype(qf(tensor<double, dim>{}, tensor<double, dim>{}, T{}[0]...));
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
    tensor<double, dim> n_q = cross(J_q);

    double scale = norm(n_q);

    get<0>(outputs[i]) = qf(x_q, n_q / scale, inputs[i] ...) * scale;
  }
  return outputs;
}

#if 0
/**
 * @tparam S type used to specify which argument to differentiate with respect to.
 *    `void` => evaluation kernel with no differentiation
 *    `DerivativeWRT<i>` => evaluation kernel with AD applied to trial space `i`
 * @tparam T the "function signature" of the form `test(trial0, trial1, ...)`
 * @tparam derivatives_type the type of the derivative of the q-function
 * @tparam lambda the type of the q-function
 *
 * @brief Functor type providing a callback for the evaluation of the user-specified q-function over the domain
 */
template <typename S, typename T, typename derivatives_type, typename lambda, typename int_seq>
struct EvaluationKernel;

/**
 * @overload
 * @note evaluation kernel with no differentiation
 */

template <int Q, mfem::Geometry::Type geom, typename test, typename... trials, typename lambda, int... int_seq>
struct EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda,
                        std::integer_sequence<int, int_seq...>> {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided

  /// @brief an integer sequence used to iterate through the trial spaces that appear in this kernel
  static constexpr auto Iseq = std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>{};

  /// @brief the element type for the test space
  using test_element = finite_element<geom, test>;

  /// @brief the element type for each trial space
  static constexpr tuple<finite_element<geom, trials>...> trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   */
  EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector& J, const mfem::Vector& X,
                   std::size_t num_elements, lambda qf)
      : J_(J), X_(X), num_elements_(num_elements), qf_(qf)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R)
  {
    static constexpr int sdim = dimension_of(geom) + 1;  // spatial dimension

    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    constexpr int nqp = num_quadrature_points(geom, Q);
    auto          J   = reinterpret_cast<const tensor<double, sdim - 1, sdim, nqp>*>(J_.Read());
    auto          X   = reinterpret_cast<const tensor<double, sdim, nqp>*>(X_.Read());
    auto          r   = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    static constexpr TensorProductQuadratureRule<Q> rule{};

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {
      // load the jacobians, positions and normals for each quadrature point in this element
      auto J_e = J[e];
      auto X_e = X[e];

      tuple<decltype(
          finite_element<geom, trials>::interpolate(typename finite_element<geom, trials>::dof_type{}, rule))...>
          qf_inputs{};

      for_constexpr<num_trial_spaces>([&](auto j) {
        using trial_element = decltype(type<j>(trial_elements));

        auto u = reinterpret_cast<const typename trial_element::dof_type*>(U[j]->Read());

        // (batch) interpolate each quadrature point's value
        get<j>(qf_inputs) = trial_element::interpolate(u[e], rule);
      });

      // (batch) evalute the q-function at each quadrature point
      auto qf_outputs = batch_apply_qf(qf_, X_e, J_e, qf_inputs, Iseq);

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(qf_outputs, rule, &r[e]);
    }
  }

  const mfem::Vector& J_;             ///< values of sqrt(det(J^T * J)) at each quadrature point
  const mfem::Vector& X_;             ///< Spatial positions of each quadrature point
  std::size_t         num_elements_;  ///< how many elements in the domain
  lambda              qf_;            ///< q-function
};

/**
 * @overload
 * @note evaluation kernel that also calculates derivative w.r.t. `I`th trial space
 */
template <int differentiation_index, int Q, mfem::Geometry::Type geom, typename test, typename... trials, typename derivatives_type,
          typename lambda, int... indices>
struct EvaluationKernel<DerivativeWRT<differentiation_index>, KernelConfig<Q, geom, test, trials...>, derivatives_type,
                        lambda, std::integer_sequence<int, indices...>> {
  static constexpr auto exec             = ExecutionSpace::CPU;  ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = static_cast<int>(sizeof...(trials));  ///< how many trial spaces are provided

  /// @brief an integer sequence used to iterate through the trial spaces that appear in this kernel
  static constexpr auto Iseq = std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>{};

  /// @brief the element type for the test space
  using test_element = finite_element<geom, test>;

  /// @brief the element type for each trial space
  static constexpr tuple<finite_element<geom, trials>...> trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param qf_derivatives a container for the derivatives of the q-function w.r.t. trial space I
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   */
  EvaluationKernel(DerivativeWRT<differentiation_index>, KernelConfig<Q, geom, test, trials...>,
                   CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J, const mfem::Vector& X,
                   std::size_t num_elements, lambda qf)
      : qf_derivatives_(qf_derivatives), J_(J), X_(X), num_elements_(num_elements), qf_(qf)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R)
  {
    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    constexpr int sdim = dimension_of(geom) + 1;  // spatial dimension
    constexpr int nqp  = num_quadrature_points(geom, Q);
    auto          J    = reinterpret_cast<const tensor<double, sdim - 1, sdim, nqp>*>(J_.Read());
    auto          X    = reinterpret_cast<const tensor<double, sdim, nqp>*>(X_.Read());
    auto          r    = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    static constexpr TensorProductQuadratureRule<Q> rule{};

    tuple u_e = {
        reinterpret_cast<const typename decltype(type<indices>(trial_elements))::dof_type*>(U[indices]->Read())...};

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {
      // load the jacobians, positions and normals for each quadrature point in this element
      auto J_e = J[e];
      auto X_e = X[e];

      // batch-calculate values / derivatives of each trial space, at each quadrature point
      tuple qf_inputs = {promote_each_to_dual_when<indices == differentiation_index>(
          type<indices>(trial_elements).interpolate(get<indices>(u_e)[e], rule))...};

      // (batch) evalute the q-function at each quadrature point
      auto qf_outputs = batch_apply_qf(qf_, X_e, J_e, qf_inputs, Iseq);

      // write out the q-function derivatives after multiplying by J_e so that
      // won't need to be applied in the action_of_gradient and element_gradient kernels
      for (int q = 0; q < leading_dimension(qf_outputs); q++) {
        qf_derivatives_(e, q) = get_gradient(qf_outputs[q]);
      }

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(get_value(qf_outputs), rule, &r[e]);
    }
  }

  ExecArrayView<derivatives_type, 2, exec> qf_derivatives_;  ///< derivatives of the q-function w.r.t. trial space `I`
  const mfem::Vector&                      J_;               ///< values of sqrt(det(J^T * J)) at each quadrature point
  const mfem::Vector&                      X_;               ///< Spatial positions of each quadrature point
  std::size_t                              num_elements_;    ///< how many elements in the domain
  lambda                                   qf_;              ///< q-function
};

template <int Q, mfem::Geometry::Type geom, typename test, typename... trials, typename lambda>
EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector&, const mfem::Vector&, int, lambda)
    -> EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda,
                        std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>>;

template <int differentiation_index, int Q, mfem::Geometry::Type geom, typename test, typename... trials, typename derivatives_type,
          typename lambda>
EvaluationKernel(DerivativeWRT<differentiation_index>, KernelConfig<Q, geom, test, trials...>,
                 CPUArrayView<derivatives_type, 2>, const mfem::Vector&, const mfem::Vector&, int, lambda)
    -> EvaluationKernel<DerivativeWRT<differentiation_index>, KernelConfig<Q, geom, test, trials...>, derivatives_type,
                        lambda, std::make_integer_sequence<int, static_cast<int>(sizeof...(trials))>>;
                        
#endif


template <int differentiation_index, int Q, mfem::Geometry::Type geom, typename test, typename... trials, typename lambda_type,
          typename derivative_type, int... indices>
void evaluation_kernel_impl(FunctionSignature<test(trials...)>, const std::vector<const double*>& inputs,
                            double* outputs, const double* positions, const double* jacobians, lambda_type qf,
                            derivative_type * qf_derivatives,
                            uint32_t num_elements, std::integer_sequence<int, indices...>)
{
  using test_element = finite_element<geom, test>;

  /// @brief the element type for each trial space
  static constexpr tuple<finite_element<geom, trials>...> trial_elements{};

  // mfem provides this information as opaque arrays of doubles,
  // so we reinterpret the pointer with
  constexpr int dim = dimension_of(geom) + 1;
  constexpr int nqp = num_quadrature_points(geom, Q);
  auto J    = reinterpret_cast<const tensor<double, dim - 1, dim, nqp>*>(jacobians);
  auto x    = reinterpret_cast<const tensor<double, dim, nqp>*>(positions);
  auto r    = reinterpret_cast<typename test_element::dof_type*>(outputs);
  static constexpr TensorProductQuadratureRule<Q> rule{};

  static constexpr int qpts_per_elem = num_quadrature_points(geom, Q);

  tuple u = {reinterpret_cast<const typename decltype(type<indices>(trial_elements))::dof_type*>(inputs[indices])...};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    // load the jacobians and positions for each quadrature point in this element
    auto J_e = J[e];
    auto x_e = x[e];

    // batch-calculate values / derivatives of each trial space, at each quadrature point
    [[maybe_unused]] tuple qf_inputs = {promote_each_to_dual_when<indices == differentiation_index>(
        get<indices>(trial_elements).interpolate(get<indices>(u)[e], rule))...};

    // (batch) evalute the q-function at each quadrature point
    //
    // note: the weird immediately-invoked lambda expression is
    // a workaround for a bug in GCC(<12.0) where it fails to
    // decide which function overload to use, and crashes
    auto qf_outputs = batch_apply_qf(qf, x_e, J_e, get<indices>(qf_inputs)...);

    // write out the q-function derivatives after applying the
    // physical_to_parent transformation, so that those transformations
    // won't need to be applied in the action_of_gradient and element_gradient kernels
    if constexpr (differentiation_index != -1) {
      for (int q = 0; q < leading_dimension(qf_outputs); q++) {
        qf_derivatives[e * qpts_per_elem + uint32_t(q)] = get_gradient(qf_outputs[q]);
      }
    }

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(get_value(qf_outputs), rule, &r[e]);
  }
}

//clang-format off
template <typename S, typename T>
auto chain_rule(const S& dfdx, const T& dx)
{
  return serac::chain_rule(serac::get<0>(serac::get<0>(dfdx)), serac::get<0>(dx)) +
         serac::chain_rule(serac::get<1>(serac::get<0>(dfdx)), serac::get<1>(dx));
}
//clang-format on

template <typename derivative_type, int n, typename T>
auto batch_apply_chain_rule(derivative_type* qf_derivatives, const tensor<T, n>& inputs)
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
 * @tparam Q Quadrature parameter describing how many points per dimension
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
template < int Q, mfem::Geometry::Type geom, typename test, typename trial, typename derivatives_type>
void action_of_gradient_kernel(const double * dU, double * dR,
                               derivatives_type * qf_derivatives, std::size_t num_elements)
{
  using test_element  = finite_element<geom, test>;
  using trial_element = finite_element<geom, trial>;

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  constexpr int nqp = num_quadrature_points(geom, Q);
  auto du = reinterpret_cast<const typename trial_element::dof_type*>(dU);
  auto dr = reinterpret_cast<typename test_element::dof_type*>(dR);
  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    // (batch) interpolate each quadrature point's value
    auto qf_inputs = trial_element::interpolate(du[e], rule);

    // (batch) evalute the q-function at each quadrature point
    auto qf_outputs = batch_apply_chain_rule(qf_derivatives + e * nqp, qf_inputs);

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
 * @tparam Q Quadrature parameter describing how many points per dimension
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
void element_gradient_kernel(ExecArrayView<double, 3, ExecutionSpace::CPU> dK,
                             derivatives_type * qf_derivatives, std::size_t num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  constexpr int nquad = num_quadrature_points(g, Q);

  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    auto* output_ptr = reinterpret_cast<typename test_element::dof_type*>(&dK(e, 0, 0));

    tensor<derivatives_type, nquad> derivatives{};
    for (int q = 0; q < nquad; q++) {
      derivatives(q) = qf_derivatives[e * nquad + q];
    }

    for (int J = 0; J < trial_element::ndof; J++) {
      auto source_and_flux = trial_element::batch_apply_shape_fn(J, derivatives, rule);
      test_element::integrate(source_and_flux, rule, output_ptr + J, trial_element::ndof);
    }
  }
}


template <int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename lambda_type,
          typename derivative_type>
std::function<void(const std::vector<const double*>&, double*, bool)> evaluation_kernel(
    signature s, lambda_type qf, const double* positions, const double* jacobians, std::shared_ptr<derivative_type> qf_derivatives,
    uint32_t num_elements)
{
  return [=](const std::vector<const double*>& inputs, double* outputs, bool /* update state */) {
    evaluation_kernel_impl<wrt, Q, geom>(s, inputs, outputs, positions, jacobians, qf, 
                                                     qf_derivatives.get(), num_elements, s.index_seq);
  };
}

template < int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type >
std::function<void(const double*, double*)> jvp_kernel(signature, 
    std::shared_ptr<derivative_type> qf_derivatives, uint32_t num_elements) {
  return [=](const double * du, double * dr){
    using test_space = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types >::type;
    action_of_gradient_kernel< Q, geom, test_space, trial_space >(du, dr, qf_derivatives.get(), num_elements);
  };
}

template < int wrt, int Q, mfem::Geometry::Type geom, typename signature, typename derivative_type >
std::function<void(ExecArrayView<double, 3, ExecutionSpace::CPU>)> element_gradient_kernel(signature,
  std::shared_ptr<derivative_type> qf_derivatives, uint32_t num_elements) {
  return [=](ExecArrayView<double, 3, ExecutionSpace::CPU> K_elem){
    using test_space = typename signature::return_type;
    using trial_space = typename std::tuple_element<wrt, typename signature::parameter_types >::type;
    element_gradient_kernel< geom, test_space, trial_space, Q >(K_elem, qf_derivatives.get(), num_elements);
  };
}

}  // namespace boundary_integral

}  // namespace serac
