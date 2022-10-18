// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/quadrature_data.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"
#include "serac/numerics/functional/evector_view.hpp"

#include <array>

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
struct QFunctionArgument;

/// @overload
template <int p, int dim>
struct QFunctionArgument<H1<p, 1>, Dimension<dim> > {
  using type = tuple<double, tensor<double, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<H1<p, c>, Dimension<dim> > {
  using type = tuple<tensor<double, c>, tensor<double, c, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p, int dim>
struct QFunctionArgument<L2<p, 1>, Dimension<dim> > {
  using type = tuple<double, tensor<double, dim> >;  ///< what will be passed to the q-function
};
/// @overload
template <int p, int c, int dim>
struct QFunctionArgument<L2<p, c>, Dimension<dim> > {
  using type = tuple<tensor<double, c>, tensor<double, c, dim> >;  ///< what will be passed to the q-function
};

/// @overload
template <int p>
struct QFunctionArgument<Hcurl<p>, Dimension<2> > {
  using type = tuple<tensor<double, 2>, double>;  ///< what will be passed to the q-function
};

/// @overload
template <int p>
struct QFunctionArgument<Hcurl<p>, Dimension<3> > {
  using type = tuple<tensor<double, 3>, tensor<double, 3> >;  ///< what will be passed to the q-function
};

template <int i, int dim, typename... trials, typename lambda, typename qpt_data_type>
auto get_derivative_type(lambda qf, qpt_data_type&& qpt_data)
{
  using qf_arguments = serac::tuple<typename QFunctionArgument<trials, serac::Dimension<dim> >::type...>;
  return get_gradient(detail::apply_qf(qf, tensor<double, dim>{}, qpt_data, make_dual_wrt<i>(qf_arguments{})));
};

template <int i>
struct DerivativeWRT {
};

template <int Q, Geometry g, typename test, typename... trials>
struct KernelConfig {
};

template <typename lambda, int dim, int n, typename ... T, int ... I >
auto batch_apply_qf(lambda qf, const tensor< double, dim, n > x, Nothing * /*qpt_data*/ , const tuple < tensor<T, n> ... > inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, T{} ...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    for (int j = 0; j < dim; j++) { 
      x_q[j] = x(j, i); 
    }
    outputs[i] = qf(x_q, get<I>(inputs)[i] ...);
  }
  return outputs;
}

template <typename lambda, int dim, int n, typename qpt_data_type, typename ... T, int ... I >
auto batch_apply_qf(lambda qf, const tensor< double, dim, n > x, qpt_data_type * qpt_data, const tuple < tensor<T, n> ... > inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, qpt_data[0], T{} ...));
  tensor<return_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    for (int j = 0; j < dim; j++) { x_q[j] = x(j, i); }

    auto qdata = qpt_data[i];
    outputs[i] = qf(x_q, qdata, get<I>(inputs)[i] ...);
  }
  return outputs;
}

template <int index_to_differentiate, typename derivative_type, typename lambda, int dim, int n, typename ... T, int ... I >
auto batch_apply_qf_with_AD(derivative_type * qf_derivatives, lambda qf, const tensor< double, dim, n > x, Nothing *, const tuple < tensor<T, n> ... > inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, promote_to_dual_when< I == index_to_differentiate >(T{}) ...));
  using value_type = decltype(get_value(return_type{}));
  tensor<value_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    for (int j = 0; j < dim; j++) { x_q[j] = x(j, i); }
    auto outputs_and_derivatives = qf(x_q, promote_to_dual_when< I == index_to_differentiate >(get<I>(inputs)[i]) ...);
    outputs[i] = get_value(outputs_and_derivatives);
    qf_derivatives[i] = get_gradient(outputs_and_derivatives);
  }
  return outputs;
}

template <int index_to_differentiate, typename derivative_type, typename lambda, int dim, int n, typename qpt_data_type, typename ... T, int ... I >
auto batch_apply_qf_with_AD(derivative_type * qf_derivatives, lambda qf, const tensor< double, dim, n > x, qpt_data_type * qpt_data, const tuple < tensor<T, n> ... > inputs, std::integer_sequence< int, I ... >)
{
  using return_type = decltype(qf(tensor<double,dim>{}, qpt_data[0], promote_to_dual_when< I == index_to_differentiate >(T{}) ...));
  using value_type = decltype(get_value(return_type{}));
  tensor<value_type, n> outputs{};
  for (int i = 0; i < n; i++) {
    tensor< double, dim > x_q;
    for (int j = 0; j < dim; j++) { 
      x_q[j] = x(j, i); 
    }
    auto qdata = qpt_data[i];
    auto outputs_and_derivatives = qf(x_q, qdata, promote_to_dual_when< I == index_to_differentiate >(get<I>(inputs)[i]) ...);
    outputs[i] = get_value(outputs_and_derivatives);
    qf_derivatives[i] = get_gradient(outputs_and_derivatives);
  }
  return outputs;
}

/**
 * @tparam S type used to specify which argument to differentiate with respect to.
 *    `void` => evaluation kernel with no differentiation
 *    `DerivativeWRT<i>` => evaluation kernel with AD applied to trial space `i`
 * @tparam T a configuration argument containing:
 *    quadrature rule information,
 *    element geometry
 *    function signature of the form `test(trial0, trial1, ...)`
 * @tparam derivatives_type the type of the derivative of the q-function
 * @tparam lambda the type of the q-function
 *
 * @brief Functor type providing a callback for the evaluation of the user-specified q-function over the domain
 */
template <typename S, typename T, typename derivatives_type, typename lambda, typename qpt_data_type, typename int_seq >
struct EvaluationKernel;

/**
 * @overload
 * @note evaluation kernel with no differentiation
 */
template <int Q, Geometry geom, typename test, typename... trials, typename lambda, typename qpt_data_type, int ... int_seq >
struct EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda, qpt_data_type, std::integer_sequence < int, int_seq ... > > {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided
  static constexpr auto Iseq = std::make_integer_sequence<int, sizeof ... (trials)>{};

  using test_element = finite_element<geom, test>;
  static constexpr type_list < finite_element< geom, trials > ... > trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   * @param data user-specified quadrature data to pass to the q-function
   */
  EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector& J, const mfem::Vector& X,
                   std::size_t num_elements, lambda qf, std::shared_ptr<QuadratureData<qpt_data_type> > data)
      : J_(J), X_(X), num_elements_(num_elements), qf_(qf), data_(data)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   * @param update_state whether or not to overwrite material state quadrature data
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R, bool /*update_state*/)
  {

    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    auto X = reinterpret_cast<const typename batched_position<geom, Q>::type*>(X_.Read());
    auto r = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    auto J = reinterpret_cast<const typename batched_jacobian<geom, Q>::type*>(J_.Read());
    static constexpr TensorProductQuadratureRule<Q> rule{};

    auto& qdata = *data_;

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {

      // load the jacobians and positions for each quadrature point in this element
      auto J_e = J[e];
      auto X_e = X[e];

      tuple < 
        decltype(finite_element< geom, trials >::interpolate(typename finite_element< geom, trials >::dof_type{}, rule)) ... 
      > qf_inputs{};

      for_constexpr< num_trial_spaces >([&](auto j){

        using trial_element = decltype(trial_elements[j]);
        
        auto u = reinterpret_cast<const typename trial_element::dof_type*>(U[j]->Read());

        // (batch) interpolate each quadrature point's value
        get<j>(qf_inputs) = trial_element::interpolate(u[e], rule);

        // use J to transform values / derivatives on the parent element 
        // to the corresponding values / derivatives on the physical element 
        parent_to_physical< trial_element::family >(get<j>(qf_inputs), J_e);

      });

      // (batch) evalute the q-function at each quadrature point
      auto qf_outputs = batch_apply_qf(qf_, X_e, &qdata(e, 0), qf_inputs, Iseq);

      // use J to transform sources / fluxes on the physical element 
      // back to the corresponding sources / fluxes on the parent element 
      physical_to_parent< test_element::family >(qf_outputs, J_e);

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(qf_outputs, rule, &r[e]);

    }

  }

  const mfem::Vector&                             J_;             ///< Jacobian matrix entries at each quadrature point
  const mfem::Vector&                             X_;             ///< Spatial positions of each quadrature point
  std::size_t                                     num_elements_;  ///< how many elements in the domain
  lambda                                          qf_;            ///< q-function

  // TODO: address sanitizer is warning me that data_ is potentially double-freeing
  std::shared_ptr<QuadratureData<qpt_data_type> > data_;          ///< (optional) user-provided quadrature data
};

/**
 * @overload
 * @note evaluation kernel that also calculates derivative w.r.t. `I`th trial space
 */
template <int I, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda,
          typename qpt_data_type, int ... j>
struct EvaluationKernel<DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda,
                        qpt_data_type, std::integer_sequence< int , j ... > > {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided
  static constexpr auto Iseq = std::make_integer_sequence<int, sizeof ... (trials)>{};

  using test_element = finite_element<geom, test>;
  static constexpr tuple < finite_element< geom, trials > ... > trial_elements{};

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param qf_derivatives a container for the derivatives of the q-function w.r.t. trial space I
   * @param J values of the jacobian matrix at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   * @param data user-specified quadrature data to pass to the q-function
   */
  EvaluationKernel(DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>,
                   CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J, const mfem::Vector& X,
                   std::size_t num_elements, lambda qf, std::shared_ptr<QuadratureData<qpt_data_type> > data)
      : qf_derivatives_(qf_derivatives), J_(J), X_(X), num_elements_(num_elements), qf_(qf), data_(data)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   * @param update_state whether or not to overwrite material state quadrature data
   */
  void operator()(const std::vector<const mfem::Vector*> U, mfem::Vector& R, bool /*update_state*/)
  {

    // mfem provides this information as opaque arrays of doubles, 
    // so we reinterpret the pointer with 
    auto X = reinterpret_cast<const typename batched_position<geom, Q>::type*>(X_.Read());
    auto r = reinterpret_cast<typename test_element::dof_type*>(R.ReadWrite());
    auto J = reinterpret_cast<const typename batched_jacobian<geom, Q>::type*>(J_.Read());
    static constexpr TensorProductQuadratureRule<Q> rule{};

    auto& qdata = *data_;

    tuple u_e = {
      reinterpret_cast<const typename decltype(type<j>(trial_elements))::dof_type*>(U[j]->Read()) ...
    };

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {

      // load the jacobians and positions for each quadrature point in this element
      auto J_e = J[e];
      auto X_e = X[e];

      // batch-calculate values / derivatives of each trial space, at each quadrature point
      tuple qf_inputs = {
        promote_each_to_dual_when< j == I >(type<j>(trial_elements).interpolate(get<j>(u_e)[e], rule)) ...
      };

      // use J to transform values / derivatives on the parent element 
      // to the to the corresponding values / derivatives on the physical element 
      (parent_to_physical< type<j>(trial_elements).family >(get<j>(qf_inputs), J_e), ...);

      // (batch) evalute the q-function at each quadrature point
      //auto qf_outputs = batch_apply_qf_with_AD< I >(&qf_derivatives_(e, 0), qf_, X_e, &qdata(e, 0), qf_inputs, Iseq);
      auto qf_outputs = batch_apply_qf(qf_, X_e, &qdata(e, 0), qf_inputs, Iseq);

      // use J to transform sources / fluxes on the physical element 
      // back to the corresponding sources / fluxes on the parent element 
      physical_to_parent< test_element::family >(qf_outputs, J_e);

      // write out the q-function derivatives after applying the
      // physical_to_parent transformation, so that those transformations
      // won't need to be applied in the action_of_gradient and element_gradient kernels
      for (int q = 0; q < leading_dimension(qf_outputs); q++) {
        qf_derivatives_(e, q) = get_gradient(qf_outputs[q]);
      }

      // (batch) integrate the material response against the test-space basis functions
      test_element::integrate(get_value(qf_outputs), rule, &r[e]);

    }

  }

  ExecArrayView<derivatives_type, 2, exec> qf_derivatives_;  ///< derivatives of the q-function w.r.t. trial space `I`
  const mfem::Vector&                      J_;               ///< Jacobian matrix entries at each quadrature point
  const mfem::Vector&                      X_;               ///< Spatial positions of each quadrature point
  std::size_t                              num_elements_;    ///< how many elements in the domain
  lambda                                   qf_;              ///< q-function
  std::shared_ptr<QuadratureData<qpt_data_type> > data_;     ///< (optional) user-provided quadrature data
};

template <int Q, Geometry geom, typename test, typename... trials, typename lambda, typename qpt_data_type>
EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector&, const mfem::Vector&, int, lambda,
                 std::shared_ptr<QuadratureData<qpt_data_type> >)
    -> EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda, qpt_data_type, std::make_integer_sequence< int, sizeof ... (trials) > >;

template <int i, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda,
          typename qpt_data_type>
EvaluationKernel(DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, CPUArrayView<derivatives_type, 2>,
                 const mfem::Vector&, const mfem::Vector&, int, lambda, std::shared_ptr<QuadratureData<qpt_data_type> >)
    -> EvaluationKernel<DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda,
                        qpt_data_type, std::make_integer_sequence< int, sizeof ... (trials) > >;

//clang-format off
template <bool is_QOI, typename S, typename T>
auto chain_rule(const S& dfdx, const T& dx)
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

template <bool is_QOI, typename derivative_type, int n, typename T >
auto batch_apply_chain_rule(derivative_type * qf_derivatives, const tensor<T, n> & inputs)
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
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void action_of_gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR,
                               CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                               std::size_t num_elements)
{

  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;

  static constexpr bool is_QOI = (test::family == Family::QOI);

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  [[maybe_unused]] auto J = reinterpret_cast<const typename batched_jacobian<g, Q>::type*>(J_.Read());
  auto du = reinterpret_cast<const typename trial_element::dof_type*>(dU.Read());
  auto dr = reinterpret_cast<typename test_element::dof_type*>(dR.ReadWrite());
  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    // load the jacobians and positions for each quadrature point in this element
    //auto J_e = J[e];

    // (batch) interpolate each quadrature point's value
    auto qf_inputs = trial_element::interpolate(du[e], rule);

    // use J to transform values / derivatives on the parent element 
    // to the corresponding values / derivatives on the physical element 
    //parent_to_physical< trial_element::family >(qf_inputs, J_e);

    // (batch) evalute the q-function at each quadrature point
    auto qf_outputs = batch_apply_chain_rule<is_QOI>(&qf_derivatives(e, 0), qf_inputs);

    // use J to transform sources / fluxes on the physical element 
    // back to the corresponding sources / fluxes on the parent element 
    //physical_to_parent< test_element::family >(qf_outputs, J_e);

    // (batch) integrate the material response against the test-space basis functions
    test_element::integrate(qf_outputs, rule, &dr[e]);
  
  }

}

// K_e(i, j, k, l, m, n) := (B(u, i) * B(v, j) * B(w, k)) * X_q(u, v, w) * ((B(u, l) * B(v, m) * B(w, n))

// K_e(i, j, k, l, m, n) := {(B(u, i) * B(v, j) * B(w, k)) * 
//                              X_q(u, v, w, p, q) * 
//                           (B(u, l) * B(v, m) * B(w, n))
//
//  naive:
//  (B(u, i) * B(v, j) * B(w, k)) X_q(u, v, w) (B(u, l) * B(v, m) * B(w, n))
//  O(n^6 * q^3) 
//
//  B(u, i) * B(v, j) * B(w, k) X_q(u, v, w) * B(u, l) * B(v, m) * B(w, n)
//
//  X1(i,l,v,w) := (B(u, i) * B(u, l) * X_q(u, v, w))
//  O(n^2 * q^3) 
//
//  X2(i,l,m,n,w) := B(w, k) * B(w, n) * X1(i, l, v, w))
//  O(n^4 * q^2) 
//
//  K_e(i, j, k, l, m, n) := B(v, j) * B(v, m) * X2(i, l, m, n, w))
//  O(n^6 * q^3) 

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
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void element_gradient_kernel_new(ExecArrayView<double, 3, ExecutionSpace::CPU> dK,
                             CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector&,
                             std::size_t num_elements)
{
  using test_element        = finite_element<g, test>;
  using trial_element       = finite_element<g, trial>;

  constexpr int nquad = (g == Geometry::Quadrilateral) ? Q * Q : Q * Q * Q;

  static constexpr TensorProductQuadratureRule<Q> rule{};

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    auto * output_ptr = reinterpret_cast<typename test_element::dof_type *>(&dK(e, 0, 0));

    tensor< derivatives_type, nquad > derivatives{};
    for (int q = 0; q < nquad; q++) {
      derivatives(q) = qf_derivatives(e, q);
    }

    for (int J = 0; J < trial_element::ndof; J++) {
      auto source_and_flux = trial_element::batch_apply_shape_fn(J, derivatives, rule);
      test_element::integrate(source_and_flux, rule, output_ptr + J, trial_element::ndof);
    }
  }
}

template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void element_gradient_kernel(ExecArrayView<double, 3, ExecutionSpace::CPU> dk,
                             CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J_,
                             std::size_t num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  test_dim   = test_element::components;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr int  trial_dim  = trial_element::components;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    tensor<double, test_ndof, trial_ndof, test_dim, trial_dim> K_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi_q  = rule.points[q];
      auto   dxi_q = rule.weights[q];
      auto   J_q   = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx    = det(J_q) * dxi_q;

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q));

      if constexpr (std::is_same<test, QOI>::value) {
        auto& q0 = serac::get<0>(dq_darg);  // derivative of QoI w.r.t. field value
        auto& q1 = serac::get<1>(dq_darg);  // derivative of QoI w.r.t. field derivative

        auto N = evaluate_shape_functions<trial_element>(xi_q, J_q);

        for (int j = 0; j < trial_ndof; j++) {
          K_elem[0][j][0] += (q0 * N[j].value + q1 * N[j].derivative) * dx;
        }
      }

      if constexpr (!std::is_same<test, QOI>::value) {
        auto& q00 = serac::get<0>(serac::get<0>(dq_darg)); // derivative of source term w.r.t. field value
        auto& q01 = serac::get<1>(serac::get<0>(dq_darg)); // derivative of source term w.r.t. field derivative
        auto q10 = serac::get<0>(serac::get<1>(dq_darg));  // derivative of   flux term w.r.t. field value
        auto q11 = serac::get<1>(serac::get<1>(dq_darg));  // derivative of   flux term w.r.t. field derivative

        auto M = evaluate_shape_functions<test_element>(xi_q, J_q);
        auto N = evaluate_shape_functions<trial_element>(xi_q, J_q);

        // clang-format off
        for (int i = 0; i < test_ndof; i++) {
          for (int j = 0; j < trial_ndof; j++) {
            K_elem[i][j] += (
              M[i].value      * q00 * N[j].value +
              M[i].value      * q01 * N[j].derivative + 
              M[i].derivative * q10 * N[j].value +
              M[i].derivative * q11 * N[j].derivative
            ) * dx;
          } 
        }
        // clang-format on
      }
    }

    // once we've finished the element integration loop, write our element gradients
    // out to memory, to be later assembled into the global gradient by mfem
    // clang-format off
    if constexpr (std::is_same< test, QOI >::value) {
      for (int k = 0; k < trial_ndof; k++) {
        for (int l = 0; l < trial_dim; l++) {
          dk(static_cast<size_t>(e), 0, static_cast<size_t>(k + trial_ndof * l)) += K_elem[0][k][0][l];
        }
      }
    } 

    if constexpr (!std::is_same< test, QOI >::value) {
      // Note: we "transpose" these values to get them into the layout that mfem expects
      for_loop<test_ndof, test_dim, trial_ndof, trial_dim>([&](int i, int j, int k, int l) {
        dk(static_cast<size_t>(e), static_cast<size_t>(i + test_ndof * j), static_cast<size_t>(k + trial_ndof * l)) += K_elem[i][k][j][l];
      });
    }
    // clang-format on
  }
}

}  // namespace domain_integral

}  // namespace serac
