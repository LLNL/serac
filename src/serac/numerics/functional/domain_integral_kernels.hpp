// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "serac/numerics/quadrature_data.hpp"
#include "serac/numerics/functional/integral_utilities.hpp"
#include "serac/numerics/functional/evector_view.hpp"

namespace serac {

namespace domain_integral {

template < typename space, typename dimension >
struct QFunctionArgument;

// define what arguments DomainIntegral will pass to 
// qfunctions, depending on the dimension and trial space
template <int p, int dim >
struct QFunctionArgument< H1< p, 1 >, Dimension<dim> >{
  using type = tuple< double, tensor <double, dim> >; 
};
template <int p, int c, int dim >
struct QFunctionArgument< H1< p, c >, Dimension<dim> >{
  using type = tuple< tensor<double, c>, tensor <double, c, dim> >; 
};

template <int p >
struct QFunctionArgument< Hcurl< p >, Dimension<2> >{
  using type = tuple< tensor< double, 2 >, double >;
};

template <int p >
struct QFunctionArgument< Hcurl< p >, Dimension<3> >{
  using type = tuple< tensor< double, 3 >, tensor< double, 3> >;
};

template < int dim, typename ... trials, typename lambda >
auto get_derivative_type(lambda qf) {
  using qf_arguments = serac::tuple < typename QFunctionArgument< trials, Dimension<dim> >::type ... >;
  return get_gradient(detail::apply_qf(qf, tensor<double, dim>{}, make_dual(qf_arguments{}), nullptr));
};

template < int i, int dim, typename ... trials, typename lambda >
auto get_derivative_type(lambda qf) {
  using qf_arguments = serac::tuple < typename QFunctionArgument< trials, serac::Dimension<dim> >::type ... >;
  return get_gradient(detail::apply_qf(qf, tensor<double, dim>{}, make_dual_wrt<i>(qf_arguments{}), nullptr));
};

/**
 * @brief The base kernel template used to create different finite element calculation routines
 *
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function (see below) w.r.t. its input arguments
 * @tparam lambda The actual quadrature-function (either lambda function or functor object) to
 * be evaluated at each quadrature point.
 * @see https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional
 * information on the idea behind a quadrature function and its inputs/outputs
 * @tparam qpt_data_type The type of the data to store for each quadrature point
 *
 * @param[in] U The full set of per-element DOF values (primary input)
 * @param[inout] R The full set of per-element residuals (primary output)
 * @param[out] derivatives_ptr The address at which derivatives of @a lambda with
 * respect to its arguments will be stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @param[in] X_ The actual (not reference) coordinates of all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 * @param[in] qf The actual quadrature function, see @p lambda
 * @param[inout] data The data for each quadrature point
 */
template <int which, int Q, Geometry g, typename test, typename ... trials, typename T, typename derivatives_type, typename lambda,
          typename qpt_data_type = void>
void evaluation_kernel(T u, mfem::Vector& R, CPUView<derivatives_type, 2> qf_derivatives,
                       const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda&& qf,
                       QuadratureData<qpt_data_type>& data = dummy_qdata)
{
  using test_element               = finite_element<g, test>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (uint32_t e = 0; e < uint32_t(num_elements); e++) {

    // get the DOF values for this particular element
    auto u_elem = u[e];

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {

      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   x_q = make_tensor<dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = Preprocess<g, trials...>(u_elem, xi, J_q);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      auto qf_output = detail::apply_qf(qf, x_q, make_dual_wrt<which>(arg), data(int(e), q));

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
      qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q)) = get_gradient(qf_output);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, int(e));
  }
}

template < int i >
struct DerivativeWRT{};

template < int Q, Geometry g, typename test, typename ... trials >
struct KernelConfig{};

template <typename S, typename T, typename derivatives_type, typename lambda, typename qpt_data_type>
struct EvaluationKernel;

template <int i, int Q, Geometry geom, typename test, typename ... trials, typename derivatives_type, typename lambda, typename qpt_data_type>
struct EvaluationKernel< DerivativeWRT<i>, KernelConfig< Q, geom, test, trials ... >, derivatives_type, lambda, qpt_data_type > {

  static constexpr auto exec = ExecutionSpace::CPU;

  static constexpr int num_trial_spaces = int(sizeof ... (trials));

  using EVector_t = EVectorView < exec, finite_element< geom, trials > ... >;

  EvaluationKernel(DerivativeWRT<i>, KernelConfig< Q, geom, test, trials ... >, std::shared_ptr<derivatives_type[]> ptr, const mfem::Vector & J, const mfem::Vector & X, int num_elements, int num_quadrature_points_per_element, lambda qf, QuadratureData<qpt_data_type>& data) : 
    qf_derivatives_ptr_(ptr),
    qf_derivatives_(ptr.get(), num_elements, num_quadrature_points_per_element),
    J_(J),
    X_(X),
    num_elements_(num_elements),
    qf_(qf), 
    data_(data) {}

  void operator() (const std::array< mfem::Vector, num_trial_spaces > & U, mfem::Vector& R) {
    std::array< const double *, num_trial_spaces > ptrs;
    for (uint32_t j = 0; j < num_trial_spaces; j++) { ptrs[j] = U[j].Read(); }
    EVector_t u(ptrs, size_t(num_elements_));

    domain_integral::evaluation_kernel<i, Q, geom, test, trials...>(u, R, qf_derivatives_, J_, X_, num_elements_, qf_, data_);
  }

  std::shared_ptr<derivatives_type[]> qf_derivatives_ptr_;
  ArrayView<derivatives_type, 2, exec> qf_derivatives_;
  const mfem::Vector & J_; 
  const mfem::Vector & X_; 
  int num_elements_; 
  lambda qf_; 
  QuadratureData<qpt_data_type>& data_;

};

template <int i, int Q, Geometry geom, typename test, typename ... trials, typename derivatives_type, typename lambda, typename qpt_data_type>
EvaluationKernel(DerivativeWRT<i>, KernelConfig< Q, geom, test, trials ... >, std::shared_ptr<derivatives_type[]>, const mfem::Vector &, const mfem::Vector &, int, int, lambda, QuadratureData<qpt_data_type>&) ->  
EvaluationKernel< DerivativeWRT<i>, KernelConfig< Q, geom, test, trials ... >, derivatives_type, lambda, qpt_data_type >;

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
template <int Q, Geometry g, typename test, typename ... trials, typename T, typename derivatives_type>
void action_of_gradient_kernel(T du, mfem::Vector& dR, CPUView<derivatives_type, 2> qf_derivatives,
                               const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (uint32_t e = 0; e < uint32_t(num_elements); e++) {

    // get the (change in) values for this particular element
    auto du_elem = du[e];

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the (change in) value/derivatives at this quadrature point
      auto darg = Preprocess<g, trials...>(du_elem, xi, J_q);

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = qf_derivatives(static_cast<size_t>(e), static_cast<size_t>(q));

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, static_cast<int>(e));
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
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void element_gradient_kernel(ArrayView<double, 3, ExecutionSpace::CPU> dk, CPUView<derivatives_type, 2> qf_derivatives,
                             const mfem::Vector& J_, int num_elements)
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
  for (int e = 0; e < num_elements; e++) {
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
          K_elem[0][j] += (q0 * N[j].value + q1 * N[j].derivative) * dx;
        }
      }

      if constexpr (!std::is_same<test, QOI>::value) {
        auto& q00 = serac::get<0>(serac::get<0>(dq_darg));  // derivative of source term w.r.t. field value
        auto& q01 = serac::get<1>(serac::get<0>(dq_darg));  // derivative of source term w.r.t. field derivative
        auto& q10 = serac::get<0>(serac::get<1>(dq_darg));  // derivative of   flux term w.r.t. field value
        auto& q11 = serac::get<1>(serac::get<1>(dq_darg));  // derivative of   flux term w.r.t. field derivative

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
