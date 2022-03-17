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
  return get_gradient(detail::apply_qf(qf, tensor<double, dim>{}, make_dual_wrt<i>(qf_arguments{}), qpt_data));
};

template <int i>
struct DerivativeWRT {
};

template <int Q, Geometry g, typename test, typename... trials>
struct KernelConfig {
};

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
template <typename S, typename T, typename derivatives_type, typename lambda, typename qpt_data_type>
struct EvaluationKernel;

/**
 * @overload
 * @note evaluation kernel with no differentiation
 */
template <int Q, Geometry geom, typename test, typename... trials, typename lambda, typename qpt_data_type>
struct EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda, qpt_data_type> {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided

  using EVector_t =
      EVectorView<exec, finite_element<geom, trials>...>;  ///< the type of container used to access element values

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
                   std::size_t num_elements, lambda qf, QuadratureData<qpt_data_type>& data)
      : J_(J), X_(X), num_elements_(num_elements), qf_(qf), data_(data)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::array<mfem::Vector, num_trial_spaces>& U, mfem::Vector& R)
  {
    std::array<const double*, num_trial_spaces> ptrs;
    for (uint32_t j = 0; j < num_trial_spaces; j++) {
      ptrs[j] = U[j].Read();
    }
    EVector_t u(ptrs, std::size_t(num_elements_));

    using test_element              = finite_element<geom, test>;
    using element_residual_type     = typename test_element::residual_type;
    static constexpr int  dim       = dimension_of(geom);
    static constexpr int  test_ndof = test_element::ndof;
    static constexpr auto rule      = GaussQuadratureRule<geom, Q>();

    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements_);
    auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements_);
    auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, int(num_elements_));  // TODO: integer conversions

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {
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
        auto arg = Preprocess<geom, trials...>(u_elem, xi, J_q);

        // evaluate the user-specified constitutive model
        //
        // note: make_dual(arg) promotes those arguments to dual number types
        // so that qf_output will contain values and derivatives
        auto qf_output = detail::apply_qf(qf_, x_q, arg, data_(int(e), q));

        // integrate qf_output against test space shape functions / gradients
        // to get element residual contributions
        r_elem += Postprocess<test_element>(qf_output, xi, J_q) * dx;
      }

      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, int(e));
    }
  }

  const mfem::Vector&            J_;             ///< Jacobian matrix entries at each quadrature point
  const mfem::Vector&            X_;             ///< Spatial positions of each quadrature point
  std::size_t                    num_elements_;  ///< how many elements in the domain
  lambda                         qf_;            ///< q-function
  QuadratureData<qpt_data_type>& data_;          ///< (optional) user-provided quadrature data
};

/**
 * @overload
 * @note evaluation kernel that also calculates derivative w.r.t. `I`th trial space
 */
template <int I, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda,
          typename qpt_data_type>
struct EvaluationKernel<DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda,
                        qpt_data_type> {
  static constexpr auto exec             = ExecutionSpace::CPU;     ///< this specialization is CPU-specific
  static constexpr int  num_trial_spaces = int(sizeof...(trials));  ///< how many trial spaces are provided

  using EVector_t =
      EVectorView<exec, finite_element<geom, trials>...>;  ///< the type of container used to access element values

  /**
   * @brief initialize the functor by providing the necessary quadrature point data
   *
   * @param qf_derivatives a container for the derivatives of the q-function w.r.t. trial space I
   * @param J values of sqrt(det(J^T * J)) at each quadrature point
   * @param X Spatial positions of each quadrature point
   * @param num_elements how many elements in the domain
   * @param qf q-function
   * @param data user-specified quadrature data to pass to the q-function
   */
  EvaluationKernel(DerivativeWRT<I>, KernelConfig<Q, geom, test, trials...>,
                   CPUArrayView<derivatives_type, 2> qf_derivatives, const mfem::Vector& J, const mfem::Vector& X,
                   std::size_t num_elements, lambda qf, QuadratureData<qpt_data_type>& data)
      : qf_derivatives_(qf_derivatives), J_(J), X_(X), num_elements_(num_elements), qf_(qf), data_(data)
  {
  }

  /**
   * @brief integrate the q-function over the specified domain, at the specified trial space values
   *
   * @param U input E-vectors
   * @param R output E-vector
   */
  void operator()(const std::array<mfem::Vector, num_trial_spaces>& U, mfem::Vector& R)
  {
    std::array<const double*, num_trial_spaces> ptrs;
    for (uint32_t j = 0; j < num_trial_spaces; j++) {
      ptrs[j] = U[j].Read();
    }
    EVector_t u(ptrs, std::size_t(num_elements_));

    using test_element              = finite_element<geom, test>;
    using element_residual_type     = typename test_element::residual_type;
    static constexpr int  dim       = dimension_of(geom);
    static constexpr int  test_ndof = test_element::ndof;
    static constexpr auto rule      = GaussQuadratureRule<geom, Q>();

    // mfem provides this information in 1D arrays, so we reshape it
    // into strided multidimensional arrays before using
    auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements_);
    auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements_);
    auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, int(num_elements_));  // TODO: integer conversions

    // for each element in the domain
    for (uint32_t e = 0; e < num_elements_; e++) {
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
        auto arg = Preprocess<geom, trials...>(u_elem, xi, J_q);

        // evaluate the user-specified constitutive model
        //
        // note: make_dual(arg) promotes those arguments to dual number types
        // so that qf_output will contain values and derivatives
        auto qf_output = detail::apply_qf(qf_, x_q, make_dual_wrt<I>(arg), data_(int(e), q));

        // integrate qf_output against test space shape functions / gradients
        // to get element residual contributions
        r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

        // here, we store the derivative of the q-function w.r.t. its input arguments
        //
        // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
        qf_derivatives_(static_cast<size_t>(e), static_cast<size_t>(q)) = get_gradient(qf_output);
      }

      // once we've finished the element integration loop, write our element residuals
      // out to memory, to be later assembled into global residuals by mfem
      detail::Add(r, r_elem, int(e));
    }
  }

  ExecArrayView<derivatives_type, 2, exec> qf_derivatives_;  ///< derivatives of the q-function w.r.t. trial space `I`
  const mfem::Vector&                      J_;               ///< Jacobian matrix entries at each quadrature point
  const mfem::Vector&                      X_;               ///< Spatial positions of each quadrature point
  std::size_t                              num_elements_;    ///< how many elements in the domain
  lambda                                   qf_;              ///< q-function
  QuadratureData<qpt_data_type>&           data_;            ///< (optional) user-provided quadrature data
};

template <int Q, Geometry geom, typename test, typename... trials, typename lambda, typename qpt_data_type>
EvaluationKernel(KernelConfig<Q, geom, test, trials...>, const mfem::Vector&, const mfem::Vector&, int, lambda,
                 QuadratureData<qpt_data_type>&)
    -> EvaluationKernel<void, KernelConfig<Q, geom, test, trials...>, void, lambda, qpt_data_type>;

template <int i, int Q, Geometry geom, typename test, typename... trials, typename derivatives_type, typename lambda,
          typename qpt_data_type>
EvaluationKernel(DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, CPUArrayView<derivatives_type, 2>,
                 const mfem::Vector&, const mfem::Vector&, int, lambda, QuadratureData<qpt_data_type>&)
    -> EvaluationKernel<DerivativeWRT<i>, KernelConfig<Q, geom, test, trials...>, derivatives_type, lambda,
                        qpt_data_type>;

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
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename test_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, int(num_elements));     // TODO: integer conversions
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, int(num_elements));  // TODO: integer conversions

  // for each element in the domain
  for (uint32_t e = 0; e < num_elements; e++) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, int(e));  // TODO: integer conversions

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
      auto darg = Preprocess<trial_element>(du_elem, xi, J_q);

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
