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

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "serac/physics/utilities/quadrature_data.hpp"

#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/integral_utilities.hpp"
#include "serac/physics/utilities/functional/quadrature.hpp"
#include "serac/physics/utilities/functional/tuple_arithmetic.hpp"
#include "serac/physics/utilities/functional/domain_integral_shared.hpp"

#if defined(__CUDACC__)
#include "serac/physics/utilities/functional/integral_cuda.cuh"
#endif

namespace serac {

namespace domain_integral {

/**
 * @brief The base kernel template used to create different finite element calculation routines
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p DomainIntegral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
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
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda,
          typename qpt_data_type = void>
void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
                       const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf,
                       QuadratureData<qpt_data_type>& data = dummy_qdata)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = detail::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = detail::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the DOF values for this particular element
    tensor u_elem = detail::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   x_q = make_tensor<dim>([&](int i) { return X(q, i, e); });  // Physical coords of qpt
      auto   J_q = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = det(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = Preprocess<trial_element>(u_elem, xi, J_q);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      // TODO: Refactor the call to qf here since the current approach is somewhat messy
      auto qf_output = [&qf, &x_q, &arg, &data, e, q]() {
        if constexpr (std::is_same_v<qpt_data_type, void>) {
          // [[maybe_unused]] not supported in captures
          (void)data;
          (void)e;
          (void)q;
          return qf(x_q, make_dual(arg));
        } else {
          return qf(x_q, make_dual(arg), data(e, q));
        }
      }();

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
      detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements) = get_gradient(qf_output);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(r, r_elem, e);
  }
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
void gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR, derivatives_type* derivatives_ptr,
                     const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  dim        = dimension_of(g);
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = detail::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = detail::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the (change in) values for this particular element
    tensor du_elem = detail::Load<trial_element>(du, e);

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

      auto dq_darg = detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements);

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += Postprocess<test_element>(dq, xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    detail::Add(dr, dr_elem, e);
  }
}

}  // namespace domain_integral

/**
 * @brief The base kernel template used to compute tangent element entries that can be assembled
 * into a tangent matrix
 *
 * @tparam test The type of the test function space
 * @tparam trial The type of the trial function space
 * The above spaces can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
 *
 * Template parameters other than the test and trial spaces are used for customization + optimization
 * and are erased through the @p std::function members of @p Integral
 * @tparam g The shape of the element (only quadrilateral and hexahedron are supported at present)
 * @tparam Q Quadrature parameter describing how many points per dimension
 * @tparam derivatives_type Type representing the derivative of the q-function w.r.t. its input arguments
 *
 *
 * @param[inout] K_e The full set of per-element element tangents [test_ndofs x test_dim, trial_ndofs x trial_dim]
 * @param[in] derivatives_ptr The address at which derivatives of the q-function with
 * respect to its arguments are stored
 * @param[in] J_ The Jacobians of the element transformations at all quadrature points
 * @see mfem::GeometricFactors
 * @param[in] num_elements The number of elements in the mesh
 */
template <Geometry g, typename test, typename trial, int Q, typename derivatives_type>
void gradient_matrix_kernel(mfem::Vector& K_e, derivatives_type* derivatives_ptr, const mfem::Vector& J_,
                            int num_elements)
{
  using test_element  = finite_element<g, test>;
  using trial_element = finite_element<g, trial>;
  //  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int                  test_ndof  = test_element::ndof;
  static constexpr int                  test_dim   = test_element::components;
  static constexpr int                  trial_ndof = trial_element::ndof;
  static constexpr int                  trial_dim  = test_element::components;
  static constexpr auto                 rule       = GaussQuadratureRule<g, Q>();
  static constexpr int                  dim        = dimension_of(g);
  [[maybe_unused]] static constexpr int curl_dim   = dim == 3 ? 3 : 1;

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto dk = mfem::Reshape(K_e.ReadWrite(), test_ndof * test_dim, trial_ndof * trial_dim, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    tensor<double, test_ndof * test_dim, trial_ndof * trial_dim> K_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto                    xi_q   = rule.points[q];
      auto                    dxi_q  = rule.weights[q];
      auto                    J_q    = make_tensor<dim, dim>([&](int i, int j) { return J(q, i, j, e); });
      auto                    detJ_q = det(J_q);
      [[maybe_unused]] double dx     = detJ_q * dxi_q;

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = detail::AccessDerivatives(derivatives_ptr, e, q, rule, num_elements);

      // evaluate shape functions
      [[maybe_unused]] auto M = test_element::shape_functions(xi_q);
      [[maybe_unused]] auto N = trial_element::shape_functions(xi_q);
      if constexpr (test_element::family == Family::HCURL) {
        M = dot(M, inv(J_q));
      }
      if constexpr (trial_element::family == Family::HCURL) {
        N = dot(N, inv(J_q));
      }

      auto f00 = serac::get<0>(serac::get<0>(dq_darg));
      auto f01 = serac::get<1>(serac::get<0>(dq_darg));
      auto f10 = serac::get<0>(serac::get<1>(dq_darg));
      auto f11 = serac::get<1>(serac::get<1>(dq_darg));

      // df0_du stiffness contribution
      // size(M) = test_ndof
      // size(N) = trial_ndof
      // size(df0_du) = test_dim x trial_dim
      if constexpr (!is_zero<decltype(f00)>::value) {
        auto df0_du = convert_to_tensor_with_shape<test_dim, trial_dim>(f00);
        for_loop<test_ndof, test_dim, trial_ndof, trial_dim>([&](auto i, auto id, auto j, auto jd) {
          // maybe we should have a mapping for dofs x dim
          K_elem[i + test_ndof * id][j + trial_ndof * jd] += M[i] * df0_du[id][jd] * N[j] * dx;
        });
      }

      if constexpr (test_element::family == Family::H1 || test_element::family == Family::L2) {
        [[maybe_unused]] auto dM_dx = dot(test_element::shape_function_gradients(xi_q), inv(J_q));
        [[maybe_unused]] auto dN_dx = dot(trial_element::shape_function_gradients(xi_q), inv(J_q));

        // df0_dgradu stiffness contribution
        // size(M) = test_ndof
        // size(df0_dgradu) = test_dim x trial_dim x dim
        // size(dN_dx) = trial_ndof x dim
        if constexpr (!is_zero<decltype(f01)>::value) {
          auto df0_dgradu = convert_to_tensor_with_shape<test_dim, trial_dim, dim>(f01);
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, dim>(
              [&](auto i, auto id, auto j, auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                    M[i] * df0_dgradu[id][jd][dummy_i] * dN_dx[j][dummy_i] * dx;
              });
        }

        // // df1_du stiffness contribution
        // size(dM_dx) = test_ndof x dim
        // size(N) = trial_ndof
        // size(df1_du) = test_dim x dim x trial_dim
        if constexpr (!is_zero<decltype(f10)>::value) {
          auto& df1_du = f10;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  K_elem[i * test_dim][j * trial_dim] += dM_dx[i][dummy_i] * df1_du[dummy_i] * N[j] * dx;
                } else {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      dM_dx[i][dummy_i] * df1_du[id][dummy_i][jd] * N[j] * dx;
                }
              });
        }

        // df1_dgradu stiffness contribution
        // size(dM_dx) = test_ndof x dim
        // size(dN_dx) = trial_ndof x dim
        // size(df1_dgradu) = test_dim x dim x trial_dim x dim
        if constexpr (!is_zero<decltype(f11)>::value) {
          auto& df1_dgradu = f11;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, dim, dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i, auto dummy_j) {
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  K_elem[i][j] += dM_dx[i][dummy_i] * df1_dgradu[dummy_i][dummy_j] * dN_dx[j][dummy_j] * dx;
                } else {
                  // maybe we should have a mapping for dofs x dim
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      dM_dx[i][dummy_i] * df1_dgradu[id][dummy_i][jd][dummy_j] * dN_dx[j][dummy_j] * dx;
                }
              });
        }
      } else {  // HCurl

        [[maybe_unused]] auto curl_M = test_element::shape_function_curl(xi_q) / det(J_q);
        if constexpr (dim == 3) {
          curl_M = dot(curl_M, transpose(J_q));
        }

        [[maybe_unused]] auto curl_N = trial_element::shape_function_curl(xi_q) / det(J_q);
        if constexpr (dim == 3) {
          curl_N = dot(curl_N, transpose(J_q));
        }

        // df0_dgradu stiffness contribution
        // size(M) = test_ndof
        // size(df0_dcurlu) = test_dim x trial_dim x curl_dim
        // size(curl_N) = trial_ndof x curl_dim
        if constexpr (!is_zero<decltype(f01)>::value) {
          auto df0_dcurlu = convert_to_tensor_with_shape<test_dim, trial_dim, curl_dim>(f01);
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_dim>(
              [&](auto i, auto id, auto j, auto jd, [[maybe_unused]] auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (dim == 3) {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                      M[i] * df0_dcurlu[id][jd][dummy_i] * curl_N[j][dummy_i] * dx;
                } else {
                  K_elem[i + test_ndof * id][j + trial_ndof * jd] += M[i] * df0_dcurlu[id][jd] * curl_N[j] * dx;
                }
              });
        }

        // // df1_du stiffness contribution
        // size(curl_M) = test_ndof x curl_dim
        // size(N) = trial_ndof
        // size(df1_du) = test_dim x curl_dim x trial_dim
        if constexpr (!is_zero<decltype(f10)>::value) {
          auto& df1_du = f10;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, auto dummy_i) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  if constexpr (dim == 3) {
                    K_elem[i * test_dim][j * trial_dim] += curl_M[i][dummy_i] * df1_du[dummy_i] * N[j] * dx;
                  } else {
                    K_elem[i * test_dim][j * trial_dim] += curl_M[i] * df1_du * N[j] * dx;
                  }
                } else {
                  if constexpr (dim == 3) {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                        curl_M[i][dummy_i] * df1_du[id][dummy_i][jd] * N[j] * dx;
                  } else {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] += curl_M[i] * df1_du[id][jd] * N[j] * dx;
                  }
                }
              });
        }

        // df1_dcurlu stiffness contribution
        // size(curl_M) = test_ndof x curl_dim
        // size(curl_N) = trial_ndof x curl_dim
        // size(df1_dcurl) = test_dim x curl_dim x trial_dim x curl_dim
        if constexpr (!is_zero<decltype(f11)>::value) {
          auto& df1_dcurlu = f11;
          for_loop<test_ndof, test_dim, trial_ndof, trial_dim, curl_dim, curl_dim>(
              [&](auto i, [[maybe_unused]] auto id, auto j, [[maybe_unused]] auto jd, [[maybe_unused]] auto dummy_i,
                  [[maybe_unused]] auto dummy_j) {
                // maybe we should have a mapping for dofs x dim
                if constexpr (test_dim == 1 && trial_dim == 1) {
                  if constexpr (dim == 3) {
                    K_elem[i][j] += curl_M[i][dummy_i] * df1_dcurlu[dummy_i][dummy_j] * curl_N[j][dummy_j] * dx;
                  } else {
                    K_elem[i][j] += curl_M[i] * df1_dcurlu * curl_N[j] * dx;
                  }
                } else {
                  if constexpr (dim == 3) {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] +=
                        curl_M[i][dummy_i] * df1_dcurlu[id][dummy_i][jd][dummy_j] * curl_N[j][dummy_j] * dx;
                  } else {
                    K_elem[i + test_ndof * id][j + trial_ndof * jd] += curl_M[i] * df1_dcurlu[id][jd] * curl_N[j] * dx;
                  }
                }
              });
        }
      }
    }

    // once we've finished the element integration loop, write our element stifness
    // out to memory, to be later assembled into global stifnesss by mfem
    for (int i = 0; i < test_ndof * test_dim; i++) {
      for (int j = 0; j < trial_ndof * trial_dim; j++) {
        dk(i, j, e) += K_elem[i][j];
      }
    }
  }
}

/**
 * @brief Describes a single integral term in a weak forumulation of a partial differential equation
 * @tparam spaces A @p std::function -like set of template parameters that describe the test and trial
 * function spaces, i.e., @p test(trial)
 */
template <typename spaces, typename execution_policy>
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

    // the derivative_type data is stored in a shared_ptr here, because it can't be a
    // member variable on the DomainIntegral class template (it depends on the lambda function,
    // which isn't known until the time of construction).
    //
    // This shared_ptr should have a comparable lifetime to the DomainIntegral instance itself, since
    // the reference count will increase when it is captured by the lambda functions below, and
    // the reference count will go back to zero after those std::functions are deconstructed in
    // DomainIntegral::~DomainIntegral()
    //
    // derivatives are stored as a 2D array, such that quadrature point q of element e is accessed by
    // qf_derivatives[e * quadrature_points_per_element + q]
    auto qf_derivatives = make_shared_array<derivative_type, execution_policy>(num_quadrature_points);

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    //
    // note: the qf_derivatives_ptr is copied by value to each lambda function below,
    //       to allow the evaluation kernel to pass derivative values to the gradient kernel

    if constexpr (std::is_same_v<execution_policy, serac::cpu_policy>) {
      std::cout << "cpu_policy\n";
      evaluation_ = [=, &data](const mfem::Vector& U, mfem::Vector& R) {
        domain_integral::evaluation_kernel<geometry, test_space, trial_space, Q>(U, R, qf_derivatives.get(), J_, X_,
                                                                                 num_elements, qf, data);
      };

      gradient_ = [=](const mfem::Vector& dU, mfem::Vector& dR) {
        domain_integral::gradient_kernel<geometry, test_space, trial_space, Q>(dU, dR, qf_derivatives.get(), J_,
                                                                               num_elements);
      };

      gradient_mat_ = [=](mfem::Vector& K_e) {
        gradient_matrix_kernel<geometry, test_space, trial_space, Q>(K_e, qf_derivatives.get(), J_, num_elements);
      };
    }

    if constexpr (std::is_same_v<execution_policy, serac::gpu_policy>) {
      // todo
      std::cout << "gpu_policy\n";
#if defined(__CUDACC__)
      evaluation_ = [=](const mfem::Vector& U, mfem::Vector& R) {
        domain_integral::evaluation_kernel_cuda<geometry, test_space, trial_space, Q>(U, R, qf_derivatives.get(), J_,
                                                                                      X_, num_elements, qf);
      };

      gradient_ = [=](const mfem::Vector& dU, mfem::Vector& dR) {
        domain_integral::gradient_kernel_cuda<geometry, test_space, trial_space, Q>(dU, dR, qf_derivatives.get(), J_,
                                                                                    num_elements);
      };
#endif
    }
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
  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E) const { gradient_(input_E, output_E); }

  /**
   * @brief Computes the element stiffness matrices, storing them in an `mfem::Vector` that has been reshaped into a
   * multidimensional array
   * @param[inout] K_e The reshaped vector as a mfem::DeviceTensor of size (test_dim * test_dof, trial_dim * trial_dof,
   * elem)
   */
  void ComputeElementMatrices(mfem::Vector& K_e) const { gradient_mat_(K_e); }

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
  std::function<void(const mfem::Vector&, mfem::Vector&)> gradient_;
  /**
   * @brief Type-erased handle to gradient matrix assembly kernel
   * @see gradient_matrix_kernel
   */
  std::function<void(mfem::Vector&)> gradient_mat_;
};

}  // namespace serac
