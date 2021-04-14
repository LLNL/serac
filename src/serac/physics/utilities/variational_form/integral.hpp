#pragma once

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include "serac/physics/utilities/variational_form/tensor.hpp"
#include "serac/physics/utilities/variational_form/quadrature.hpp"
#include "serac/physics/utilities/variational_form/quadrature_data.hpp"
#include "serac/physics/utilities/variational_form/finite_element.hpp"
#include "serac/physics/utilities/variational_form/tuple_arithmetic.hpp"

namespace impl {

template <typename space>
auto Reshape(double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  } else {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

template <typename space>
auto Reshape(const double* u, int n1, int n2)
{
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  } else {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

// these impl::Load functions extract the dof values for a particular element
//
// for the case of only 1 dof per node, impl::Load returns a tensor<double, ndof>
template <int ndof>
inline auto Load(const mfem::DeviceTensor<2, const double>& u, int e)
{
  return make_tensor<ndof>([&u, e](int i) { return u(i, e); });
}

// for the case of multiple dofs per node, impl::Load returns a tensor<double, components, ndof>
template <int ndof, int components>
inline auto Load(const mfem::DeviceTensor<3, const double>& u, int e)
{
  return make_tensor<components, ndof>([&u, e](int j, int i) { return u(i, j, e); });
}

template <typename space, typename T>
auto Load(const T& u, int e)
{
  if constexpr (space::components == 1) {
    return impl::Load<space::ndof>(u, e);
  } else {
    return impl::Load<space::ndof, space::components>(u, e);
  }
};

template <int ndof>
void Add(const mfem::DeviceTensor<2, double>& r_global, tensor<double, ndof> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    r_global(i, e) += r_local[i];
  }
}

template <int ndof, int components>
void Add(const mfem::DeviceTensor<3, double>& r_global, tensor<double, ndof, components> r_local, int e)
{
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}

// the impl::Preprocess computes the arguments to be passed into the q-function
//
// by default:
//   H1 family elements will compute {value, gradient}
//   Hcurl family elements will compute {value, curl}
//   Hdiv family elements will compute {value, divergence}  TODO
//   L2 family elements will compute value                  TODO
//
// in the future, the user will be able to override these defaults
// to omit unused components (e.g. specify that they only need the gradient)
template <typename element_type, typename T, int dim>
auto Preprocess(T u, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    return std::tuple{dot(u, element_type::shape_functions(xi)),
                      dot(u, dot(element_type::shape_function_gradients(xi), inv(J)))};
  }

  if constexpr (element_type::family == Family::HCURL) {
    auto value = dot(u, dot(element_type::shape_functions(xi), inv(J)));
    auto curl  = dot(u, element_type::shape_function_curl(xi) / det(J));
    if constexpr (dim == 3) {
      curl = dot(curl, transpose(J));
    }
    return std::tuple{value, curl};
  }
}

// this specialization of impl::Preprocess is called when doing integrals
// where the spatial dimension is different from the dimension of the element geometry
// (i.e. surface integrals, line integrals, etc)
//
// in this case, only the function values are calculated
// (Question: are gradients useful in these cases or not?)
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
auto Preprocess(T u, const tensor<double, geometry_dim> xi,
                [[maybe_unused]] const tensor<double, spatial_dim, geometry_dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    return dot(u, element_type::shape_functions(xi));
  }

  if constexpr (element_type::family == Family::HCURL) {
    return dot(u, dot(element_type::shape_functions(xi), inv(J)));
  }
}

// the impl::Postprocess function computes residual contributions from the output of the q-function
//
// this involves integrating the q-function output against functions from the test
// function space.
//
// by default:
//   H1 family elements integrate std::get<0>(f) against the test space shape functions
//                            and std::get<1>(f) against the test space shape function gradients
//
//   Hcurl family elements integrate std::get<0>(f) against the test space shape functions
//                               and std::get<1>(f) against the curl of the test space shape functions
//
//   TODO:
//   Hdiv family elements integrate std::get<0>(f) against the test space shape functions
//                              and std::get<1>(f) against the divergence of the test space shape functions
//
//   TODO
//   L2 family elements integrate f against test space shape functions
//
// in the future, the user will be able to override these defaults
// to omit unused components (e.g. provide only the term to be integrated against test function gradients)
template <typename element_type, typename T, int dim>
auto Postprocess(T f, const tensor<double, dim> xi, const tensor<double, dim, dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    auto W     = element_type::shape_functions(xi);
    auto dW_dx = dot(element_type::shape_function_gradients(xi), inv(J));
    return outer(W, std::get<0>(f)) + dot(dW_dx, std::get<1>(f));
  }

  if constexpr (element_type::family == Family::HCURL) {
    auto W      = dot(element_type::shape_functions(xi), inv(J));
    auto curl_W = element_type::shape_function_curl(xi) / det(J);
    if constexpr (dim == 3) {
      curl_W = dot(curl_W, transpose(J));
    }
    return (W * std::get<0>(f) + curl_W * std::get<1>(f));
  }
}

// this specialization of impl::Postprocess is called when doing integrals
// where the spatial dimension is different from the dimension of the element geometry
// (i.e. surface integrals, line integrals, etc)
//
// in this case, q-function outputs are only integrated against test space shape functions
// (Question: should test function gradients be supported here or not?)
template <typename element_type, typename T, int geometry_dim, int spatial_dim>
auto Postprocess(T f, const tensor<double, geometry_dim> xi,
                 [[maybe_unused]] const tensor<double, spatial_dim, geometry_dim> J)
{
  if constexpr (element_type::family == Family::H1) {
    return outer(element_type::shape_functions(xi), f);
  }

  if constexpr (element_type::family == Family::HCURL) {
    return outer(element_type::shape_functions(xi), dot(inv(J), f));
  }
}

// impl::Measure takes in a Jacobian matrix and computes the
// associated length / area / volume ratio of the transformation.
//
// in general, this is given by sqrt(det(J^T * J)), but for the case
// where J is square, this is equivalent to just det(J)
template <int m, int n>
auto Measure(tensor<double, m, n> A)
{
  if constexpr (m == n) {
    return det(A);
  } else {
    return sqrt(det(transpose(A) * A));
  }
}

}  // namespace impl

// this is the base kernel template that is used to
// create different finite element calculation routines
//
// the customization options include:
//   - Geometry: element shape (only quadrilateral and hexahedron are supported at present)
//   - test/trial spaces: can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
//   - Q: quadrature parameter describing how many points per dimension
//   - derivatives_type: type representing the derivative of the q-function (see below) w.r.t. its input arguments
//   - lambda: the actual quadrature-function (either lambda function or functor object) to
//             be evaluated at each quadrature point.
//             See https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework
//             for additional information on the idea behind a quadrature function and its inputs/outputs
//   - data: The per-quadrature point state data, e.g., for material history
template <::Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
          typename derivatives_type, typename lambda, typename PointData = void>
void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
                       const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf,
                       QuadratureData<PointData>& data = dummy_qdata)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto X = mfem::Reshape(X_.Read(), rule.size(), spatial_dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
  auto u = impl::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = impl::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the values for this particular element
    tensor u_elem = impl::Load<trial_element>(u, e);

    // this is where we will accumulate the element residual tensor
    element_residual_type r_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   x_q = make_tensor<spatial_dim>([&](int i) { return X(q, i, e); });
      auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = impl::Measure(J_q) * dxi;

      // evaluate the value/derivatives needed for the q-function at this quadrature point
      auto arg = impl::Preprocess<trial_element>(u_elem, xi, J_q);

      // evaluate the user-specified constitutive model
      //
      // note: make_dual(arg) promotes those arguments to dual number types
      // so that qf_output will contain values and derivatives
      auto qf_output = [&qf, &x_q, &arg, &data, e, q]() {
        if constexpr (std::is_same_v<PointData, void>) {
          return qf(x_q, make_dual(arg));
        } else {
          return qf(x_q, make_dual(arg), data(e, q));
        }
      }();

      // integrate qf_output against test space shape functions / gradients
      // to get element residual contributions
      r_elem += impl::Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;

      // here, we store the derivative of the q-function w.r.t. its input arguments
      //
      // this will be used by other kernels to evaluate gradients / adjoints / directional derivatives
      derivatives_ptr[e * int(rule.size()) + q] = get_gradient(qf_output);
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    impl::Add(r, r_elem, e);
  }
}

// this is the base kernel template that is used to
// create custom directional derivative kernels associated with finite element calculations
//
// the customization options include:
//   - Geometry: element shape (only quadrilateral and hexahedron are supported at present)
//   - test/trial spaces: can be any combination of {H1, Hcurl, Hdiv (TODO), L2 (TODO)}
//   - Q: quadrature parameter describing how many points per dimension
//   - derivatives_type: type representing the derivative of the q-function (see below) w.r.t. its input arguments
//
// note: lambda does not appear as a template argument, as the directional derivative is
//       inherently just a linear transformation
template <::Geometry g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
          typename derivatives_type>
void gradient_kernel(const mfem::Vector& dU, mfem::Vector& dR, derivatives_type* derivatives_ptr,
                     const mfem::Vector& J_, int num_elements)
{
  using test_element               = finite_element<g, test>;
  using trial_element              = finite_element<g, trial>;
  using element_residual_type      = typename trial_element::residual_type;
  static constexpr int  test_ndof  = test_element::ndof;
  static constexpr int  trial_ndof = trial_element::ndof;
  static constexpr auto rule       = GaussQuadratureRule<g, Q>();

  // mfem provides this information in 1D arrays, so we reshape it
  // into strided multidimensional arrays before using
  auto J  = mfem::Reshape(J_.Read(), rule.size(), spatial_dim, geometry_dim, num_elements);
  auto du = impl::Reshape<trial>(dU.Read(), trial_ndof, num_elements);
  auto dr = impl::Reshape<test>(dR.ReadWrite(), test_ndof, num_elements);

  // for each element in the domain
  for (int e = 0; e < num_elements; e++) {
    // get the (change in) values for this particular element
    tensor du_elem = impl::Load<trial_element>(du, e);

    // this is where we will accumulate the (change in) element residual tensor
    element_residual_type dr_elem{};

    // for each quadrature point in the element
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      // get the position of this quadrature point in the parent and physical space,
      // and calculate the measure of that point in physical space.
      auto   xi  = rule.points[q];
      auto   dxi = rule.weights[q];
      auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
      double dx  = impl::Measure(J_q) * dxi;

      // evaluate the (change in) value/derivatives at this quadrature point
      auto darg = impl::Preprocess<trial_element>(du_elem, xi, J_q);

      // recall the derivative of the q-function w.r.t. its arguments at this quadrature point
      auto dq_darg = derivatives_ptr[e * int(rule.size()) + q];

      // use the chain rule to compute the first-order change in the q-function output
      auto dq = chain_rule(dq_darg, darg);

      // integrate dq against test space shape functions / gradients
      // to get the (change in) element residual contributions
      dr_elem += impl::Postprocess<test_element>(dq, xi, J_q) * dx;
    }

    // once we've finished the element integration loop, write our element residuals
    // out to memory, to be later assembled into global residuals by mfem
    impl::Add(dr, dr_elem, e);
  }
}

namespace impl {
template <typename spaces>
struct get_trial_space;  // undefined

template <typename test_space, typename trial_space>
struct get_trial_space<test_space(trial_space)> {
  using type = trial_space;
};

template <typename spaces>
struct get_test_space;  // undefined

template <typename test_space, typename trial_space>
struct get_test_space<test_space(trial_space)> {
  using type = test_space;
};
}  // namespace impl

template <typename T>
using test_space_t = typename impl::get_test_space<T>::type;

template <typename T>
using trial_space_t = typename impl::get_trial_space<T>::type;

template <typename space, int geometry_dim, int spatial_dim>
struct lambda_argument;

template <int p, int c, int dim>
struct lambda_argument<H1<p, c>, dim, dim> {
  using type = std::tuple<reduced_tensor<double, c>, reduced_tensor<double, c, dim>>;
};

// for now, we only provide the interpolated values for surface integrals
template <int p, int c, int geometry_dim, int spatial_dim>
struct lambda_argument<H1<p, c>, geometry_dim, spatial_dim> {
  using type = reduced_tensor<double, c>;
};

template <int p>
struct lambda_argument<Hcurl<p>, 2, 2> {
  using type = std::tuple<tensor<double, 2>, double>;
};

template <int p>
struct lambda_argument<Hcurl<p>, 3, 3> {
  using type = std::tuple<tensor<double, 3>, tensor<double, 3>>;
};

/**
 * @brief Determines the return type of a qfunction lambda
 * @tparam lambda_type The type of the lambda itself
 * @tparam x_t The type of the "value" itself
 * @tparam u_du_t The type of the derivative
 * @tparam PointData The type of the per-quadrature state data, @p void when not applicable
 */
template <typename lambda_type, typename x_t, typename u_du_t, typename PointData, typename SFINAE = void>
struct qf_result {
  using type = std::invoke_result_t<lambda_type, x_t, decltype(make_dual(std::declval<u_du_t>()))>;
};

template <typename lambda_type, typename x_t, typename u_du_t, typename PointData>
struct qf_result<lambda_type, x_t, u_du_t, PointData, std::enable_if_t<!std::is_same_v<PointData, void>>> {
  // Expecting that qf lambdas take an lvalue reference to a state
  using type = std::invoke_result_t<lambda_type, x_t, decltype(make_dual(std::declval<u_du_t>())),
                                    std::add_lvalue_reference_t<PointData>>;
};

static constexpr ::Geometry supported_geometries[] = {::Geometry::Point, ::Geometry::Segment, ::Geometry::Quadrilateral,
                                                      ::Geometry::Hexahedron};

template <typename spaces>
struct Integral {
  using test_space  = test_space_t<spaces>;
  using trial_space = trial_space_t<spaces>;

  template <int geometry_dim, int spatial_dim, typename lambda_type, typename PointData = void>
  Integral(int num_elements, const mfem::Vector& J, const mfem::Vector& X, Dimension<geometry_dim>,
           Dimension<spatial_dim>, lambda_type&& qf, QuadratureData<PointData>& data = dummy_qdata)
      : J_(J), X_(X)
  {
    constexpr auto geometry = supported_geometries[geometry_dim];
    constexpr auto Q        = std::max(test_space::order, trial_space::order) + 1;

    // these lines of code figure out the argument types that will be passed
    // into the quadrature function in the finite element kernel.
    //
    // we use them to observe the output type and allocate memory to store
    // the derivative information at each quadrature point
    using x_t             = tensor<double, spatial_dim>;
    using u_du_t          = typename lambda_argument<trial_space, geometry_dim, spatial_dim>::type;
    using qf_result_type  = typename qf_result<lambda_type, x_t, u_du_t, PointData>::type;
    using derivative_type = decltype(get_gradient(std::declval<qf_result_type>()));

    auto num_quadrature_points = static_cast<uint32_t>(X.Size() / spatial_dim);
    qf_derivatives.resize(sizeof(derivative_type) * num_quadrature_points);

    auto qf_derivatives_ptr = reinterpret_cast<derivative_type*>(qf_derivatives.data());

    // this is where we actually specialize the finite element kernel templates with
    // our specific requirements (element type, test/trial spaces, quadrature rule, q-function, etc).
    //
    // std::function's type erasure lets us wrap those specific details inside a function with known signature
    //
    // note: the qf_derivatives_ptr is copied by value to each lambda function below,
    //       to allow the evaluation kernel to pass derivative values to the gradient kernel
    evaluation = [=, &data](const mfem::Vector& U, mfem::Vector& R) {
      evaluation_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(U, R, qf_derivatives_ptr, J_,
                                                                                         X_, num_elements, qf, data);
    };

    gradient = [=](const mfem::Vector& dU, mfem::Vector& dR) {
      gradient_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(dU, dR, qf_derivatives_ptr, J_,
                                                                                       num_elements);
    };
  }

  void Mult(const mfem::Vector& input_E, mfem::Vector& output_E) const { evaluation(input_E, output_E); }

  void GradientMult(const mfem::Vector& input_E, mfem::Vector& output_E) const { gradient(input_E, output_E); }

  const mfem::Vector J_;
  const mfem::Vector X_;

  std::vector<char> qf_derivatives;

  std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation;
  std::function<void(const mfem::Vector&, mfem::Vector&)> gradient;
};
