#include "mfem.hpp"
#include "genericintegrator.hpp"
#include "qfuncintegrator.hpp"

#pragma once

namespace impl{

template < typename space >
auto Reshape(double * u, int n1, int n2) {
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  } else {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

template < typename space  >
auto Reshape(const double * u, int n1, int n2) {
  if constexpr (space::components == 1) {
    return mfem::Reshape(u, n1, n2);
  } else {
    return mfem::Reshape(u, n1, space::components, n2);
  }
};

template < int ndof >
inline auto Load(const mfem::DeviceTensor<2, const double> & u, int e) {
  return make_tensor<ndof>([&u, e](int i){ return u(i, e); });
}

template < int ndof, int components >
inline auto Load(const mfem::DeviceTensor<3, const double> & u, int e) {
  return make_tensor<components, ndof>([&u, e](int j, int i){ return u(i, j, e); });
}

template < typename space, typename T >
auto Load(const T & u, int e) {
  if constexpr (space::components == 1) {
    return impl::Load<space::ndof>(u, e);
  } else {
    return impl::Load<space::ndof, space::components>(u, e);
  }
};

template < int ndof >
void Add(const mfem::DeviceTensor<2, double> & r_global, tensor< double, ndof > r_local, int e) {
  for (int i = 0; i < ndof; i++) {
    r_global(i, e) += r_local[i];
  }
}

template < int ndof, int components >
void Add(const mfem::DeviceTensor<3, double> & r_global, tensor< double, ndof, components > r_local, int e) {
  for (int i = 0; i < ndof; i++) {
    for (int j = 0; j < components; j++) {
      r_global(i, j, e) += r_local[i][j];
    }
  }
}

template < typename element_type, typename T, int dim = element_type::dim >
auto Preprocess(T u, const tensor<double, dim> xi, const tensor<double,dim,dim> J) {
  if constexpr (element_type::family == Family::H1) {
    return std::tuple{
      dot(u, element_type::shape_functions(xi)),
      dot(u, dot(element_type::shape_function_gradients(xi), inv(J)))
    };
  }

  if constexpr (element_type::family == Family::HCURL) {
    return std::tuple{
      dot(u, dot(element_type::shape_functions(xi), inv(J))),
      dot(u, element_type::shape_function_curl(xi) / det(J))
    };
  }
}

template < typename element_type, typename T, int dim = element_type::dim >
auto Postprocess(T f, const tensor<double, dim> xi, const tensor<double,dim,dim> J) {
  if constexpr (element_type::family == Family::H1) {
    auto W = element_type::shape_functions(xi);
    auto dW_dx = dot(element_type::shape_function_gradients(xi), inv(J));
    return outer(W, std::get<0>(f)) + dot(dW_dx, std::get<1>(f));
  }

  if constexpr (element_type::family == Family::HCURL) {
    auto W = dot(element_type::shape_functions(xi), inv(J));
    auto curl_W = element_type::shape_function_curl(xi) / det(J);
    return (W * std::get<0>(f) + curl_W * std::get<1>(f));
  }
}

} // namespace impl


template < ::Geometry g, typename test, typename trial, int Q, typename lambda > 
void evaluation_kernel(const mfem::Vector & U, mfem::Vector & R, const mfem::Vector & J_, const mfem::Vector & X_, int num_elements, lambda qf) {

  using test_element = finite_element< g, test >;
  using trial_element = finite_element< g, trial >;
  static constexpr int dim = dimension(g);
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule< g, Q >();

  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = impl::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = impl::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor u_local = impl::Load<trial_element>(u, e);

    reduced_tensor <double, test_element::ndof, test_element::components> r_local{};
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto x_q = make_tensor< dim >([&](int i){ return X(q, i, e); });
      auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      auto [u_q, du_dx_q] = impl::Preprocess<trial_element>(u_local, xi, J_q);

      auto qf_output = qf(x_q, u_q, du_dx_q);

      r_local += impl::Postprocess<test_element>(qf_output, xi, J_q) * dx;
    }

    impl::Add(r, r_local, e);

  }

}

template < ::Geometry g, typename test, typename trial, int Q, typename derivatives_type, typename lambda > 
void evaluation_with_derivatives_kernel(const mfem::Vector & U, mfem::Vector & R, derivatives_type * derivatives_ptr, const mfem::Vector & J_, const mfem::Vector & X_, int num_elements, lambda qf) {

  using test_element = finite_element< g, test >;
  using trial_element = finite_element< g, trial >;
  static constexpr int dim = dimension(g);
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule< g, Q >();

  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = impl::Reshape<trial>(U.Read(), trial_ndof, num_elements);
  auto r = impl::Reshape<test>(R.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor u_local = impl::Load<trial_element>(u, e);

    reduced_tensor <double, test_element::ndof, test_element::components> r_local{};
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto x_q = make_tensor< dim >([&](int i){ return X(q, i, e); });
      auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      auto N = trial_element::shape_functions(xi);
      auto dN_dx = dot(trial_element::shape_function_gradients(xi), inv(J_q));

      auto u_q = dot(u_local, N);
      auto du_dx_q = dot(u_local, dN_dx);

      auto args = std::tuple{x_q, u_q, du_dx_q};

      auto [f0, f1] = std::apply(qf, args);

      auto W = test_element::shape_functions(xi);
      auto dW_dx = dot(test_element::shape_function_gradients(xi), inv(J_q));

      r_local += (outer(W, f0) + dot(dW_dx, f1)) * dx;

      derivatives_ptr[e * int(rule.size()) + q] = derivatives_type{};
    }

    impl::Add(r, r_local, e);

  }

}

template < ::Geometry g, typename test_space, typename trial_space, int Q, typename lambda > 
void gradient_kernel(const mfem::Vector & dU, mfem::Vector & dR, mfem::Vector & J_, mfem::Vector & X_, int num_elements, lambda qf) {

  using test_element = finite_element< g, test_space >;
  using trial_element = finite_element< g, trial_space >;
  static constexpr int dim = dimension(g);
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule< g, Q >();

  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto du = mfem::Reshape(dU.Read(), trial_ndof, num_elements);
  auto dr = mfem::Reshape(dR.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor du_local = make_tensor<trial_ndof>([&du, e](int i){ return du(i, e); });

    tensor <double, test_ndof > dr_local{};
    for (int q = 0; q < static_cast<int>(rule.size()); q++) {
      auto xi = rule.points[q];
      auto dxi = rule.weights[q];
      auto x_q = make_tensor< dim >([&](int i){ return X(q, i, e); });
      auto J_q = make_tensor< dim, dim >([&](int i, int j){ return J(q, i, j, e); });
      double dx = det(J_q) * dxi;

      auto N = trial_element::shape_functions(xi);
      auto dN_dx = dot(trial_element::shape_function_gradients(xi), inv(J_q));

      auto u_q = dot(du_local, N);
      auto du_dx_q = dot(du_local, dN_dx);

      auto args = std::tuple{x_q, u_q, du_dx_q};

      auto [f0, f1] = std::apply(qf, args);

      auto W = test_element::shape_functions(xi);
      auto dW_dx = dot(test_element::shape_function_gradients(xi), inv(J_q));

      dr_local += (W * f0 + dot(dW_dx, f1)) * dx;
    }

    for (int i = 0; i < test_ndof; i++) {
      dr(i, e) += dr_local[i];
    }

  }

}


template < typename operations, typename lambda_type >
struct IntegrandImpl {
  lambda_type lambda;
};

template < typename operations, typename lambda_type >
auto Integrand(lambda_type lambda) {
  return IntegrandImpl< operations, lambda_type >{lambda};
};

namespace impl{
  template < typename spaces >
  struct get_trial_space; // undefined

  template < typename test_space, typename trial_space >
  struct get_trial_space< test_space(trial_space) >{
    using type = trial_space;
  }; 

  template < typename spaces >
  struct get_test_space; // undefined

  template < typename test_space, typename trial_space >
  struct get_test_space< test_space(trial_space) >{
    using type = test_space;
  };
}

template < typename T >
using test_space_t = typename impl::get_test_space< T >::type;

template < typename T >
using trial_space_t = typename impl::get_trial_space< T >::type;

template < typename space, int dim >
struct lambda_argument;

template < int p, int c, int dim >
struct lambda_argument< H1<p, c>, dim >{
  using type = std::tuple< reduced_tensor<double, c >, reduced_tensor<double, c, dim> >;
};

template < int p >
struct lambda_argument< Hcurl<p>, 2 >{
  using type = std::tuple< tensor<double, 2>, double >;
};

template < int p >
struct lambda_argument< Hcurl<p>, 3 >{
  using type = std::tuple< tensor<double, 3>, tensor<double,3> >;
};

template < typename spaces >
struct VolumeIntegral {

  static constexpr int dim = 2;
  using test_space = test_space_t< spaces >;
  using trial_space = trial_space_t< spaces >;

  template < typename lambda_type >
  VolumeIntegral(int num_elements, const mfem::Vector & J, const mfem::Vector & X, lambda_type && qf) : J_(J), X_(X) {

    // these lines of code figure out the argument types that will be passed
    // into the quadrature function in the finite element kernel.
    //
    // we use them to observe the output type and allocate memory to store 
    // the derivative information at each quadrature point
    using x_t = tensor< double, dim >;
    using u_du_t = typename lambda_argument< trial_space, dim >::type;
    using arg_t = decltype(std::tuple_cat(std::tuple{x_t{}}, make_dual(u_du_t{})));
    using derivative_type = decltype(get_gradient(std::apply(qf, arg_t{})));

    // derivatives of integrand w.r.t. {u, du_dx}
    auto num_quadrature_points = static_cast<uint32_t>(X.Size() / dim);
    qf_derivatives.resize(sizeof(derivative_type) * num_quadrature_points);

    auto qf_derivatives_ptr = reinterpret_cast< derivative_type * >(qf_derivatives.data());

    constexpr int Q = std::max(test_space::order, trial_space::order) + 1;

    evaluation = [=](const mfem::Vector & U, mfem::Vector & R){ 
      evaluation_kernel< ::Geometry::Quadrilateral, test_space, trial_space, Q >(U, R, J_, X_, num_elements, qf);
    };

    //evaluation_with_derivatives = [=](const mfem::Vector & U, mfem::Vector & R){ 
    //  evaluation_with_derivatives_kernel< ::Geometry::Quadrilateral, test_space, trial_space, Q >(U, R, qf_derivatives_ptr, J_, X_, num_elements, qf);
    //};

  }

  void Mult(const mfem::Vector & input_E, mfem::Vector & output_E) const {
    evaluation(input_E, output_E);
  }

  const mfem::Vector J_; 
  const mfem::Vector X_;

  std::vector < char > qf_derivatives;

  std::function < void(const mfem::Vector &, mfem::Vector &) > evaluation;
  std::function < void(const mfem::Vector &, mfem::Vector &) > evaluation_with_derivatives;

};

struct Gradient {

  mfem::Vector & operator()(const mfem::Vector & /*x*/) {
    return output;
  }

  // operator HypreParMatrix() { /* not currently supported */ }

  mfem::Vector output;

};

template < typename T >
struct WeakForm;

template < typename test, typename trial >
struct WeakForm< test(trial) > : public mfem::Operator {

  WeakForm(mfem::ParFiniteElementSpace * test_fes, mfem::ParFiniteElementSpace * trial_fes) :
    test_space(test_fes),
    trial_space(trial_fes),
    P_test(test_space->GetProlongationMatrix()),
    G_test(test_space->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
    P_trial(trial_space->GetProlongationMatrix()),
    G_trial(trial_space->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)) {

    MFEM_ASSERT(G_test, "Some GetElementRestriction error");
    MFEM_ASSERT(G_trial, "Some GetElementRestriction error");

    input_L.SetSize(P_test->Height(), mfem::Device::GetMemoryType());
    input_E.SetSize(G_test->Height(), mfem::Device::GetMemoryType());

    output_E.SetSize(G_trial->Height(), mfem::Device::GetMemoryType());
    output_L.SetSize(P_trial->Height(), mfem::Device::GetMemoryType());
  }

  template < typename lambda >
  void AddVolumeIntegral(lambda && integrand, mfem::Mesh & domain) {

    auto num_elements = domain.GetNE();
    if (num_elements == 0) {
      std::cout << "error: mesh has no elements" << std::endl;
      return;
    }

    auto dim = domain.Dimension();
    for (int e = 0; e < num_elements; e++) {
      if (domain.GetElementType(e) != supported_types[dim]) {          
        std::cout << "error: mesh contains unsupported element types" << std::endl;
      }
    }

    const mfem::FiniteElement& el = *test_space->GetFE(0);

    const mfem::IntegrationRule ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

    auto geom = domain.GetGeometricFactors(ir, mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS);

    // emplace_back rather than push_back to avoid dangling references in std::function
    volume_integrals.emplace_back(num_elements, geom->J, geom->X, integrand);

  }

  virtual void Mult(const mfem::Vector & input_T, mfem::Vector & output_T) const {

    // get the values for each local processor
    P_trial->Mult(input_T, input_L); 

    // get the values for each element on the local processor
    G_trial->Mult(input_L, input_E); 

    // compute residual contributions at the element level and sum them
    // 
    // note: why should we serialize these integral evaluations?
    //       these could be performed in parallel and merged in the reduction process 
    //
    // TODO investigate performance of alternative implementation described above
    output_E = 0.0;
    for (auto integral : volume_integrals) {
      integral.Mult(input_E, output_E);
    }
    
    // scatter-add to compute residuals on the local processor
    G_test->MultTranspose(output_E, output_L); 

    // scatter-add to compute global residuals
    P_test->MultTranspose(output_L, output_T);

    output_T.HostReadWrite();
    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      output_T(ess_tdof_list[i]) = 0.0;
    }

  }

  // note: this gets more interesting when having more than one trial space
  void SetEssentialBC(const mfem::Array<int>& ess_attr) {
    static_assert(std::is_same_v<test, trial>, "can't specify essential bc on incompatible spaces");
    test_space->GetEssentialTrueDofs(ess_attr, ess_tdof_list);
  }

  Gradient & gradient() { return grad; }

  mutable mfem::Vector input_L, input_E, output_L, output_E;

  mfem::ParFiniteElementSpace * test_space, * trial_space;
  mfem::Array<int> ess_tdof_list;

  const mfem::Operator * P_test, * G_test;
  const mfem::Operator * P_trial, * G_trial;

  std::vector < VolumeIntegral< test(trial) > > volume_integrals;

  Gradient grad;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {
    mfem::Element::POINT,
    mfem::Element::SEGMENT,
    mfem::Element::QUADRILATERAL,
    mfem::Element::HEXAHEDRON
  };

};
