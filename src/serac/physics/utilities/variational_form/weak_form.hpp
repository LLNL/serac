#include "mfem.hpp"
#include "genericintegrator.hpp"
#include "qfuncintegrator.hpp"

#pragma once

template < ::Geometry g, typename test_space, typename trial_space, int Q, typename lambda > 
void evaluation_kernel(mfem::Vector & R, const mfem::Vector & U, mfem::Vector & J_, mfem::Vector & X_, int num_elements, lambda qf) {

  using test_element = finite_element< g, test_space >;
  using trial_element = finite_element< g, trial_space >;
  static constexpr int dim = dimension(g);
  static constexpr int test_ndof = test_element::ndof;
  static constexpr int trial_ndof = trial_element::ndof;
  static constexpr auto rule = GaussQuadratureRule< g, Q >();

  auto X = mfem::Reshape(X_.Read(), rule.size(), dim, num_elements);
  auto J = mfem::Reshape(J_.Read(), rule.size(), dim, dim, num_elements);
  auto u = mfem::Reshape(U.Read(), trial_ndof, num_elements);
  auto r = mfem::Reshape(R.ReadWrite(), test_ndof, num_elements);

  for (int e = 0; e < num_elements; e++) {
    tensor u_local = make_tensor<trial_ndof>([&u, e](int i){ return u(i, e); });

    tensor <double, test_ndof > r_local{};
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

      r_local += (W * f0 + dot(dW_dx, f1)) * dx;
    }

    for (int i = 0; i < test_ndof; i++) {
      r(i, e) += r_local[i];
    }

  }

}

template < ::Geometry g, typename test_space, typename trial_space, int Q, typename lambda > 
void gradient_kernel(mfem::Vector & dR, const mfem::Vector & dU, mfem::Vector & J_, mfem::Vector & X_, int num_elements, lambda qf) {

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

struct VolumeIntegral {

  void Mult(const mfem::Vector & input_E, mfem::Vector & output_E) {
    output_E = input_E;
  }

};

struct Gradient {

  mfem::Vector & operator()(const mfem::Vector & /*x*/) {
    return output;
  }

  // operator HypreParMatrix() { /* not currently supported */ }

  mfem::Vector output;

};

template < typename T >
class WeakForm;

template < typename test, typename trial >
class WeakForm< test(trial) > : public mfem::Operator {

  WeakForm(mfem::ParFiniteElementSpace * test_space, mfem::ParFiniteElementSpace * trial_space) :
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

  void Add(VolumeIntegral /*f*/) {

  }

  void Mult(const mfem::Vector & input_T, mfem::Vector & output_T) {

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

  }

  Gradient & gradient() { return grad; }

  mfem::Vector input_L, input_E, output_L, output_E;

  const mfem::Operator * P_test, * G_test;
  const mfem::Operator * P_trial, * G_trial;

  std::vector < VolumeIntegral > volume_integrals;

  Gradient grad;

};
