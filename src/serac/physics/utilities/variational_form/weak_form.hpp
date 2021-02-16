#include "mfem.hpp"
#include "genericintegrator.hpp"
#include "qfuncintegrator.hpp"

#pragma once

template < ::Geometry g, PolynomialDegree p, int Q, int components > 
void fem_kernel(const mfem::Vector & values, mfem::Vector & residuals, mfem::Vector & J, mfem::Vector & W, int num_elements) {

  using element_type = finite_element< g, Family::H1, p, components >;

  // for each element
  for (int e = 0; e < num_elements; e++) {

    // load the values associated with that element
    reduced_tensor < double, element_type::ndof, components > element_values{};
    for (int i = 0; i < element_type::ndof; i++) {
      if constexpr (components == 1) {
        element_values[i] = values[element_type::ndof * e + i];
      } else {
        for (int j = 0; j < components; j++) {
          element_values[i] = values[components * (element_type::ndof * e + i) + j];
        }
      }
    }

    // loop over quadrature points
    reduced_tensor < double, element_type::ndof, components > element_residuals{};
    for (int q = 0; q < Q; q++) {

    }

    // store the element residuals 
    for (int i = 0; i < element_type::ndof; i++) {
      if constexpr (components == 1) {
        residuals[element_type::ndof * e + i] = element_residuals[i];
      } else {
        for (int j = 0; j < components; j++) {
          residuals[components * (element_type::ndof * e + i) + j] = element_residuals[i][j];
        }
      }
    }

  }

}

struct VolumeIntegral {

  

  mfem::Vector & operator()(const Vector & x) {
    return output;
  }

  mfem::Vector output;

};

struct Gradient {

  mfem::Vector & operator()(const Vector & x) {
    return output;
  }

  // operator HypreParMatrix() { /* not currently supported */ }

  mfem::Vector output;

};

template < typename T >
class WeakForm;

template < typename test, typename trial >
class WeakForm< test(trial) > : public mfem::Operator {

  WeakForm(ParFiniteElementSpace * test_space, ParFiniteElementSpace * trial_space) {
    P_test = test_space->GetProlongationMatrix();
    G_test = test_space->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
    MFEM_ASSERT(G_test, "Some GetElementRestriction error");

    input_L.SetSize(P_test->Height(), Device::GetMemoryType());
    input_E.SetSize(G_test->Height(), Device::GetMemoryType());


    P_trial = trial_space->GetProlongationMatrix();
    G_trial = trial_space->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
    MFEM_ASSERT(G_trial, "Some GetElementRestriction error");

    output_E.SetSize(G_trial->Height(), Device::GetMemoryType());
    output_L.SetSize(P_trial->Height(), Device::GetMemoryType());
    output_T.SetSize(P_trial->Width(), Device::GetMemoryType());
  }

  template < typename annotation >
  void Add(integrand f) {

  }

  mfem::Vector & operator()(const Vector & input_T) {
    P_trial->Mult(input_T, input_L);
    G_trial->Mult(input_L, input_E);

    output_E = 0.0;
    for (auto integral : volume_integrals) {
      output_E += integral(input_E);
    }

    G_test->MultTranspose(output_E, output_L);
    P_test->MultTranspose(output_L, output_T);

    return output_T;
  }

  Gradient & gradient() { return grad; }

  mfem::Vector input_L, input_E, output_T, output_L, output_E;

  mfem::Operator * P_test, * G_test;
  mfem::Operator * P_trial, * G_trial;

  std::vector < VolumeIntegral > volume_integrals;

  Gradient grad;

}
