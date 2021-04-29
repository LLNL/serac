#pragma once

#include "mfem.hpp"

#include "serac/physics/utilities/variational_form/tensor.hpp"
#include "serac/physics/utilities/variational_form/quadrature.hpp"
#include "serac/physics/utilities/variational_form/finite_element.hpp"
#include "serac/physics/utilities/variational_form/tuple_arithmetic.hpp"
#include "serac/physics/utilities/variational_form/integral.hpp"

template <typename T>
struct WeakForm;

// WeakForm is intended to be like std::function for finite element kernels
//
// that is: you tell it the inputs (trial spaces) for a kernel, and the outputs (test space) like std::function
//
// e.g.
//   this code represents a function that takes an integer argument and returns a double
//   std::function< double(double, int) > my_func;
//
//   this code represents a function that takes values from an Hcurl field and returns a residual vector associated with
//   an H1 field WeakForm< H1(Hcurl) > my_residual;
//
// to actually use it, you use the methods WeakForm::Add****Integral(integrand, domain_of_integration),
// where integrand is a q-function lambda or functor (see
// https://libceed.readthedocs.io/en/latest/libCEEDapi/#theoretical-framework for additional information on the idea
// behind a quadrature function and its inputs/outputs) and domain_of_integration is an mesh.
//
// supported methods of this type include:
//
// for domains made up of quadrilaterals embedded in R^2
// my_residual.AddAreaIntegral(integrand, domain_of_integration);
//                      or
// my_residual.AddDomainIntegral(Dimension<2>{}, integrand, domain_of_integration);
//
// for domains made up of quadrilaterals embedded in R^3
// my_residual.AddSurfaceIntegral(integrand, domain_of_integration);
//
// for domains made up of hexahedra embedded in R^3
// my_residual.AddVolumeIntegral(integrand, domain_of_integration);
//                      or
// my_residual.AddDomainIntegral(Dimension<3>{}, integrand, domain_of_integration);
template <typename test, typename trial>
struct WeakForm<test(trial)> : public mfem::Operator {
  enum class Operation
  {
    Mult,
    GradientMult
  };

  class Gradient : public mfem::Operator {
  public:
    Gradient(WeakForm& f) : mfem::Operator(f.Height()), form(f){};

    virtual void Mult(const mfem::Vector& x, mfem::Vector& y) const override { form.GradientMult(x, y); }

  private:
    WeakForm<test(trial)>& form;
  };

  WeakForm(mfem::ParFiniteElementSpace* test_fes, mfem::ParFiniteElementSpace* trial_fes)
      : Operator(test_fes->GetTrueVSize(), trial_fes->GetTrueVSize()),
        test_space(test_fes),
        trial_space(trial_fes),
        P_test(test_space->GetProlongationMatrix()),
        G_test(test_space->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        P_trial(trial_space->GetProlongationMatrix()),
        G_trial(trial_space->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC)),
        grad(*this)
  {
    MFEM_ASSERT(G_test, "Some GetElementRestriction error");
    MFEM_ASSERT(G_trial, "Some GetElementRestriction error");

    input_L.SetSize(P_test->Height(), mfem::Device::GetMemoryType());
    input_E.SetSize(G_test->Height(), mfem::Device::GetMemoryType());

    output_E.SetSize(G_trial->Height(), mfem::Device::GetMemoryType());
    output_L.SetSize(P_trial->Height(), mfem::Device::GetMemoryType());

    my_output_T.SetSize(test_fes->GetTrueVSize(), mfem::Device::GetMemoryType());

    dummy.SetSize(trial_fes->GetTrueVSize(), mfem::Device::GetMemoryType());
  }

  template <int geometry_dim, int spatial_dim, typename lambda>
  void AddIntegral(Dimension<geometry_dim>, Dimension<spatial_dim>, lambda&& integrand, mfem::Mesh& domain)
  {
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

    if constexpr (geometry_dim == spatial_dim) {
      const mfem::FiniteElement&   el = *test_space->GetFE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
      auto                         geom =
          domain.GetGeometricFactors(ir, mfem::GeometricFactors::COORDINATES | mfem::GeometricFactors::JACOBIANS);
      domain_integrals.emplace_back(num_elements, geom->J, geom->X, Dimension<geometry_dim>{}, Dimension<spatial_dim>{},
                                    integrand);
      return;
    } else if constexpr ((geometry_dim + 1) == spatial_dim) {
      const mfem::FiniteElement&   el = *test_space->GetBE(0);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
      constexpr auto flags = mfem::FaceGeometricFactors::COORDINATES | mfem::FaceGeometricFactors::JACOBIANS |
                             mfem::FaceGeometricFactors::NORMALS;
      auto geom = domain.GetFaceGeometricFactors(ir, flags, mfem::FaceType::Boundary);
      boundary_integrals.emplace_back(num_elements, geom->J, geom->X, Dimension<geometry_dim>{},
                                      Dimension<spatial_dim>{}, integrand);
      return;
    } else if constexpr (true) {
      // if static_assert has a literal 'false' for its first arg,
      // it will trigger even when this branch isn't selected. So,
      // we define an expression that is always false (see first
      // if constexpr expression) to work around this limitation
      constexpr bool always_false = (geometry_dim == spatial_dim);
      static_assert(always_false, "unsupported integral dimensionality");
    }
  }

  template <typename lambda>
  void AddAreaIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<2>{} /* geometry */, Dimension<2>{} /* spatial */, integrand, domain);
  }

  template <typename lambda>
  void AddVolumeIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<3>{} /* geometry */, Dimension<3>{} /* spatial */, integrand, domain);
  }

  template <int d, typename lambda>
  void AddDomainIntegral(Dimension<d>, lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<d>{} /* geometry */, Dimension<d>{} /* spatial */, integrand, domain);
  }

  template <typename lambda>
  void AddSurfaceIntegral(lambda&& integrand, mfem::Mesh& domain)
  {
    AddIntegral(Dimension<2>{} /* geometry */, Dimension<3>{} /* spatial */, integrand, domain);
  }

  template <Operation op = Operation::Mult>
  void Evaluation(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
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
    for (auto& integral : domain_integrals) {
      if constexpr (op == Operation::Mult) {
        integral.Mult(input_E, output_E);
      }

      if constexpr (op == Operation::GradientMult) {
        integral.GradientMult(input_E, output_E);
      }
    }

    // scatter-add to compute residuals on the local processor
    G_test->MultTranspose(output_E, output_L);

    // scatter-add to compute global residuals
    P_test->MultTranspose(output_L, output_T);

    output_T.HostReadWrite();
    for (int i = 0; i < ess_tdof_list.Size(); i++) {
      if constexpr (op == Operation::Mult) {
        output_T(ess_tdof_list[i]) = 0.0;
      }

      if constexpr (op == Operation::GradientMult) {
        output_T(ess_tdof_list[i]) = input_T(ess_tdof_list[i]);
      }
    }
  }

  virtual void Mult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::Mult>(input_T, output_T);
  }

  mfem::Vector& operator()(const mfem::Vector& input_T) const
  {
    Evaluation<Operation::Mult>(input_T, my_output_T);
    return my_output_T;
  }

  virtual void GradientMult(const mfem::Vector& input_T, mfem::Vector& output_T) const
  {
    Evaluation<Operation::GradientMult>(input_T, output_T);
  }

  virtual mfem::Operator& GetGradient(const mfem::Vector& x) const
  {
    Mult(x, dummy);  // this is ugly
    return grad;
  }

  // note: this gets more interesting when having more than one trial space
  void SetEssentialBC(const mfem::Array<int>& ess_attr)
  {
    static_assert(std::is_same_v<test, trial>, "can't specify essential bc on incompatible spaces");
    test_space->GetEssentialTrueDofs(ess_attr, ess_tdof_list);
  }

  mutable mfem::Vector input_L, input_E, output_L, output_E, my_output_T, dummy;

  mfem::ParFiniteElementSpace *test_space, *trial_space;
  mfem::Array<int>             ess_tdof_list;

  const mfem::Operator *P_test, *G_test;
  const mfem::Operator *P_trial, *G_trial;

  std::vector<Integral<test(trial)> > domain_integrals;
  std::vector<Integral<test(trial)> > boundary_integrals;

  // simplex elements are currently not supported;
  static constexpr mfem::Element::Type supported_types[4] = {mfem::Element::POINT, mfem::Element::SEGMENT,
                                                             mfem::Element::QUADRILATERAL, mfem::Element::HEXAHEDRON};

  mutable Gradient grad;
};
