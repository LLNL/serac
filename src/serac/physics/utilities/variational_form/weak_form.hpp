#include "mfem.hpp"
#include "genericintegrator.hpp"
#include "qfuncintegrator.hpp"

#pragma once

template < typename T >
class WeakForm;

template < typename test_space, typename trial_space >
class WeakForm< test_space(trial_space) > : public mfem::Operator {

  WeakForm();

  template < typename integrand >
  void AddVolumeIntegral(integrand f) {}

  template < typename integrand >
  void AddBoundaryIntegral(integrand f) {}

  mfem::Vector & operator()(const Vector & x) { return output; }

  mfem::Vector output;
};

template <
 : public Operator {
  class Gradient : public Operator {
  public:
    Gradient(ParVariationalForm& f) : Operator(f.Height()), form(f){};

    void Mult(const Vector& x, Vector& y) const override { form.GradientMult(x, y); }

  private:
    ParVariationalForm& form;
  };

public:
  ParVariationalForm(ParFiniteElementSpace* f);

  void Mult(const Vector& x, Vector& y) const override;

  void AddDomainIntegrator(GenericIntegrator* i)
  {
    domain_integrators.Append(i);
    i->Setup(*fes);
  }

  template <typename integrator_type, typename qfunc_type, typename... qfunc_args_type>
  void AddDomainIntegrator(qfunc_type f, qfunc_args_type const&... fargs)
  {
    if constexpr (std::is_same_v<integrator_type, DomainLFIntegrator>) {
      // FIXME: Remove when GCC 8 is no longer supported, clumsy fix for
      // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85883.  Overhead should
      // be low as this will just call the implicit move constructor
      auto i = new QFunctionIntegrator(QFunctionIntegrator(
          [&](auto... args) {
            auto du = std::get<1>(std::tuple{args...});
            return std::tuple{f(args...), decltype(du){}};
          },
          0, *fes->GetParMesh(), fargs...));

      domain_integrators.Append(i);
      i->Setup(*fes);
    }

    if constexpr (std::is_same_v<integrator_type, DiffusionIntegrator>) {
      auto i = new QFunctionIntegrator(QFunctionIntegrator(
          [&](auto... args) {
            auto du = std::get<1>(std::tuple{args...});
            return std::tuple{0.0, f(args...) * du};
          },
          0, *fes->GetParMesh(), fargs...));

      domain_integrators.Append(i);
      i->Setup(*fes);
    }
  }

  void AssumeLinear() { is_linear = true; }

  // Return an Operator that provides a Mult(x, y) which is the MatVec of the
  // gradient of the ParVariationalForm wrt x. Acts as a "passthrough" to
  // ::GradientMult in order to satisfy mfem interfaces.
  Operator& GetGradient(const Vector& x) const override;

  // Return an assmbled parallel matrix which represents the gradient of the
  // ParVariationalForm wrt to x.
  HypreParMatrix* GetGradientMatrix(const Vector& x);

  void SetEssentialBC(const Array<int>& ess_attr);

protected:
  // y = F'(x_lin) * v
  void GradientMult(const Vector& v, Vector& y) const;

  bool                      is_linear = false;
  ParFiniteElementSpace*    fes;
  Array<GenericIntegrator*> domain_integrators;
  Array<int>                ess_tdof_list;

  // T -> L
  const Operator* P;

  // L -> E
  const Operator* G;

  mutable Vector x_local, y_local, v_local, px, py, pv;

  // State to build the Gradient on, single source of the true state.
  mutable Vector x_lin;

  mutable Gradient grad;

  HypreParMatrix* gradient_matrix = nullptr;
};
