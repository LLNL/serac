#ifndef COEFFICIENT_HPP
#define COEFFICIENT_HPP

#include <functional>

#include "physics/utilities/finite_element_state.hpp"

namespace serac {

class TransformedCoefficient;

struct CoefficientWrapper {
public:
  CoefficientWrapper(){};
  CoefficientWrapper(double);
  CoefficientWrapper(std::function<double(const mfem::Vector&)>);
  CoefficientWrapper(std::function<double(const mfem::Vector&, double)>);
  CoefficientWrapper(const FiniteElementState&);
  CoefficientWrapper(CoefficientWrapper&&, std::function<double(const double)>&& f);

  CoefficientWrapper(CoefficientWrapper&& other) = default;
  CoefficientWrapper& operator=(CoefficientWrapper&& other) = default;

  bool is_initialized() { return self_ == nullptr; }

  void SetTime(double t) { self_->SetTime(t); }

  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) { return self_->Eval(T, ip); }

  operator mfem::Coefficient &() { return *self_; }
  operator mfem::Coefficient &() const { return *self_; }

  std::unique_ptr<mfem::Coefficient> self_;
};

struct VectorCoefficientWrapper {
public:
  VectorCoefficientWrapper(){};
  VectorCoefficientWrapper(mfem::Vector);
  VectorCoefficientWrapper(int, std::function<void(const mfem::Vector&, mfem::Vector&)>);

  // operator bool() { return self_ == nullptr; }
  operator mfem::VectorCoefficient &() { return *self_; }
  operator mfem::VectorCoefficient &() const { return *self_; }

  std::unique_ptr<mfem::VectorCoefficient> self_;
};

class TransformedCoefficient : public mfem::Coefficient {
public:
  TransformedCoefficient(CoefficientWrapper&& coef, std::function<double(const double)>&& f)
      : mfem::Coefficient(), coef_(std::move(coef)), f_(std::move(f))
  {
  }

  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
  {
    return f_(coef_.Eval(T, ip));
  };

private:
  CoefficientWrapper                  coef_;
  std::function<double(const double)> f_;
};

}  // namespace serac

#endif