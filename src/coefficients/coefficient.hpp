#ifndef COEFFICIENT_HPP
#define COEFFICIENT_HPP

#include <functional>
#include "coefficients/stdfunction_coefficient.hpp"
#include "physics/utilities/finite_element_state.hpp"

namespace serac{

class TransformedCoefficient;

struct coefficient {
  public:
    coefficient(){};
    coefficient(double);
    coefficient(std::function <double(const mfem::Vector &)>);
    coefficient(const FiniteElementState &);
    coefficient(coefficient &&, std::function<double(const double)> && f);

    coefficient(coefficient && other) = default;
    coefficient & operator=(coefficient && other) = default;

    operator bool() { return self_ == nullptr; }

    double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
      return self_->Eval(T, ip);
    }

    operator mfem::Coefficient&() { return *self_; }
    operator mfem::Coefficient&() const { return *self_; }

    std::unique_ptr<mfem::Coefficient> self_;
};

struct vector_coefficient {
  public:
    vector_coefficient(){};
    vector_coefficient(mfem::Vector);
    vector_coefficient(int, std::function <void(const mfem::Vector &, mfem::Vector &)>);

    operator bool() { return self_ == nullptr; }
    operator mfem::VectorCoefficient&() { return *self_; }
    operator mfem::VectorCoefficient&() const { return *self_; }

    std::unique_ptr<mfem::VectorCoefficient> self_;
};

class TransformedCoefficient : public mfem::Coefficient {

public:

  TransformedCoefficient(coefficient && coef, std::function<double(const double)> && f) : mfem::Coefficient(), coef_(std::move(coef)), f_(std::move(f)) {}

  virtual double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
    return f_(coef_.Eval(T, ip));
  };

private:
  coefficient coef_;
  std::function<double(const double)> f_;
};

}

#endif