#ifndef WHATEVER_COEFFICIENT_HPP
#define WHATEVER_COEFFICIENT_HPP

#include "coefficients/stdfunction_coefficient.hpp"

namespace serac{

struct whatever_coefficient {
  public:
    whatever_coefficient(){};
    whatever_coefficient(double);
    whatever_coefficient(std::function <double(const mfem::Vector &)>);

    operator bool() { return self_ == nullptr; }
    operator mfem::Coefficient&() { return *self_; }
    operator mfem::Coefficient&() const { return *self_; }

    std::unique_ptr<mfem::Coefficient> self_;
};

struct whatever_vector_coefficient {
  public:
    whatever_vector_coefficient(){};
    whatever_vector_coefficient(mfem::Vector);
    whatever_vector_coefficient(int, std::function <void(const mfem::Vector &, mfem::Vector &)>);

    operator bool() { return self_ == nullptr; }
    operator mfem::VectorCoefficient&() { return *self_; }
    operator mfem::VectorCoefficient&() const { return *self_; }

    std::unique_ptr<mfem::VectorCoefficient> self_;
};

}

#endif