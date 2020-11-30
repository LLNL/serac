#ifndef COEFFICIENT_HPP
#define COEFFICIENT_HPP

#include "coefficients/stdfunction_coefficient.hpp"
#include "physics/utilities/finite_element_state.hpp"

namespace serac{

struct coefficient {
  public:
    coefficient(){};
    coefficient(double);
    coefficient(std::function <double(const mfem::Vector &)>);
    coefficient(FiniteElementState &);

    template < typename lambda >
    coefficient(coefficient &, lambda && f) {
      //TODO
      f(); 
    };

    operator bool() { return self_ == nullptr; }
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

}

#endif