#pragma once

#include "mfem.hpp"

#include <functional>

class StdFunctionOperator : public mfem::Operator {
public:
  StdFunctionOperator(int n) : mfem::Operator(n) {}
  void Mult(const mfem::Vector& k, mfem::Vector& y) const { function(k, y); };
  mfem::Operator& GetGradient(const mfem::Vector& k) const { return jacobian(k); };
  mutable std::function<void(const mfem::Vector&, mfem::Vector&)> function;
  mutable std::function<mfem::Operator&(const mfem::Vector&)>     jacobian;
};
