#include <functional>

#include "mfem.hpp"


class FirstOrderODE : public mfem::TimeDependentOperator {
public:
  using explicit_signature = void(const double, const mfem::Vector&, mfem::Vector&);
  using implicit_signature = void(const double, const double, const mfem::Vector&, mfem::Vector&);
  std::function<explicit_signature> explicit_func;
  std::function<implicit_signature> implicit_func;

  FirstOrderODE(int n) : mfem::TimeDependentOperator(n, 0.0){};
  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { explicit_func(t, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { implicit_func(t, dt, u, du_dt); }
};

class SecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
public:
  using explicit_signature = void(const double, const mfem::Vector&, const mfem::Vector&, mfem::Vector&);
  using implicit_signature = void(const double, const double, const double, const mfem::Vector&, const mfem::Vector&, mfem::Vector&);
  std::function<explicit_signature> explicit_func;
  std::function<implicit_signature> implicit_func;

  SecondOrderODE(int n) : mfem::SecondOrderTimeDependentOperator(n, 0.0){};
  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const { 
    explicit_func(t, u, du_dt, d2u_dt2); 
  }
  void ImplicitSolve(const double dt0, const double dt1, const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) { 
    implicit_func(t, dt0, dt1, u, du_dt, d2u_dt2); 
  }
};