#pragma once

#include <functional>
#include <variant>

#include "mfem.hpp"
#include "numerics/expr_template_ops.hpp"
#include "physics/utilities/boundary_condition_manager.hpp"
#include "physics/utilities/equation_solver.hpp"

class SecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
public:
  using signature = void(const double, const double, const double, const mfem::Vector&, const mfem::Vector&,
                         mfem::Vector&);
  SecondOrderODE() : mfem::SecondOrderTimeDependentOperator(0, 0.0) {}
  SecondOrderODE(int n, std::function<signature> f) : mfem::SecondOrderTimeDependentOperator(n, 0.0), f(f) {}

  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
  {
    f(t, 0.0, 0.0, u, du_dt, d2u_dt2);
  }
  void ImplicitSolve(const double c0, const double c1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    f(t, c0, c1, u, du_dt, d2u_dt2);
  }

  std::function<signature> f;
};

class FirstOrderODE : public mfem::TimeDependentOperator {
public:
  using signature = void(const double, const double, const mfem::Vector&, mfem::Vector&);
  FirstOrderODE() : mfem::TimeDependentOperator(0, 0.0) {}
  FirstOrderODE(int n, std::function<signature> f) : mfem::TimeDependentOperator(n, 0.0), f(f) {}
  FirstOrderODE(SecondOrderODE other_ode) : mfem::TimeDependentOperator(2 * other_ode.Height(), other_ode.GetTime()) {

    int n = other_ode.Height();

    // here, we assume that x and dx_dt have the following block form:
    // 
    //     ⎡ du_dt ⎤          ⎡ d2u_dt2 ⎤
    // x = ⎢  ---  ⎢, dx_dt = ⎢   ---   ⎢
    //     ⎣   u   ⎦          ⎣  du_dt  ⎦
    //
    f = [=](const double t, const double dt, const mfem::Vector& x, mfem::Vector& dx_dt) { 

      mfem::Vector u(x.GetData() + n, n);
      mfem::Vector du_dt(x.GetData(), n);
      mfem::Vector du_dt_next(dx_dt.GetData() + n, n);
      mfem::Vector d2u_dt2_next(dx_dt.GetData(), n);

      // solve for d2u_dt2_next at the end of the timestep
      other_ode.f(t, dt * dt, dt, u, du_dt, d2u_dt2_next);

      du_dt_next = du_dt + d2u_dt2_next * dt;

    };
  }
  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { f(t, 0.0, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { f(t, dt, u, du_dt); }
  std::function<signature> f;
};
