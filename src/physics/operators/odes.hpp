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



class LinearFirstOrderODE : public mfem::TimeDependentOperator {
public:
  mfem::HypreParMatrix&            M_;
  mfem::HypreParMatrix&            K_;
  mfem::Vector&                    f_;
  serac::BoundaryConditionManager& bcs_;

  serac::EquationSolver invMf_;
  serac::EquationSolver invT_;

  mutable mfem::Vector                  uf;
  mutable mfem::Vector                  uc;
  mutable mfem::Vector                  duc_dt;
  mutable mfem::Vector                  uc_plus;
  mutable mfem::Vector                  uc_minus;
  std::unique_ptr<mfem::HypreParMatrix> T_;
  std::unique_ptr<mfem::HypreParMatrix> Mf_;
  double                                previous_dt;
  double                                epsilon;

  LinearFirstOrderODE(mfem::HypreParMatrix& M, mfem::HypreParMatrix& K, mfem::Vector& f,
                      serac::BoundaryConditionManager& bcs, const serac::LinearSolverParameters& params)
      : mfem::TimeDependentOperator(f.Size(), 0.0),
        M_(M),
        K_(K),
        f_(f),
        bcs_(bcs),
        invMf_(M.GetComm(), params),
        invT_(M.GetComm(), params),
        previous_dt(-1.0),
        epsilon(1.0e-8)
  {
    auto preconditioner = std::make_unique<mfem::HypreSmoother>();
    preconditioner->SetType(mfem::HypreSmoother::Jacobi);

    Mf_ = std::make_unique<mfem::HypreParMatrix>(M);
    bcs_.eliminateAllEssentialDofsFromMatrix(*Mf_);
    invMf_.linearSolver().iterative_mode = false;
    invMf_.SetPreconditioner(std::move(preconditioner));
    invMf_.SetOperator(*Mf_);

    preconditioner = std::make_unique<mfem::HypreSmoother>();
    preconditioner->SetType(mfem::HypreSmoother::Jacobi);

    invT_.linearSolver().iterative_mode = false;
    invT_.SetPreconditioner(std::move(preconditioner));

    uf       = mfem::Vector(f.Size());
    uc       = mfem::Vector(f.Size());
    uc_plus  = mfem::Vector(f.Size());
    uc_minus = mfem::Vector(f.Size());
    duc_dt   = mfem::Vector(f.Size());
  }

  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const
  {
    uf = u;
    uf.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    uc       = 0.0;
    uc_plus  = 0.0;
    uc_minus = 0.0;
    for (const auto& bc : bcs_.essentials()) {
      bc.projectBdrToDofs(uc, t);
      bc.projectBdrToDofs(uc_plus, t + epsilon);
      bc.projectBdrToDofs(uc_minus, t - epsilon);
    }
    duc_dt = (uc_plus - uc_minus) / (2.0 * epsilon);

    du_dt = invMf_ * (f_ - M_ * duc_dt - K_ * uf - K_ * uc);
  }

  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt)
  {
    if (dt != previous_dt) {
      // T = M + dt K
      T_.reset(mfem::Add(1.0, M_, dt, K_));

      // Eliminate the essential DOFs from the T matrix
      bcs_.eliminateAllEssentialDofsFromMatrix(*T_);
      invT_.SetOperator(*T_);

      previous_dt = dt;
    }

    uf = u;
    uf.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    uc       = 0.0;
    uc_plus  = 0.0;
    uc_minus = 0.0;
    for (const auto& bc : bcs_.essentials()) {
      bc.projectBdrToDofs(uc, t);
      bc.projectBdrToDofs(uc_plus, t + epsilon);
      bc.projectBdrToDofs(uc_minus, t - epsilon);
    }
    duc_dt = (uc_plus - uc_minus) / (2.0 * epsilon);

    du_dt = invT_ * (f_ - M_ * duc_dt - K_ * uc - K_ * uf);
  }
};

class LinearSecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
  mfem::HypreParMatrix&            M_;
  mfem::HypreParMatrix&            C_;
  mfem::HypreParMatrix&            K_;
  mfem::Vector&                    f_;
  serac::BoundaryConditionManager& bcs_;

  serac::EquationSolver invMf_;
  serac::EquationSolver invT_;

  mutable mfem::Vector                  uf;
  mutable mfem::Vector                  duf_dt;
  mutable mfem::Vector                  uc;
  mutable mfem::Vector                  duc_dt;
  mutable mfem::Vector                  d2uc_dt2;
  mutable mfem::Vector                  uc_plus;
  mutable mfem::Vector                  uc_minus;
  std::unique_ptr<mfem::HypreParMatrix> T_;
  std::unique_ptr<mfem::HypreParMatrix> Mf_;
  double                                previous_dt0;
  double                                previous_dt1;
  double                                epsilon;

  LinearSecondOrderODE(mfem::HypreParMatrix& M, mfem::HypreParMatrix& C, mfem::HypreParMatrix& K, mfem::Vector& f,
                       serac::BoundaryConditionManager& bcs, const serac::LinearSolverParameters& params)
      : mfem::SecondOrderTimeDependentOperator(f.Size(), 0.0),
        M_(M),
        C_(C),
        K_(K),
        f_(f),
        bcs_(bcs),
        invMf_(M.GetComm(), params),
        invT_(M.GetComm(), params),
        previous_dt0(-1.0),
        previous_dt1(-1.0),
        epsilon(1.0e-4)
  {
    auto preconditioner = std::make_unique<mfem::HypreSmoother>();
    preconditioner->SetType(mfem::HypreSmoother::Jacobi);

    Mf_ = std::make_unique<mfem::HypreParMatrix>(M);
    bcs_.eliminateAllEssentialDofsFromMatrix(*Mf_);
    invMf_.linearSolver().iterative_mode = false;
    invMf_.SetPreconditioner(std::move(preconditioner));
    invMf_.SetOperator(*Mf_);

    preconditioner = std::make_unique<mfem::HypreSmoother>();
    preconditioner->SetType(mfem::HypreSmoother::Jacobi);

    invT_.linearSolver().iterative_mode = false;
    invT_.SetPreconditioner(std::move(preconditioner));

    uf       = mfem::Vector(f.Size());
    duf_dt   = mfem::Vector(f.Size());
    uc       = mfem::Vector(f.Size());
    duc_dt   = mfem::Vector(f.Size());
    d2uc_dt2 = mfem::Vector(f.Size());
    uc_plus  = mfem::Vector(f.Size());
    uc_minus = mfem::Vector(f.Size());
  }

  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
  {
    uf = u;
    uf.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    duf_dt = du_dt;
    duf_dt.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    uc       = 0.0;
    uc_plus  = 0.0;
    uc_minus = 0.0;
    for (const auto& bc : bcs_.essentials()) {
      bc.projectBdrToDofs(uc, t);
      bc.projectBdrToDofs(uc_plus, t + epsilon);
      bc.projectBdrToDofs(uc_minus, t - epsilon);
    }

    duc_dt   = (uc_plus - uc_minus) / (2.0 * epsilon);
    d2uc_dt2 = (uc_plus - 2 * uc + uc_minus) / (epsilon * epsilon);

    d2u_dt2 = invMf_ * (f_ - (M_ * d2uc_dt2 - C_ * duc_dt - K_ * uc) - (C_ * duf_dt - K_ * uf));
  }

  void ImplicitSolve(const double dt0, const double dt1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    if (dt0 != previous_dt0 || dt1 != previous_dt1) {
      // T = M + dt1 * C + 0.5 * (dt0 * dt0) * K
      T_.reset(mfem::Add(1.0, M_, dt1, C_));
      T_.reset(mfem::Add(1.0, *T_, 0.5 * dt0 * dt0, K_));

      bcs_.eliminateAllEssentialDofsFromMatrix(*T_);
      invT_.SetOperator(*T_);
    }

    uf = u;
    uf.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    duf_dt = du_dt;
    duf_dt.SetSubVector(bcs_.allEssentialDofs(), 0.0);

    uc       = 0.0;
    uc_plus  = 0.0;
    uc_minus = 0.0;
    for (const auto& bc : bcs_.essentials()) {
      bc.projectBdrToDofs(uc, t);
      bc.projectBdrToDofs(uc_plus, t + epsilon);
      bc.projectBdrToDofs(uc_minus, t - epsilon);
    }
    duc_dt   = (uc_plus - uc_minus) / (2.0 * epsilon);
    d2uc_dt2 = (uc_plus - 2 * uc + uc_minus) / (epsilon * epsilon);

    d2u_dt2 = invT_ * (f_ - (M_ * d2uc_dt2 - C_ * duc_dt - K_ * uc) - (C_ * duf_dt - K_ * uf));

    previous_dt0 = dt0;
    previous_dt1 = dt1;
  }
};
