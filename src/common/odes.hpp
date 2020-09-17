#include <functional>
#include <variant>

#include "common/boundary_condition_manager.hpp"
#include "common/expr_template_ops.hpp"
#include "mfem.hpp"
#include "solvers/equation_solver.hpp"

class FirstOrderODE : public mfem::TimeDependentOperator {
public:
  using explicit_signature = void(const double, const mfem::Vector&, mfem::Vector&);
  using implicit_signature = void(const double, const double, const mfem::Vector&, mfem::Vector&);

  FirstOrderODE(int n) : mfem::TimeDependentOperator(n, 0.0){};
  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { explicit_func(t, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { implicit_func(t, dt, u, du_dt); }

  template <typename T, typename... arg_types>
  void SetMethod(arg_types&&... args)
  {
    method = std::make_unique<T>(args...);
    method->Init(*this);
  }

  void Step(mfem::Vector& u, double& t, double& dt) { method->Step(u, t, dt); }

  std::unique_ptr<mfem::ODESolver>  method;
  std::function<explicit_signature> explicit_func;
  std::function<implicit_signature> implicit_func;
};

class LinearFirstOrderODE : public mfem::TimeDependentOperator {
public:
  mfem::HypreParMatrix&            M_;
  mfem::HypreParMatrix&            K_;
  mfem::Vector&                    f_;
  serac::BoundaryConditionManager& bcs_;

  serac::EquationSolver invMf_;
  serac::EquationSolver invT_;

  // std::unique_ptr<mfem::ODESolver> method_;

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
      // std::unique_ptr<mfem::ODESolver>&& method)
      : mfem::TimeDependentOperator(f.Size(), 0.0),
        M_(M),
        K_(K),
        f_(f),
        bcs_(bcs),
        invMf_(M.GetComm(), params),
        invT_(M.GetComm(), params),
        // method_(std::move(method)),
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

    // method->Init(*this);
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
    duc_dt = (uc_plus - uc_minus) * (1.0 / (2.0 * epsilon));

    du_dt = invMf_ * (f_ - K_ * uf - K_ * uc - M_ * duc_dt);
  }

  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt)
  {
    if (dt != previous_dt) {
      // T = M + dt K
      T_.reset(mfem::Add(1.0, M_, dt, K_));

      // Eliminate the essential DOFs from the T matrix
      bcs_.eliminateAllEssentialDofsFromMatrix(*T_);
      invT_.SetOperator(*T_);
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

    previous_dt = dt;
  }

  //  template <typename T, typename... arg_types>
  //  void SetMethod(arg_types&&... args) {
  //    method_ = std::make_unique<T>(args...);
  //    method_->Init(*this);
  //  }

  // void Step(mfem::Vector& u, double& t, double& dt) { method_->Step(u, t, dt); }
};

class LinearSecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
  mfem::HypreParMatrix&            M_;
  mfem::HypreParMatrix&            C_;
  mfem::HypreParMatrix&            K_;
  mfem::Vector&                    f_;
  serac::BoundaryConditionManager& bcs_;
  serac::EquationSolver&           invMf_;
  serac::EquationSolver&           invT_;

  mutable mfem::Vector                  uf;
  mutable mfem::Vector                  duf_dt;
  mutable mfem::Vector                  uc;
  mutable mfem::Vector                  duc_dt;
  mutable mfem::Vector                  d2uc_dt2;
  mutable mfem::Vector                  uc_plus;
  mutable mfem::Vector                  uc_minus;
  std::unique_ptr<mfem::HypreParMatrix> T_;
  double                                previous_dt0;
  double                                previous_dt1;
  double                                epsilon;

  LinearSecondOrderODE(mfem::HypreParMatrix& M, mfem::HypreParMatrix& C, mfem::HypreParMatrix& K, mfem::Vector& f,
                       serac::BoundaryConditionManager& bcs, serac::EquationSolver& invMf, serac::EquationSolver& invT)
      : M_(M),
        C_(C),
        K_(K),
        f_(f),
        bcs_(bcs),
        invMf_(invMf),
        invT_(invT),
        previous_dt0(-1.0),
        previous_dt1(-1.0),
        epsilon(1.0e-4)
  {
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

    duc_dt   = (uc_plus - uc_minus) * (1.0 / (2.0 * epsilon));
    d2uc_dt2 = (uc_plus - 2 * uc + uc_minus) * (1.0 / (epsilon * epsilon));

    d2u_dt2 = invMf_ * (f_ - K_ * uf - K_ * uc - M_ * duc_dt);
  }

  void ImplicitSolve(const double dt0, const double dt1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    if (dt0 != previous_dt0 || dt1 != previous_dt1) {
      T_.reset(mfem::Add(1.0, M_, dt0, C_));
      T_.reset(mfem::Add(1.0, *T_, dt1 * dt1, K_));
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
    duc_dt   = (uc_plus - uc_minus) * (1.0 / (2.0 * epsilon));
    d2uc_dt2 = (uc_plus - 2 * uc + uc_minus) * (1.0 / (epsilon * epsilon));

    d2u_dt2 = invT_ * (f_ - M_ * d2uc_dt2 - K_ * uc - K_ * uf);

    previous_dt0 = dt0;
    previous_dt1 = dt1;
  }

  template <typename T, typename... arg_types>
  void SetMethod(arg_types&&... args)
  {
    method = std::make_unique<T>(args...);
    method->Init(*this);
  }

  void Step(mfem::Vector& u, double& t, double& dt) { method->Step(u, t, dt); }

  std::unique_ptr<mfem::ODESolver> method;
};

class SecondOrderODE : public mfem::SecondOrderTimeDependentOperator {
public:
  using explicit_signature = void(const double, const mfem::Vector&, const mfem::Vector&, mfem::Vector&);
  using implicit_signature = void(const double, const double, const double, const mfem::Vector&, const mfem::Vector&,
                                  mfem::Vector&);
  std::function<explicit_signature> explicit_func;
  std::function<implicit_signature> implicit_func;

  SecondOrderODE(int n) : mfem::SecondOrderTimeDependentOperator(n, 0.0){};
  void Mult(const mfem::Vector& u, const mfem::Vector& du_dt, mfem::Vector& d2u_dt2) const
  {
    explicit_func(t, u, du_dt, d2u_dt2);
  }
  void ImplicitSolve(const double dt0, const double dt1, const mfem::Vector& u, const mfem::Vector& du_dt,
                     mfem::Vector& d2u_dt2)
  {
    implicit_func(t, dt0, dt1, u, du_dt, d2u_dt2);
  }
};