// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#ifndef DYNAMIC_SOLVER
#define DYNAMIC_SOLVER

#include "mfem.hpp"

class ReducedSystemOperator;

// After spatial discretization, the hyperelastic model can be written as a
//  system of ODEs:
//     dv/dt = -M^{-1}*(H(x) + S*v)
//     dx/dt = v,
//  where x is the vector representing the deformation, v is the velocity field,
//  M is the mass matrix, S is the viscosity matrix, and H(x) is the nonlinear
//  hyperelastic operator.
//
//  Class HyperelasticOperator represents the right-hand side of the above
//  system of ODEs. 
class DynamicSolver : public mfem::TimeDependentOperator
{
protected:
   mfem::ParFiniteElementSpace &m_fe_space;
   mfem::ParNonlinearForm *m_H_form;
   mfem::ParBilinearForm *m_M_form;
   mfem::ParBilinearForm *m_S_form;

   mfem::Coefficient &m_viscosity;

   mutable mfem::Operator *m_jacobian;
   
   mfem::HypreParMatrix *m_M_mat; // Mass matrix from ParallelAssemble()
   mfem::CGSolver m_M_solver;    // Krylov solver for inverting the mass matrix M
   mfem::HypreSmoother m_M_prec; // Preconditioner for the mass matrix M
   
   // Nonlinear operator defining the reduced backward Euler equation for the
   // velocity. Used in the implementation of method ImplicitSolve.
   ReducedSystemOperator *m_reduced_oper;
   
   /// Newton solver for the operator
   mfem::NewtonSolver m_newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   mfem::Solver *m_J_solver;
   /// Preconditioner for the Jacobian
   mfem::Solver *m_J_prec;
   /// nonlinear material model 
   mfem::HyperelasticModel *m_model;
   /// essential degrees of freedom list
   mfem::Array<int> m_ess_tdof_list;   
   /// working vector
   mutable mfem::Vector m_z;
   
public:
   DynamicSolver(mfem::ParFiniteElementSpace &fes,
                 mfem::Array<int> &ess_bdr,
                 double mu,
                 double K,
                 mfem::Coefficient &visc,
                 double rel_tol,
                 double abs_tol,
                 int iter,
                 bool gmres,
                 bool slu);

   /// Required to use the native newton solver
   virtual void Mult(const mfem::Vector &vx, mfem::Vector &dvx_dt) const;
   /// Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   /// This is the only requirement for high-order SDIRK implicit integration.
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &k);

   /// Get FE space
   const mfem::ParFiniteElementSpace *GetFESpace() { return &m_fe_space; }

   virtual ~DynamicSolver();
};

//  Nonlinear operator of the form:
//  k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
//  where M and S are given BilinearForms, H is a given NonlinearForm, v and x
//  are given vectors, and dt is a scalar. 
class ReducedSystemOperator : public mfem::Operator
{
private:
   mfem::ParBilinearForm *m_M_form;
   mfem::ParBilinearForm *m_S_form;
   mfem::ParNonlinearForm *m_H_form;
   mutable mfem::HypreParMatrix *m_jacobian;
   double m_dt;
   const mfem::Vector *m_v, *m_x;
   mutable mfem::Vector m_w, m_z;
   const mfem::Array<int> &m_ess_tdof_list;

public:
   ReducedSystemOperator(mfem::ParBilinearForm *M, mfem::ParBilinearForm *S,
                         mfem::ParNonlinearForm *H, const mfem::Array<int> &ess_tdof_list);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt, const mfem::Vector *v, const mfem::Vector *x);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).   
   virtual mfem::Operator &GetGradient(const mfem::Vector &k) const;

   virtual ~ReducedSystemOperator();
};

#endif
