// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#ifndef DYNAMIC_SOLVER
#define DYNAMIC_SOLVER

#include "mfem.hpp"

using namespace mfem;

class ReducedSystemOperator;

/** After spatial discretization, the hyperelastic model can be written as a
 *  system of ODEs:
 *     dv/dt = -M^{-1}*(H(x))
 *     dx/dt = v,
 *  where x is the vector representing the deformation, v is the velocity field,
 *  M is the mass matrix, and H(x) is the nonlinear
 *  hyperelastic operator.
 *
 *  Class HyperelasticOperator represents the right-hand side of the above
 *  system of ODEs. */
class DynamicSolver : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fe_space;
   ParNonlinearForm *Hform;
   ParBilinearForm *Mform;
   
   mutable Operator *Jacobian;
   const Vector *x;
   
   HypreParMatrix *Mmat; // Mass matrix from ParallelAssemble()
   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M
   
   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   ReducedSystemOperator *reduced_oper;
   
   /// Newton solver for the operator
   NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// nonlinear material model 
   HyperelasticModel *model;

   mutable Vector z;
   
public:
   DynamicSolver(ParFiniteElementSpace &fes,
                 Array<int> &ess_bdr,
                 double mu,
                 double K,
                 double rel_tol,
                 double abs_tol,
                 int iter,
                 bool gmres,
                 bool slu);

   /// Required to use the native newton solver
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;
   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   /// Get FE space
   const ParFiniteElementSpace *GetFESpace() { return &fe_space; }

   virtual ~DynamicSolver();
};

/** Nonlinear operator of the form:
    k --> (M)*k + H(x + dt*v + dt^2*k),
    where M is the given BilinearForm, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class ReducedSystemOperator : public Operator
{
private:
   ParBilinearForm *M;
   ParNonlinearForm *H;
   mutable HypreParMatrix *Jacobian;
   double dt;
   const Vector *v, *x;
   mutable Vector w, z;
   const Array<int> &ess_tdof_list;

public:
   ReducedSystemOperator(ParBilinearForm *M_, 
                         ParNonlinearForm *H_, const Array<int> &ess_tdof_list);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k.
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();
};

#endif
