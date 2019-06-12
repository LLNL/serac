#ifndef HYPERELASTIC_SOLVER
#define HYPERELASTIC_SOLVER

#include "mfem.hpp"

using namespace mfem;

class NonlinearMechOperator : public Operator
{
protected:
   ParFiniteElementSpace &fe_space;
   ParNonlinearForm *Hform;
   
   mutable Operator *Jacobian;
   const Vector *x;

   /// Newton solver for the operator
   NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// nonlinear material model 
   HyperelasticModel *model;

public:
   NonlinearMechOperator(ParFiniteElementSpace &fes,
                         Array<int> &ess_bdr,
                         Array<int> &trac_bdr,                         
                         double mu,
                         double K,
                         VectorCoefficient &trac_coef,
                         double rel_tol,
                         double abs_tol,
                         int iter,
                         bool gmres,
                         bool slu);

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   int Solve(Vector &x) const;

   /// Get FE space
   const ParFiniteElementSpace *GetFESpace() { return &fe_space; }

   virtual ~NonlinearMechOperator();
};

#endif
