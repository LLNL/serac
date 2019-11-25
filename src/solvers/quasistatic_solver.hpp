// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef QUASISTATIC_SOLVER
#define QUASISTATIC_SOLVER

#include "mfem.hpp"

class QuasistaticSolver : public mfem::Operator
{
protected:
   mfem::ParFiniteElementSpace &fe_space;
   mfem::ParNonlinearForm *Hform;

   mutable mfem::Operator *Jacobian;
   const mfem::Vector *x;

   /// Newton solver for the operator
   mfem::NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   mfem::Solver *J_solver;
   /// Preconditioner for the Jacobian
   mfem::Solver *J_prec;
   /// nonlinear material model
   mfem::HyperelasticModel *model;

public:
   QuasistaticSolver(mfem::ParFiniteElementSpace &fes,
                       mfem::Array<int> &ess_bdr,
                       mfem::Array<int> &trac_bdr,
                       double mu,
                       double K,
                       mfem::VectorCoefficient &trac_coef,
                       double rel_tol,
                       double abs_tol,
                       int iter,
                       bool gmres,
                       bool slu);

   /// Required to use the native newton solver
   virtual mfem::Operator &GetGradient(const mfem::Vector &x) const;
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const;

   /// Driver for the newton solver
   int Solve(mfem::Vector &x) const;

   /// Get FE space
   const mfem::ParFiniteElementSpace *GetFESpace() { return &fe_space; }

   virtual ~QuasistaticSolver();
};

#endif
