// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

/// Backtracking Newton's method for solving F(x)=b for a given operator F.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class BacktrackingNewtonSolver : public mfem::IterativeSolver
{
protected:
   mutable mfem::Vector r, c, newres, test;
   double alpha;
   double maxsteps;

public:
   BacktrackingNewtonSolver(MPI_Comm _comm, double al = 0.5, double ms = 5) : mfem::IterativeSolver(_comm),
      alpha(al), maxsteps(ms) { }
   
   virtual void SetOperator(const mfem::Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** This method is equivalent to calling SetPreconditioner(). */
   virtual void SetSolver(mfem::Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const mfem::Vector &b, mfem::Vector &x) const;

   /** @brief This method can be overloaded in derived classes to implement line
       search algorithms. */
   /** The base class implementation (BacktrackingNewtonSolver) simply returns 1. A return
       value of 0 indicates a failure, interrupting the BacktrackingNewton iteration. */
   virtual double ComputeScalingFactor(const mfem::Vector &x, const mfem::Vector &dir, const mfem::Vector &b,
		  const mfem::Vector &res) const;
};

