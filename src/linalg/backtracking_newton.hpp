// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

/// Backtracking Newton's method for solving F(x)=b for a given operator F.
/// This is essentially the Newton's method from MFEM with a more sensible 
/// compute scaling factor call signature
class BacktrackingNewtonSolver : public mfem::IterativeSolver
{
protected:
   mutable mfem::Vector m_r, m_c, m_newres, m_test;
   double m_alpha;
   double m_maxsteps;

public:
   BacktrackingNewtonSolver(MPI_Comm comm, double al = 0.5, double ms = 3) : mfem::IterativeSolver(comm),
      m_alpha(al), m_maxsteps(ms) { }
   
   virtual void SetOperator(const mfem::Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   virtual void SetSolver(mfem::Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /// If `b.Size() != Height()`, then @a b is assumed to be zero.
   virtual void Mult(const mfem::Vector &b, mfem::Vector &x) const;

   /// Compute the backtracking scaling factor where x is the current state, dir is the search direction,
   /// and b is the RHS.
   virtual double ComputeScalingFactor(const mfem::Vector &x, const mfem::Vector &dir, const mfem::Vector &b,
	 const mfem::Vector &res) const;
};

