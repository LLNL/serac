// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

//============================================================================//
// Override the MFEM NewtonSolver in linalg/solvers.*pp with a Newton solver
// that has a line search. The line search determines an appropriate step 
// size for the update, which in the MFEM implementation is always assumed to 
// be the norm of the original update. A smaller step is expected to give 
// better convergence behavior at the cost of a more expensive update.
//
// This is copied from the orignal internal LiDO 1.0 nonlinear forward solver 
// developed by Seth Watts.
//============================================================================//
class LineSearchNewtonSolver : public mfem::NewtonSolver
{
   
public:

   // Three line searches are defined beyond the default behavior of using step
   // length 1.0.
   // 
   // In the Armijo backtracking line search, the inital step size of 1.2 is 
   // reduced recursively by a factor of tau in (0,1) until satisfying 
   //        r(x) - r(x + alpha_j p) >= -alpha_j sigma m
   // where the step length at iteration j is alpha_j, x is the parameter state,
   // p is the search direction, m = p^t \grad f is the slope along the search
   // direction, and sigma in (0,1) is a control parameter.
   // 
   // In the Golden section search, an intial interval of [0, 1] is reduced
   // recursively by sampling two points within the interval and selecting one
   // as the new endpoint for a reduced-size interval. The decision of where to
   // place the sampling points is made such that they have the same location
   // proportional to the interval size as the previous step. This iteration
   // converges when the size of the interval falls below a defined tolerance.
   // 
   // In the greedy forward tracking line search, an initially small step size
   // is increased by a factor of tau > 1 until the residual function stops
   // decreasing, i.e. it takes the largest step possible until doing so
   // increases the redisual relative to the previous trial value.


   enum LineSearchType { NoLineSearch          = 0,
                         ArmijoBacktracking    = 1,
                         GoldenSectionSearch   = 2,
                         GreedyForwardTracking = 3 };

   LineSearchType line_search_; // search type

   mutable double t_, tau_, sigma_; // backtracking line search params & vars
   
   mutable double total_solve_time_, worst_solve_time_,
                  total_form_time_,  worst_form_time_;

   mutable int total_iterations_, worst_iterations_;
   
   mutable bool lin_solve_converged_;

   LineSearchNewtonSolver(MPI_Comm comm); 
   
   // Override the MFEM base class NewtonSolver's SetSolver method to also
   // cast the passed in Solver to an IterativeSolver, and throw an error
   // if this cast is unsuccessful. This cast allows us to query the solver
   // for number of iterations to convergence. The restriction to only
   // iterative methods is believed not to be a problem since we always use
   // iterative solvers in pratice.
   void SetSolver(mfem::Solver &solver);
   
   double ComputeScalingFactor(const mfem::Vector &x,
                               const mfem::Vector &b) const;
   
   void SetLineSearch(const LineSearchType line_search);
   
   void SetSigmaTerm(const double sigma); 
   
   void SetTauTerm(const double tau);
   
   double ComputeTrialResidual(const double alpha,
                               const mfem::Vector &x) const;
   
   // Overload the Mult method so we can pull out solver timing.
   void Mult(const mfem::Vector &b, mfem::Vector &x) const;
   
   double GetTotalFormTime() const { return total_form_time_; }
   double GetWorstFormTime() const { return worst_form_time_; }
   
   double GetTotalSolveTime() const { return total_solve_time_; }
   double GetWorstSolveTime() const { return worst_solve_time_; }
   
   int GetTotalIterations() const { return total_iterations_; }
   int GetWorstIterations() const { return worst_iterations_; }
   
   bool GetLinSolveConverged() const { return lin_solve_converged_; }

protected:

   mfem::IterativeSolver *prec; // overload prec as IterativeSolver vice Solver
};
