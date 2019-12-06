// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "linesearch_newton.hpp"

//============================================================================//
// Implement backtracking line search to set the step size for the MFEM Newton-
// Rhaphson solver using the overridden ComputeScalingFactor function.
//============================================================================//

LineSearchNewtonSolver::LineSearchNewtonSolver(MPI_Comm comm)
   : mfem::NewtonSolver(comm), 
   total_solve_time_(0.0), 
   worst_solve_time_(0.0), 
   total_form_time_(0.0),
   worst_form_time_(0.0), 
   total_iterations_(0), 
   worst_iterations_(0), 
   line_search_(NoLineSearch), 
   lin_solve_converged_(true) 
{ } 

// Override the SetSolver method to cast an MFEM Solver to an IterativeSolver.
// Throw and error if the cast is not successful.
void LineSearchNewtonSolver::SetSolver(mfem::Solver &solver)
{
   prec = dynamic_cast<mfem::IterativeSolver*> (&solver);
   MFEM_VERIFY(prec != NULL, "Cast of solver to iterative solver failed!");
}

void LineSearchNewtonSolver::SetLineSearch(const LineSearchType line_search)
{
   line_search_ = line_search;
}

void LineSearchNewtonSolver::SetSigmaTerm(const double sigma)
{
   sigma_ = sigma;
}
   
void LineSearchNewtonSolver::SetTauTerm(const double tau)
{
   tau_ = tau;
}

double LineSearchNewtonSolver::ComputeTrialResidual(const double alpha,
                                               const mfem::Vector &x) const
{
   mfem::Vector xPlusAlphaP, resid;
   xPlusAlphaP.SetSize(x.Size());
   resid.SetSize(x.Size());
   add(x, -alpha, c, xPlusAlphaP);
   oper->Mult(xPlusAlphaP, resid);
   return Norm(resid);
}

double LineSearchNewtonSolver::ComputeScalingFactor(const mfem::Vector &x,
                                               const mfem::Vector &b) const
{
   if ( line_search_ == ArmijoBacktracking )
   {
      t_ = sigma_ * pow(Norm(c), 2);
   
      double alpha = 1.2;
   
      double fAtX = ComputeTrialResidual(0.0, x);
      double fAtXPlusAlphaP = ComputeTrialResidual(alpha, x);
   
      int j = 0;
      while ( (fAtXPlusAlphaP > (fAtX - alpha * t_)) && (alpha > 0.01) )
      {
         j++;
         alpha *= tau_;
         fAtXPlusAlphaP = ComputeTrialResidual(alpha, x);
         if (print_level >= 0)
         {
            std::cout << "                " 
                      << "j " << std::setw(2) << j << " residual " 
                      << fAtXPlusAlphaP << "\n" << std::flush;
         }
      }
         
      if (print_level >= 0)
      {
         std::cout << "                " 
                   << "Step length set to " << alpha << "\n";
      }
   
      return alpha;
   }
   
   else if ( line_search_ == GoldenSectionSearch ) 
   {
   
      tau_ = 0.6180339887498949; // golden ratio = (sqrt(5) - 1)/2
   
      // *------------*--------*------------*
      // a            b        g            d
   
      double alpha = 0.0;
      double delta = 1.0;
      double beta  = alpha + (1.0 - tau_) * (delta - alpha);
      double gamma = alpha + (   tau_   ) * (delta - alpha);
      double f_beta  = ComputeTrialResidual(beta,  x);
      double f_gamma = ComputeTrialResidual(gamma, x);
   
      double tol = (delta - alpha) * 1e-3;
   
      while ( (delta - alpha) > tol )
      {
         if ( f_beta > f_gamma )
         {
            alpha   = beta;
            beta    = gamma;
            f_beta  = f_gamma;
            gamma   = alpha + (   tau_   ) * (delta - alpha);
            f_gamma = ComputeTrialResidual(gamma, x);
         }
         else
         {
            delta   = gamma;
            gamma   = beta;
            f_gamma = f_beta;
            beta    = alpha + (1.0 - tau_) * (delta - alpha);
            f_beta  = ComputeTrialResidual(beta,  x);
         }
      }
   
      if (print_level >= 0)
      {
         std::cout << "                " 
                   << "Step length set to " << gamma << "\n";
      }
   
      return gamma;
   }
   
   if ( line_search_ == GreedyForwardTracking )
   {
      double alpha = 0.001;
   
      double fAtX = ComputeTrialResidual(0.0, x);
      double fAtXPlusAlphaP = ComputeTrialResidual(alpha, x);
   
      int j = 0;
      while ( (fAtXPlusAlphaP < fAtX) )
      {
         j++;
         alpha /= tau_;
         fAtX = fAtXPlusAlphaP;
         fAtXPlusAlphaP = ComputeTrialResidual(alpha, x);
         if (print_level >= 0)
         {
            std::cout << "                " 
                      << "j " << std::setw(2) << j << " residual " 
                      << fAtXPlusAlphaP << "\n" << std::flush;
         }
      }
      if ( j > 0 ) { alpha *= tau_; } // undo the last change in alpha
   
      if (print_level >= 0)
      {
         std::cout << "                " 
                   << "Step length set to " << alpha << "\n";
      }
   
      return alpha;
   }
   
   else // catch-all, and for line_search_ == NoLineSearch
   {
      return 1.0;
   }
}

void LineSearchNewtonSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal, normI;
   const bool have_b = (b.Size() == Height());

   if (!iterative_mode)
   {
      x = 0.0;
   }

   oper->Mult(x, r);
   if (have_b)
   {
      r -= b;
   }

   normI = norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(!std::isnan(norm) && !std::isinf(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Newton iter " << std::setw(3) << it
                   << " : |r| = " << std::scientific  
                   << std::setprecision(4) << norm;
         if (it > 0)
         {
            mfem::out << ", |r|/|r_0| = " << std::scientific 
                      << std::setprecision(3) << norm/norm0;
            mfem::out << ", |r|/|r_i| = " << std::scientific
                      << std::setprecision(3) << norm/normI;
         }
         mfem::out << '\n';
      }

      if (norm <= norm_goal)
      {
         converged = 1;
         break;
      }

      if (it >= max_iter)
      {
         converged = 0;
         break;
      }

      // Form the linear system by getting the current tangent stiffness matrix.
      mfem::StopWatch chrono_form;
      chrono_form.Clear();
      chrono_form.Start();
      prec->SetOperator(oper->GetGradient(x));
      chrono_form.Stop();
      total_form_time_ += chrono_form.RealTime();
      worst_form_time_  = std::max(worst_form_time_, chrono_form.RealTime());

      // Solve the linear system.
      mfem::StopWatch chrono_solve;
      chrono_solve.Clear();
      chrono_solve.Start();
      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]
      chrono_solve.Stop();
      total_solve_time_ += chrono_solve.RealTime();
      worst_solve_time_  = std::max(worst_solve_time_, chrono_solve.RealTime());
      total_iterations_ += prec->GetNumIterations();
      worst_iterations_  = std::max(worst_iterations_,prec->GetNumIterations());
      
      //MFEM_VERIFY(prec->GetConverged(), 
      //   "NewtonSolver's linear solver did not converge!");
      
      const double c_scale = ComputeScalingFactor(x, b);
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      add(x, -c_scale, c, x);

      oper->Mult(x, r);
      if (have_b)
      {
         r -= b;
      }
      normI = norm;
      norm = Norm(r);
      
      // Check that the linear solver exited happily. If not, abort this Newton
      // solve and return to the adaptive load stepping routine to attempt the
      // solve at a lower load level. Check involves seeing how the linear solve
      // exited, and whether the resulting solution has inf or Nan norm.
      lin_solve_converged_ = true; // prec->GetConverged();
      if ( !lin_solve_converged_ || std::isnan(norm) || std::isinf(norm) )
      {
         converged = 0;
         break;
      }
   }

   final_iter = it;
   final_norm = norm;
}



