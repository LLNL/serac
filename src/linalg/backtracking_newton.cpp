// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "backtracking_newton.hpp"

void BacktrackingNewtonSolver::SetOperator(const mfem::Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width);
   c.SetSize(width);
   newres.SetSize(width);
   test.SetSize(width);
}

void BacktrackingNewtonSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_goal;
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

   norm0 = norm = Norm(r);
   norm_goal = std::max(rel_tol*norm, abs_tol);

   prec->iterative_mode = false;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "Newton iteration " << std::setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0)
         {
            mfem::out << ", ||r||/||r_0|| = " << norm/norm0;
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

      prec->SetOperator(oper->GetGradient(x));

      prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]

      const double c_scale = ComputeScalingFactor(x, c, b, r);
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
      norm = Norm(r);
   }

   final_iter = it;
   final_norm = norm;
}

double BacktrackingNewtonSolver::ComputeScalingFactor(const mfem::Vector &x, const mfem::Vector &dir, const mfem::Vector  &b, const mfem::Vector &res) const
{
/*
   const bool have_b = (b.Size() == Height());
   double size = 1.0;
   for (int steps = 0; steps < maxsteps; steps++) {
      add(x, -1.0*size, dir, test);
      oper->Mult(test, newres);
      if (have_b) {
	 add(newres, -1.0, b, newres); 
      }
      if(Norm(res) > Norm(newres)) {
	 break;
      }  
      else {
	 size*=alpha;
      }
   }
   return size;
*/
   return 1.0;
}
