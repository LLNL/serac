// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "dynamic_solver.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"

using namespace mfem;

DynamicSolver::DynamicSolver(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             Array<int> &trac_bdr,
                                             double mu,
                                             double K,
                                             VectorCoefficient &trac_coef,
                                             double rel_tol,
                                             double abs_tol,
                                             int iter,
                                             bool gmres,
                                             bool slu)
: Operator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;

   // Define the paral2lel nonlinear form 
   Hform = new ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs); 

   // Define the material model
   model = new NeoHookeanModel(mu, K);   

   // Add the hyperelastic integrator
   Hform->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model));

   // Add the traction integrator
   Hform->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(trac_coef), trac_bdr);
   
   if (gmres) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      SuperLUSolver *superlu = NULL;
      superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      
      J_solver = superlu;
      J_prec = NULL;
   }
   else {
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(rel_tol);
      J_minres->SetAbsTol(0.0);
      J_minres->SetMaxIter(300);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;

   }

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

// Solve the Newton system
int DynamicSolver::Solve(Vector &x) const
{
   Vector zero;
   newton_solver.Mult(zero, x);

   return newton_solver.GetConverged();
}

// compute: y = H(x,p)
void DynamicSolver::Mult(const Vector &k, Vector &y) const
{
   // Apply the nonlinear form
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
Operator &DynamicSolver::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}


DynamicSolver::~DynamicSolver()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}
