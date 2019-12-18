// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "quasistatic_solver.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"

QuasistaticSolver::QuasistaticSolver(mfem::ParFiniteElementSpace &fes,
                                     mfem::Array<int> &ess_bdr,
                                     mfem::Array<int> &trac_bdr,
                                     double mu,
                                     double K,
                                     mfem::VectorCoefficient &trac_coef,
                                     double rel_tol,
                                     double abs_tol,
                                     int iter,
                                     bool gmres,
                                     bool slu)
: mfem::Operator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   // Define the paral2lel nonlinear form 
   Hform = new mfem::ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr); 

   // Define the material model
   model = new mfem::NeoHookeanModel(mu, K);   

   // Add the hyperelastic integrator
   Hform->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model));

   // Add the traction integrator
   Hform->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(trac_coef), trac_bdr);
   
   if (gmres) {
      MFEM_VERIFY(fe_space.GetOrdering() == mfem::Ordering::byVDIM, "Attempting to use BoomerAMG with nodal ordering.");

      mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      mfem::GMRESSolver *J_gmres = new mfem::GMRESSolver(fe_space.GetComm());
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      mfem::SuperLUSolver *superlu = NULL;
      superlu = new mfem::SuperLUSolver(fe_space.GetComm());
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(mfem::superlu::PARMETIS);
      
      J_solver = superlu;
      J_prec = NULL;
   }
   else {
      mfem::HypreSmoother *J_hypreSmoother = new mfem::HypreSmoother;
      J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      mfem::MINRESSolver *J_minres = new mfem::MINRESSolver(fe_space.GetComm());
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
   newton_solver.SetLineSearch(LineSearchNewtonSolver::NoLineSearch);
   newton_solver.SetSigmaTerm(0.5);
   newton_solver.SetTauTerm(5.0);
}

// Solve the Newton system
bool QuasistaticSolver::Solve(mfem::Vector &x) const
{
   mfem::Vector zero;
   newton_solver.Mult(zero, x);

   return (newton_solver.GetConverged() == 1);
}

// compute: y = H(x,p)
void QuasistaticSolver::Mult(const mfem::Vector &k, mfem::Vector &y) const
{
   // Apply the nonlinear form
   Hform->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
mfem::Operator &QuasistaticSolver::GetGradient(const mfem::Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}


QuasistaticSolver::~QuasistaticSolver()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}
