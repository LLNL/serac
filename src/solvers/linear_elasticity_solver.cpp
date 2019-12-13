// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "linear_elasticity_solver.hpp"

LinearElasticSolver::LinearElasticSolver(mfem::ParFiniteElementSpace &fes,
				         mfem::Array<int> &ess_bdr,
                                         mfem::Array<int> &trac_bdr,
                                         mfem::Coefficient &mu,
                                         mfem::Coefficient &lambda,
                                         mfem::VectorCoefficient &trac,
                                         double rel_tol,
                                         double abs_tol,
                                         int iter,
                                         bool amg,
                                         bool slu)
: m_fe_space(fes), m_Kform(nullptr), m_lform(nullptr), m_K_solver(nullptr), m_K_prec(nullptr)
{
   // Define the parallel bilinear form 
   m_Kform = new mfem::ParBilinearForm(&m_fe_space);

   // Add the elastic integrator
   m_Kform->AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda, mu));

   // Define the parallel linear form
   m_lform = new mfem::ParLinearForm(&fes);

   // Add the traction integrator
   m_lform->AddBdrFaceIntegrator(new mfem::VectorBoundaryLFIntegrator(trac), trac_bdr);

   m_fe_space.GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list);

   m_lform->Assemble();
   m_Kform->Assemble();

   if (amg) {
      MFEM_VERIFY(m_fe_space.GetOrdering() == mfem::Ordering::byVDIM, "Attempting to use BoomerAMG with nodal ordering.");

      mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&m_fe_space);
      m_K_prec = prec_amg;

      mfem::GMRESSolver *K_gmres = new mfem::GMRESSolver(m_fe_space.GetComm());
      K_gmres->SetRelTol(rel_tol);
      K_gmres->SetAbsTol(abs_tol);
      K_gmres->SetMaxIter(iter);
      K_gmres->SetPrintLevel(0);
      K_gmres->SetPreconditioner(*m_K_prec);
      m_K_solver = K_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      mfem::SuperLUSolver *superlu = nullptr;
      superlu = new mfem::SuperLUSolver(m_fe_space.GetComm());
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(mfem::superlu::PARMETIS);
      
      m_K_solver = superlu;
      m_K_prec = nullptr;
   }
   else {
      mfem::HypreSmoother *K_hypreSmoother = new mfem::HypreSmoother;
      K_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
      K_hypreSmoother->SetPositiveDiagonal(true);
      m_K_prec = K_hypreSmoother;

      mfem::MINRESSolver *K_minres = new mfem::MINRESSolver(m_fe_space.GetComm());
      K_minres->SetRelTol(rel_tol);
      K_minres->SetAbsTol(abs_tol);
      K_minres->SetMaxIter(iter);
      K_minres->SetPrintLevel(0);
      K_minres->SetPreconditioner(*m_K_prec);
      m_K_solver = K_minres;

   }
}

// Solve the Newton system
bool LinearElasticSolver::Solve(mfem::Vector &x) const
{

   mfem::HypreParMatrix A;
   mfem::Vector B, X;

   m_Kform->FormLinearSystem(m_ess_tdof_list, x, *m_lform, A, X, B);
   m_K_solver->SetOperator(A);
   m_K_solver->Mult(B, X);
   m_Kform->RecoverFEMSolution(X, *m_lform, x);

   return true;
}

LinearElasticSolver::~LinearElasticSolver()
{
   delete m_K_solver;
   if (m_K_prec != nullptr) {
      delete m_K_prec;
   }
}
