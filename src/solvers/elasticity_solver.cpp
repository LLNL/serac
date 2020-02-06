// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "elasticity_solver.hpp"

ElasticitySolver::ElasticitySolver(int order, mfem::ParMesh *pmesh) :
  BaseSolver(), m_K_form(nullptr), m_l_form(nullptr), m_K_mat(nullptr), m_K_e_mat(nullptr), m_rhs(nullptr), 
  m_bc_rhs(nullptr), m_K_solver(nullptr), m_K_prec(nullptr),
  m_mu(nullptr), m_lambda(nullptr), m_body_force(nullptr)
{
  m_state.SetSize(1);
  m_state[0].mesh = pmesh;

  // Use vector-valued H1 nodal basis functions for the displacement field
  m_state[0].coll = new mfem::H1_FECollection(order, pmesh->Dimension());
  m_state[0].space = new mfem::ParFiniteElementSpace(pmesh, m_state[0].coll, pmesh->Dimension(), mfem::Ordering::byVDIM);

  // Initialize the grid function
  m_state[0].gf = new mfem::ParGridFunction(m_state[0].space);
  *m_state[0].gf = 0.0;

  // Initialize the true DOF vector
  m_state[0].true_vec = new mfem::HypreParVector(m_state[0].space);
  *m_state[0].true_vec = 0.0;
}

void ElasticitySolver::SetDisplacementBCs(mfem::Array<int> &disp_bdr, mfem::VectorCoefficient *disp_bdr_coef)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef);

  // Get the list of essential DOFs
  m_state[0].space->GetEssentialTrueDofs(disp_bdr, m_ess_tdof_list);
}

void ElasticitySolver::SetTractionBCs(mfem::Array<int> &trac_bdr, mfem::VectorCoefficient *trac_bdr_coef)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef);
}

void ElasticitySolver::SetLameParameters(mfem::Coefficient &lambda, mfem::Coefficient &mu)
{
  m_lambda = &lambda;
  m_mu = &mu;
}

void ElasticitySolver::SetBodyForce(mfem::VectorCoefficient &force)
{
  m_body_force = &force;
}

void ElasticitySolver::SetLinearSolverParameters(const LinearSolverParameters &params)
{
  m_lin_params = params;
}

void ElasticitySolver::CompleteSetup()
{
  MFEM_ASSERT(m_mu != nullptr, "Lame mu not set in ElasticitySolver!");
  MFEM_ASSERT(m_lambda != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  m_K_form = new mfem::ParBilinearForm(m_state[0].space);

  // Add the elastic integrator
  m_K_form->AddDomainIntegrator(new mfem::ElasticityIntegrator(*m_lambda, *m_mu));
  m_K_form->Assemble();
  m_K_form->Finalize();

  // Define the parallel linear form
  
  m_l_form = new mfem::ParLinearForm(m_state[0].space);

  // Add the traction integrator
  if (m_nat_bdr_vec_coef != nullptr) {
    m_l_form->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*m_nat_bdr_vec_coef), m_nat_bdr);
    m_rhs = m_l_form->ParallelAssemble();
  } else {
    m_rhs = new mfem::HypreParVector(m_state[0].space);
    *m_rhs = 0.0;
  }

  // Assemble the stiffness matrix
  m_K_mat = m_K_form->ParallelAssemble();

  // Eliminate the essential DOFs
  m_K_e_mat = m_K_mat->EliminateRowsCols(m_ess_tdof_list);

  // Initialize the eliminate BC RHS vector
  m_bc_rhs = new mfem::HypreParVector(m_state[0].space);
  *m_bc_rhs = 0.0;

  // Initialize the true vector
  m_state[0].gf->GetTrueDofs(*m_state[0].true_vec);

  if (m_lin_params.prec == Preconditioner::BoomerAMG) {
    MFEM_VERIFY(m_state[0].space->GetOrdering() == mfem::Ordering::byVDIM, "Attempting to use BoomerAMG with nodal ordering.");

    mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
    prec_amg->SetPrintLevel(m_lin_params.print_level);
    prec_amg->SetElasticityOptions(m_state[0].space);
    m_K_prec = prec_amg;

    mfem::GMRESSolver *K_gmres = new mfem::GMRESSolver(m_state[0].space->GetComm());
    K_gmres->SetRelTol(m_lin_params.rel_tol);
    K_gmres->SetAbsTol(m_lin_params.abs_tol);
    K_gmres->SetMaxIter(m_lin_params.max_iter);
    K_gmres->SetPrintLevel(m_lin_params.print_level);
    K_gmres->SetPreconditioner(*m_K_prec);
    m_K_solver = K_gmres;

  }
  // If not AMG, just MINRES with Jacobi smoothing
  else {
    mfem::HypreSmoother *K_hypreSmoother = new mfem::HypreSmoother;
    K_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    K_hypreSmoother->SetPositiveDiagonal(true);
    m_K_prec = K_hypreSmoother;

    mfem::MINRESSolver *K_minres = new mfem::MINRESSolver(m_state[0].space->GetComm());
    K_minres->SetRelTol(m_lin_params.rel_tol);
    K_minres->SetAbsTol(m_lin_params.abs_tol);
    K_minres->SetMaxIter(m_lin_params.max_iter);
    K_minres->SetPrintLevel(m_lin_params.print_level);
    K_minres->SetPreconditioner(*m_K_prec);
    m_K_solver = K_minres;
  }
}

void ElasticitySolver::AdvanceTimestep(__attribute__((unused)) double &dt)
{
  // Initialize the true vector
  m_state[0].gf->GetTrueDofs(*m_state[0].true_vec);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    mfem::mfem_error("Only quasistatics implemented for linear elasticity!");
  }

  // Distribute the shared DOFs
  m_state[0].gf->SetFromTrueDofs(*m_state[0].true_vec);
  m_cycle += 1;
}

// Solve the Quasi-static system
void ElasticitySolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  if (m_ess_bdr_vec_coef != nullptr) {
    m_ess_bdr_vec_coef->SetTime(m_time);
    m_state[0].gf->ProjectBdrCoefficient(*m_ess_bdr_vec_coef, m_ess_bdr);
    m_state[0].gf->GetTrueDofs(*m_state[0].true_vec);
    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, m_ess_tdof_list, *m_state[0].true_vec, *m_bc_rhs);
  }

  m_K_solver->SetOperator(*m_K_mat);

  m_K_solver->Mult(*m_bc_rhs, *m_state[0].true_vec);
}

ElasticitySolver::~ElasticitySolver()
{
  delete m_K_form;
  delete m_l_form;
  delete m_K_mat;
  delete m_K_e_mat;
  delete m_rhs;
  delete m_bc_rhs;
  delete m_K_solver;
  delete m_K_prec;
}
