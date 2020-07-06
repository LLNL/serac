// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "elasticity_solver.hpp"

const int num_fields = 1;

ElasticitySolver::ElasticitySolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields),
      displacement(m_state[0]),
      m_K_form(nullptr),
      m_l_form(nullptr),
      m_K_mat(nullptr),
      m_K_e_mat(nullptr),
      m_rhs(nullptr),
      m_bc_rhs(nullptr),
      m_K_solver(nullptr),
      m_K_prec(nullptr),
      m_mu(nullptr),
      m_lambda(nullptr),
      m_body_force(nullptr)
{
  displacement.mesh = pmesh;
  displacement.coll = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension(), mfem::Ordering::byVDIM);
  displacement.space =
      std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), displacement.coll.get(), pmesh->Dimension());
  displacement.gf       = std::make_shared<mfem::ParGridFunction>(displacement.space.get());
  displacement.true_vec = mfem::HypreParVector(displacement.space.get());

  // and initial conditions
  *displacement.gf      = 0.0;
  displacement.true_vec = 0.0;

  displacement.name = "displacement";
}

void ElasticitySolver::SetDisplacementBCs(std::vector<int> &                       disp_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef, int component)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef, *displacement.space, component);
}

void ElasticitySolver::SetTractionBCs(std::vector<int> &                       trac_bdr,
                                      std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void ElasticitySolver::SetLameParameters(mfem::Coefficient &lambda, mfem::Coefficient &mu)
{
  m_lambda = &lambda;
  m_mu     = &mu;
}

void ElasticitySolver::SetBodyForce(mfem::VectorCoefficient &force) { m_body_force = &force; }

void ElasticitySolver::SetLinearSolverParameters(const LinearSolverParameters &params) { m_lin_params = params; }

void ElasticitySolver::CompleteSetup()
{
  MFEM_ASSERT(m_mu != nullptr, "Lame mu not set in ElasticitySolver!");
  MFEM_ASSERT(m_lambda != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  m_K_form = new mfem::ParBilinearForm(displacement.space.get());

  // Add the elastic integrator
  m_K_form->AddDomainIntegrator(new mfem::ElasticityIntegrator(*m_lambda, *m_mu));
  m_K_form->Assemble();
  m_K_form->Finalize();

  // Define the parallel linear form

  m_l_form = new mfem::ParLinearForm(displacement.space.get());

  // Add the traction integrator
  if (m_nat_bdr.size() > 0) {
    for (auto &nat_bc_data : m_nat_bdr) {
      m_l_form->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*nat_bc_data->vec_coef),
                                      nat_bc_data->markers);
    }
    m_l_form->Assemble();
    m_rhs = m_l_form->ParallelAssemble();
  } else {
    m_rhs  = new mfem::HypreParVector(displacement.space.get());
    *m_rhs = 0.0;
  }

  // Assemble the stiffness matrix
  m_K_mat = m_K_form->ParallelAssemble();

  // Eliminate the essential DOFs
  for (auto &bc : m_ess_bdr) {
    m_K_e_mat = m_K_mat->EliminateRowsCols(bc->true_dofs);
  }

  // Initialize the eliminate BC RHS vector
  m_bc_rhs  = new mfem::HypreParVector(displacement.space.get());
  *m_bc_rhs = 0.0;

  // Initialize the true vector
  displacement.gf->GetTrueDofs(displacement.true_vec);

  if (m_lin_params.prec == Preconditioner::BoomerAMG) {
    MFEM_VERIFY(displacement.space->GetOrdering() == mfem::Ordering::byVDIM,
                "Attempting to use BoomerAMG with nodal ordering.");

    mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
    prec_amg->SetPrintLevel(m_lin_params.print_level);
    prec_amg->SetElasticityOptions(displacement.space.get());
    m_K_prec = prec_amg;

    mfem::GMRESSolver *K_gmres = new mfem::GMRESSolver(displacement.space->GetComm());
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

    mfem::MINRESSolver *K_minres = new mfem::MINRESSolver(displacement.space->GetComm());
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
  displacement.gf->GetTrueDofs(displacement.true_vec);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    mfem::mfem_error("Only quasistatics implemented for linear elasticity!");
  }

  // Distribute the shared DOFs
  displacement.gf->SetFromTrueDofs(displacement.true_vec);
  m_cycle += 1;
}

// Solve the Quasi-static system
void ElasticitySolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  for (auto &bc : m_ess_bdr) {
    bc->vec_coef->SetTime(m_time);
    displacement.gf->ProjectBdrCoefficient(*bc->vec_coef, bc->markers);
    displacement.gf->GetTrueDofs(displacement.true_vec);
    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, bc->true_dofs, displacement.true_vec, *m_bc_rhs);
  }

  m_K_solver->SetOperator(*m_K_mat);

  m_K_solver->Mult(*m_bc_rhs, displacement.true_vec);
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
