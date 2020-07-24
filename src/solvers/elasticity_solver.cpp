// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "elasticity_solver.hpp"

#include "common/logger.hpp"

const int num_fields = 1;

ElasticitySolver::ElasticitySolver(const int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields),
      displacement(m_state[0]),
      m_mu(nullptr),
      m_lambda(nullptr),
      m_body_force(nullptr)
{
  displacement->mesh = pmesh;
  displacement->coll = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension(), mfem::Ordering::byVDIM);
  displacement->space =
      std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), displacement->coll.get(), pmesh->Dimension());
  displacement->gf       = std::make_shared<mfem::ParGridFunction>(displacement->space.get());
  displacement->true_vec = std::make_shared<mfem::HypreParVector>(displacement->space.get());

  // and initial conditions
  *displacement->gf       = 0.0;
  *displacement->true_vec = 0.0;

  displacement->name = "displacement";
}

void ElasticitySolver::SetDisplacementBCs(const std::set<int> &                       disp_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef, int component)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef, *displacement->space, component);
}

void ElasticitySolver::SetTractionBCs(const std::set<int> &                       trac_bdr,
                                      std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void ElasticitySolver::SetLameParameters(mfem::Coefficient &lambda, mfem::Coefficient &mu)
{
  m_lambda = &lambda;
  m_mu     = &mu;
}

void ElasticitySolver::SetBodyForce(const mfem::VectorCoefficient &force) { m_body_force = &force; }

void ElasticitySolver::SetLinearSolverParameters(const serac::LinearSolverParameters &params) { m_lin_params = params; }

void ElasticitySolver::CompleteSetup()
{
  SLIC_ASSERT_MSG(m_mu != nullptr, "Lame mu not set in ElasticitySolver!");
  SLIC_ASSERT_MSG(m_lambda != nullptr, "Lame lambda not set in ElasticitySolver!");

  // Define the parallel bilinear form
  m_K_form = std::make_unique<mfem::ParBilinearForm>(displacement->space.get());

  // Add the elastic integrator
  m_K_form->AddDomainIntegrator(new mfem::ElasticityIntegrator(*m_lambda, *m_mu));
  m_K_form->Assemble();
  m_K_form->Finalize();

  // Define the parallel linear form

  m_l_form = std::make_unique<mfem::ParLinearForm>(displacement->space.get());

  // Add the traction integrator
  if (m_nat_bdr.size() > 0) {
    for (auto &nat_bc : m_nat_bdr) {
      SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(nat_bc.coef), 
                    "Traction boundary condition had a non-vector coefficient.");
      m_l_form->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(nat_bc.coef)), nat_bc.markers);
    }
    m_l_form->Assemble();
    m_rhs = std::unique_ptr<mfem::HypreParVector>(m_l_form->ParallelAssemble());
  } else {
    m_rhs  = std::make_unique<mfem::HypreParVector>(displacement->space.get());
    *m_rhs = 0.0;
  }

  // Assemble the stiffness matrix
  m_K_mat = std::unique_ptr<mfem::HypreParMatrix>(m_K_form->ParallelAssemble());

  // Eliminate the essential DOFs
  for (auto &bc : m_ess_bdr) {
    m_K_e_mat = std::unique_ptr<mfem::HypreParMatrix>(m_K_mat->EliminateRowsCols(bc.true_dofs));
  }

  // Initialize the eliminate BC RHS vector
  m_bc_rhs  = std::make_unique<mfem::HypreParVector>(displacement->space.get());
  *m_bc_rhs = 0.0;

  // Initialize the true vector
  displacement->gf->GetTrueDofs(*displacement->true_vec);

  std::unique_ptr<mfem::IterativeSolver> iter_solver;

  if (m_lin_params.prec == serac::Preconditioner::BoomerAMG) {
    SLIC_WARNING_IF(displacement->space->GetOrdering() == mfem::Ordering::byVDIM,
                    "Attempting to use BoomerAMG with nodal ordering.");

    auto prec_amg = std::make_unique<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(m_lin_params.print_level);
    prec_amg->SetElasticityOptions(displacement->space.get());
    m_K_prec = std::move(prec_amg);

    iter_solver = std::make_unique<mfem::GMRESSolver>(displacement->space->GetComm());
  }
  // If not AMG, just MINRES with Jacobi smoothing
  else {
    auto K_hypreSmoother = std::make_unique<mfem::HypreSmoother>();
    K_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    K_hypreSmoother->SetPositiveDiagonal(true);
    m_K_prec = std::move(K_hypreSmoother);

    iter_solver = std::make_unique<mfem::MINRESSolver>(displacement->space->GetComm());
  }
  
  iter_solver->SetRelTol(m_lin_params.rel_tol);
  iter_solver->SetAbsTol(m_lin_params.abs_tol);
  iter_solver->SetMaxIter(m_lin_params.max_iter);
  iter_solver->SetPrintLevel(m_lin_params.print_level);
  iter_solver->SetPreconditioner(*m_K_prec);
  m_K_solver = std::move(iter_solver);
}

void ElasticitySolver::AdvanceTimestep(double &)
{
  // Initialize the true vector
  displacement->gf->GetTrueDofs(*displacement->true_vec);

  if (m_timestepper == serac::TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    SLIC_ERROR_ROOT(m_rank, "Only quasistatics implemented for linear elasticity!");
    serac::ExitGracefully(true);
  }

  // Distribute the shared DOFs
  displacement->gf->SetFromTrueDofs(*displacement->true_vec);
  m_cycle += 1;
}

// Solve the Quasi-static system
void ElasticitySolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  for (auto &bc : m_ess_bdr) {
    SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(bc.coef), 
                  "Displacement boundary condition had a non-vector coefficient.");
    auto vec_coef = std::get<std::shared_ptr<mfem::VectorCoefficient>>(bc.coef);
    vec_coef->SetTime(m_time);
    displacement->gf->ProjectBdrCoefficient(*vec_coef, bc.markers);
    displacement->gf->GetTrueDofs(*displacement->true_vec);
    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, bc.true_dofs, *displacement->true_vec, *m_bc_rhs);
  }

  m_K_solver->SetOperator(*m_K_mat);

  m_K_solver->Mult(*m_bc_rhs, *displacement->true_vec);
}

ElasticitySolver::~ElasticitySolver() {}
