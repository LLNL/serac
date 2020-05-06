// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_solver.hpp"

const int num_fields = 1;

ThermalSolver::ThermalSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields), temperature(m_state[0])
{
  temperature.mesh     = pmesh;
  temperature.coll     = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  temperature.space    = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), temperature.coll.get());
  temperature.gf       = std::make_shared<mfem::ParGridFunction>(temperature.space.get());
  temperature.true_vec = mfem::HypreParVector(temperature.space.get());
  temperature.name     = "temperature";

  // and initial conditions
  *temperature.gf      = 0.0;
  temperature.true_vec = 0.0;

  temperature.name = "temperature";
}

void ThermalSolver::SetTemperature(mfem::Coefficient &temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(m_time);
  temperature.gf->ProjectCoefficient(temp);
  m_gf_initialized[0] = true;
}

void ThermalSolver::SetTemperatureBCs(const std::vector<int> &ess_bdr, std::shared_ptr<mfem::Coefficient> ess_bdr_coef)
{
  SetEssentialBCs(ess_bdr, ess_bdr_coef);

  // Get the essential dof indicies and project the coefficient onto them
  for (auto &ess_bc_data : m_ess_bdr) {
    temperature.space->GetEssentialTrueDofs(ess_bc_data->bc_markers, ess_bc_data->true_dofs);
  }
}

void ThermalSolver::SetFluxBCs(const std::vector<int> &nat_bdr, std::shared_ptr<mfem::Coefficient> nat_bdr_coef)
{
  // Set the natural (integral) boundary condition
  SetNaturalBCs(nat_bdr, nat_bdr_coef);
}

void ThermalSolver::SetConductivity(std::shared_ptr<mfem::Coefficient> kappa)
{
  // Set the conduction coefficient
  m_kappa = kappa;
}

void ThermalSolver::SetSource(std::shared_ptr<mfem::Coefficient> source)
{
  // Set the body source integral coefficient
  m_source = source;
}

void ThermalSolver::SetLinearSolverParameters(const LinearSolverParameters &params)
{
  // Save the solver params object
  // TODO: separate the M and K solver params
  m_lin_params = params;
}

void ThermalSolver::CompleteSetup()
{
  MFEM_ASSERT(m_kappa != nullptr, "Conductivity not set in ThermalSolver!");

  // Add the domain diffusion integrator to the K form and assemble the matrix
  m_K_form = std::make_unique<mfem::ParBilinearForm>(temperature.space.get());
  m_K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(*m_kappa));
  m_K_form->Assemble(0);  // keep sparsity pattern of M and K the same
  m_K_form->Finalize();

  // Add the body source to the RS if specified
  m_l_form = std::make_unique<mfem::ParLinearForm>(temperature.space.get());
  if (m_source != nullptr) {
    m_l_form->AddDomainIntegrator(new mfem::DomainLFIntegrator(*m_source));
    m_rhs.reset(m_l_form->ParallelAssemble());
  } else {
    m_rhs  = std::make_shared<mfem::HypreParVector>(temperature.space.get());
    *m_rhs = 0.0;
  }

  // Assemble the stiffness matrix
  m_K_mat.reset(m_K_form->ParallelAssemble());

  // Eliminate the essential DOFs from the stiffness matrix
  for (auto &ess_bc_data : m_ess_bdr) {
    m_K_e_mat.reset(m_K_mat->EliminateRowsCols(ess_bc_data->true_dofs));
  }

  // Initialize the eliminated BC RHS vector
  m_bc_rhs  = std::make_shared<mfem::HypreParVector>(temperature.space.get());
  *m_bc_rhs = 0.0;

  // Initialize the true vector
  temperature.gf->GetTrueDofs(temperature.true_vec);

  if (m_timestepper != TimestepMethod::QuasiStatic) {
    // If dynamic, assemble the mass matrix
    m_M_form = std::make_unique<mfem::ParBilinearForm>(temperature.space.get());
    m_M_form->AddDomainIntegrator(new mfem::MassIntegrator());
    m_M_form->Assemble(0);  // keep sparsity pattern of M and K the same
    m_M_form->Finalize();

    m_M_mat.reset(m_M_form->ParallelAssemble());

    // Make the time integration operator and set the appropriate matricies
    m_dyn_oper = std::make_unique<DynamicConductionOperator>(temperature.space, m_lin_params, m_ess_bdr);
    m_dyn_oper->SetMMatrix(m_M_mat, m_M_e_mat);
    m_dyn_oper->SetKMatrix(m_K_mat, m_K_e_mat);
    m_dyn_oper->SetLoadVector(m_rhs);

    m_ode_solver->Init(*m_dyn_oper);
  }
}

void ThermalSolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  for (auto &ess_bc_data : m_ess_bdr) {
    ess_bc_data->scalar_coef->SetTime(m_time);
    temperature.gf->ProjectBdrCoefficient(*ess_bc_data->scalar_coef, ess_bc_data->bc_markers);
    temperature.gf->GetTrueDofs(temperature.true_vec);
    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, ess_bc_data->true_dofs, temperature.true_vec, *m_bc_rhs);
  }

  // Solve the stiffness using CG with Jacobi preconditioning
  // and the given solverparams
  m_K_solver = std::make_shared<mfem::CGSolver>(temperature.space->GetComm());
  m_K_prec   = std::make_shared<mfem::HypreSmoother>();

  m_K_solver->iterative_mode = false;
  m_K_solver->SetRelTol(m_lin_params.rel_tol);
  m_K_solver->SetAbsTol(m_lin_params.abs_tol);
  m_K_solver->SetMaxIter(m_lin_params.max_iter);
  m_K_solver->SetPrintLevel(m_lin_params.print_level);
  m_K_prec->SetType(mfem::HypreSmoother::Jacobi);
  m_K_solver->SetPreconditioner(*m_K_prec);
  m_K_solver->SetOperator(*m_K_mat);

  // Perform the linear solve
  m_K_solver->Mult(*m_bc_rhs, temperature.true_vec);
}

void ThermalSolver::AdvanceTimestep(double &dt)
{
  // Initialize the true vector
  temperature.gf->GetTrueDofs(temperature.true_vec);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    MFEM_ASSERT(m_gf_initialized[0], "Thermal state not initialized!");

    // Step the time integrator
    m_ode_solver->Step(temperature.true_vec, m_time, dt);
  }

  // Distribute the shared DOFs
  temperature.gf->SetFromTrueDofs(temperature.true_vec);
  m_cycle += 1;
}
