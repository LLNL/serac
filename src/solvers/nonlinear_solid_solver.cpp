// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "nonlinear_solid_solver.hpp"

#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"

const int num_fields = 2;

NonlinearSolidSolver::NonlinearSolidSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields),
      velocity(m_state[0]),
      displacement(m_state[1]),
      m_newton_solver(pmesh->GetComm())
{
  velocity.mesh  = pmesh;
  velocity.coll  = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  velocity.space = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), velocity.coll.get(), pmesh->Dimension(),
                                                                 mfem::Ordering::byVDIM);
  velocity.gf    = std::make_shared<mfem::ParGridFunction>(velocity.space.get());
  *velocity.gf   = 0.0;
  velocity.name  = "velocity";

  displacement.mesh  = pmesh;
  displacement.coll  = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  displacement.space = std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), displacement.coll.get(),
                                                                     pmesh->Dimension(), mfem::Ordering::byVDIM);
  displacement.gf    = std::make_shared<mfem::ParGridFunction>(displacement.space.get());
  *displacement.gf   = 0.0;
  displacement.name  = "displacement";

  // Initialize the mesh node pointers
  m_reference_nodes = std::make_unique<mfem::ParGridFunction>(displacement.space.get());
  pmesh->GetNodes(*m_reference_nodes);
  pmesh->NewNodes(*m_reference_nodes);

  m_deformed_nodes = std::make_unique<mfem::ParGridFunction>(*m_reference_nodes);

  // Initialize the true DOF vector
  int              true_size = velocity.space->TrueVSize();
  mfem::Array<int> true_offset(3);
  true_offset[0] = 0;
  true_offset[1] = true_size;
  true_offset[2] = 2 * true_size;
  m_block        = std::make_unique<mfem::BlockVector>(true_offset);

  m_block->GetBlockView(1, displacement.true_vec);
  displacement.true_vec = 0.0;

  m_block->GetBlockView(0, velocity.true_vec);
  velocity.true_vec = 0.0;
}

void NonlinearSolidSolver::SetDisplacementBCs(const std::vector<int> &                 disp_bdr,
                                              std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef, *displacement.space, -1);
}

void NonlinearSolidSolver::SetDisplacementBCs(const std::vector<int> &           disp_bdr,
                                              std::shared_ptr<mfem::Coefficient> disp_bdr_coef, int component)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef, *displacement.space, component);
}

void NonlinearSolidSolver::SetTractionBCs(const std::vector<int> &                 trac_bdr,
                                          std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef, int component)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef, component);
}

void NonlinearSolidSolver::SetHyperelasticMaterialParameters(double mu, double K)
{
  m_model.reset(new mfem::NeoHookeanModel(mu, K));
}

void NonlinearSolidSolver::SetViscosity(std::shared_ptr<mfem::Coefficient> visc) { m_viscosity = visc; }

void NonlinearSolidSolver::SetDisplacement(mfem::VectorCoefficient &disp_state)
{
  disp_state.SetTime(m_time);
  displacement.gf->ProjectCoefficient(disp_state);
  m_gf_initialized[1] = true;
}

void NonlinearSolidSolver::SetVelocity(mfem::VectorCoefficient &velo_state)
{
  velo_state.SetTime(m_time);
  velocity.gf->ProjectCoefficient(velo_state);
  m_gf_initialized[0] = true;
}

void NonlinearSolidSolver::SetSolverParameters(const LinearSolverParameters &   lin_params,
                                               const NonlinearSolverParameters &nonlin_params)
{
  m_lin_params    = lin_params;
  m_nonlin_params = nonlin_params;
}

void NonlinearSolidSolver::CompleteSetup()
{
  // Define the nonlinear form
  m_H_form = std::make_shared<mfem::ParNonlinearForm>(displacement.space.get());

  // Add the hyperelastic integrator
  if (m_timestepper == TimestepMethod::QuasiStatic) {
    m_H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(m_model.get()));
  } else {
    m_H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(m_model.get()));
  }

  // Add the traction integrator
  for (auto &nat_bc_data : m_nat_bdr) {
    m_H_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(*nat_bc_data->vec_coef), nat_bc_data->markers);
  }

  // Add the essential boundary
  mfem::Array<int> essential_dofs(0);

  // Build the dof array lookup tables
  displacement.space->BuildDofToArrays();

  // Project the essential boundary coefficients
  for (auto &bc : m_ess_bdr) {
    // Generate the scalar dof list from the vector dof list
    mfem::Array<int> dof_list(bc->true_dofs.Size());
    for (int i = 0; i < bc->true_dofs.Size(); ++i) {
      dof_list[i] = displacement.space->VDofToDof(bc->true_dofs[i]);
    }

    // Project the coefficient
    if (bc->component == -1) {
      // If it contains all components, project the vector
      displacement.gf->ProjectCoefficient(*bc->vec_coef, dof_list);
    } else {
      // If it is only a single component, project the scalar
      displacement.gf->ProjectCoefficient(*bc->scalar_coef, dof_list, bc->component);
    }

    // Add the vector dofs to the total essential BC dof list
    essential_dofs.Append(bc->true_dofs);
  }

  // Remove any duplicates from the essential BC list
  essential_dofs.Sort();
  essential_dofs.Unique();

  m_H_form->SetEssentialTrueDofs(essential_dofs);

  // If dynamic, create the mass and viscosity forms
  if (m_timestepper != TimestepMethod::QuasiStatic) {
    const double              ref_density = 1.0;  // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    m_M_form = std::make_shared<mfem::ParBilinearForm>(displacement.space.get());

    m_M_form->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    m_M_form->Assemble(0);
    m_M_form->Finalize(0);

    m_S_form = std::make_shared<mfem::ParBilinearForm>(displacement.space.get());
    m_S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*m_viscosity));
    m_S_form->Assemble(0);
    m_S_form->Finalize(0);
  }

  // Set up the jacbian solver based on the linear solver options
  if (m_lin_params.prec == Preconditioner::BoomerAMG) {
    MFEM_VERIFY(displacement.space->GetOrdering() == mfem::Ordering::byVDIM,
                "Attempting to use BoomerAMG with nodal ordering.");
    auto prec_amg = std::make_shared<mfem::HypreBoomerAMG>();
    prec_amg->SetPrintLevel(m_lin_params.print_level);
    prec_amg->SetElasticityOptions(displacement.space.get());
    m_J_prec = std::static_pointer_cast<mfem::Solver>(prec_amg);

    auto J_gmres = std::make_shared<mfem::GMRESSolver>(displacement.space->GetComm());
    J_gmres->SetRelTol(m_lin_params.rel_tol);
    J_gmres->SetAbsTol(m_lin_params.abs_tol);
    J_gmres->SetMaxIter(m_lin_params.max_iter);
    J_gmres->SetPrintLevel(m_lin_params.print_level);
    J_gmres->SetPreconditioner(*m_J_prec);
    m_J_solver = std::static_pointer_cast<mfem::Solver>(J_gmres);
  } else {
    auto J_hypreSmoother = std::make_shared<mfem::HypreSmoother>();
    J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    m_J_prec = std::static_pointer_cast<mfem::Solver>(J_hypreSmoother);

    auto J_minres = std::make_shared<mfem::MINRESSolver>(displacement.space->GetComm());
    J_minres->SetRelTol(m_lin_params.rel_tol);
    J_minres->SetAbsTol(m_lin_params.abs_tol);
    J_minres->SetMaxIter(m_lin_params.max_iter);
    J_minres->SetPrintLevel(m_lin_params.print_level);
    J_minres->SetPreconditioner(*m_J_prec);
    m_J_solver = std::static_pointer_cast<mfem::Solver>(J_minres);
  }

  // Set the newton solve parameters
  m_newton_solver.SetSolver(*m_J_solver);
  m_newton_solver.SetPrintLevel(m_nonlin_params.print_level);
  m_newton_solver.SetRelTol(m_nonlin_params.rel_tol);
  m_newton_solver.SetAbsTol(m_nonlin_params.abs_tol);
  m_newton_solver.SetMaxIter(m_nonlin_params.max_iter);

  // Set the MFEM abstract operators for use with the internal MFEM solvers
  if (m_timestepper == TimestepMethod::QuasiStatic) {
    m_newton_solver.iterative_mode = true;
    m_nonlinear_oper               = std::make_shared<NonlinearSolidQuasiStaticOperator>(m_H_form);
    m_newton_solver.SetOperator(*m_nonlinear_oper);
  } else {
    m_newton_solver.iterative_mode = false;
    m_timedep_oper = std::make_shared<NonlinearSolidDynamicOperator>(m_H_form, m_S_form, m_M_form, m_ess_bdr,
                                                                     m_newton_solver, m_lin_params);
    m_ode_solver->Init(*m_timedep_oper);
  }
}

// Solve the Quasi-static Newton system
void NonlinearSolidSolver::QuasiStaticSolve()
{
  mfem::Vector zero;
  m_newton_solver.Mult(zero, displacement.true_vec);
}

// Advance the timestep
void NonlinearSolidSolver::AdvanceTimestep(__attribute__((unused)) double &dt)
{
  // Initialize the true vector
  velocity.gf->GetTrueDofs(velocity.true_vec);
  displacement.gf->GetTrueDofs(displacement.true_vec);

  // Set the mesh nodes to the reference configuration
  displacement.mesh->NewNodes(*m_reference_nodes);
  velocity.mesh->NewNodes(*m_reference_nodes);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    m_ode_solver->Step(*m_block, m_time, dt);
  }

  // Distribute the shared DOFs
  velocity.gf->SetFromTrueDofs(velocity.true_vec);
  displacement.gf->SetFromTrueDofs(displacement.true_vec);

  // Update the mesh with the new deformed nodes
  m_deformed_nodes->Set(1.0, *displacement.gf);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    m_deformed_nodes->Add(1.0, *m_reference_nodes);
  }

  displacement.mesh->NewNodes(*m_deformed_nodes);
  velocity.mesh->NewNodes(*m_deformed_nodes);

  m_cycle += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver() {}
