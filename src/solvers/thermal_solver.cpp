// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_solver.hpp"

const int num_fields = 1;

ThermalSolver::ThermalSolver(int order, mfem::ParMesh *pmesh)
    : BaseSolver(num_fields), temperature(m_state[0]), m_kappa(nullptr), m_source(nullptr), m_dyn_oper(nullptr)
{
  temperature.mesh     = pmesh;
  temperature.coll     = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
  temperature.space    = std::make_shared<mfem::ParFiniteElementSpace>(pmesh, temperature.coll.get());
  temperature.gf       = std::make_shared<mfem::ParGridFunction>(temperature.space.get());
  temperature.true_vec = mfem::HypreParVector(temperature.space.get());
  temperature.name     = "temperature";

  // and initial conditions
  *temperature.gf      = 0.0;
  temperature.true_vec = 0.0;

  temperature.name = "temperature";
}

void ThermalSolver::SetInitialState(mfem::Coefficient &temp)
{
  // Project the coefficient onto the grid function
  temp.SetTime(m_time);
  temperature.gf->ProjectCoefficient(temp);
  m_gf_initialized = true;
}

void ThermalSolver::SetTemperatureBCs(std::vector<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef)
{
  SetEssentialBCs(ess_bdr, ess_bdr_coef);

  // Get the essential dof indicies and project the coefficient onto them
  temperature.space->GetEssentialTrueDofs(*m_ess_bdr.get(), m_ess_tdof_list);
}

void ThermalSolver::SetFluxBCs(std::vector<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef)
{
  // Set the natural (integral) boundary condition
  SetNaturalBCs(nat_bdr, nat_bdr_coef);
}

void ThermalSolver::SetConductivity(mfem::Coefficient &kappa)
{
  // Set the conduction coefficient
  m_kappa = &kappa;
}

void ThermalSolver::SetSource(mfem::Coefficient &source)
{
  // Set the body source integral coefficient
  m_source = &source;
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
  m_K_form = std::make_shared<mfem::ParBilinearForm>(temperature.space.get());
  m_K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(*m_kappa));
  m_K_form->Assemble(0);  // keep sparsity pattern of M and K the same
  m_K_form->Finalize();

  // Add the body source to the RS if specified
  m_l_form = std::make_shared<mfem::ParLinearForm>(temperature.space.get());
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
  m_K_e_mat.reset(m_K_mat->EliminateRowsCols(m_ess_tdof_list));

  // Initialize the eliminated BC RHS vector
  m_bc_rhs  = std::make_shared<mfem::HypreParVector>(temperature.space.get());
  *m_bc_rhs = 0.0;

  // Initialize the true vector
  temperature.gf->GetTrueDofs(temperature.true_vec);

  if (m_timestepper != TimestepMethod::QuasiStatic) {
    // If dynamic, assemble the mass matrix
    m_M_form = std::make_shared<mfem::ParBilinearForm>(temperature.space.get());
    m_M_form->AddDomainIntegrator(new mfem::MassIntegrator());
    m_M_form->Assemble(0);  // keep sparsity pattern of M and K the same
    m_M_form->Finalize();

    m_M_mat.reset(m_M_form->ParallelAssemble());

    // Make the time integration operator and set the appropriate matricies
    m_dyn_oper = std::make_shared<DynamicConductionOperator>(temperature.space, m_lin_params);
    m_dyn_oper->SetMMatrix(m_M_mat, m_M_e_mat);
    m_dyn_oper->SetKMatrix(m_K_mat, m_K_e_mat);
    m_dyn_oper->SetLoadVector(m_rhs);
    m_dyn_oper->SetEssentialBCs(m_ess_bdr_coef, *m_ess_bdr.get(), m_ess_tdof_list);

    m_ode_solver->Init(*m_dyn_oper);
  }
}

void ThermalSolver::QuasiStaticSolve()
{
  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  if (m_ess_bdr_coef != nullptr) {
    m_ess_bdr_coef->SetTime(m_time);
    temperature.gf->ProjectBdrCoefficient(*m_ess_bdr_coef, *m_ess_bdr.get());
    temperature.gf->GetTrueDofs(temperature.true_vec);
    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, m_ess_tdof_list, temperature.true_vec, *m_bc_rhs);
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
    MFEM_ASSERT(m_gf_initialized, "Thermal state not initialized!");

    // Step the time integrator
    m_ode_solver->Step(temperature.true_vec, m_time, dt);
  }

  // Distribute the shared DOFs
  temperature.gf->SetFromTrueDofs(temperature.true_vec);
  m_cycle += 1;
}

DynamicConductionOperator::DynamicConductionOperator(std::shared_ptr<mfem::ParFiniteElementSpace> fespace,
                                                     LinearSolverParameters &                     params)
    : mfem::TimeDependentOperator(fespace->GetTrueVSize(), 0.0),
      m_fespace(fespace),
      m_bc_rhs(nullptr),
      m_ess_bdr_coef(nullptr),
      m_z(fespace->GetTrueVSize()),
      m_y(fespace->GetTrueVSize()),
      m_x(fespace->GetTrueVSize()),
      m_old_dt(-1.0)
{
  // Set the mass solver options (CG and Jacobi for now)
  m_M_solver                 = std::make_shared<mfem::CGSolver>(m_fespace->GetComm());
  m_M_prec                   = std::make_shared<mfem::HypreSmoother>();
  m_M_solver->iterative_mode = false;
  m_M_solver->SetRelTol(params.rel_tol);
  m_M_solver->SetAbsTol(params.abs_tol);
  m_M_solver->SetMaxIter(params.max_iter);
  m_M_solver->SetPrintLevel(params.print_level);
  m_M_prec->SetType(mfem::HypreSmoother::Jacobi);
  m_M_solver->SetPreconditioner(*m_M_prec);

  // Use the same options for the T (= M + dt K) solver
  m_T_solver                 = std::make_shared<mfem::CGSolver>(m_fespace->GetComm());
  m_T_prec                   = std::make_shared<mfem::HypreSmoother>();
  m_T_solver->iterative_mode = false;
  m_T_solver->SetRelTol(params.rel_tol);
  m_T_solver->SetAbsTol(params.abs_tol);
  m_T_solver->SetMaxIter(params.max_iter);
  m_T_solver->SetPrintLevel(params.print_level);
  m_T_solver->SetPreconditioner(*m_T_prec);

  m_state_gf = new mfem::ParGridFunction(m_fespace.get());
  m_bc_rhs   = new mfem::Vector(fespace->GetTrueVSize());
}

void DynamicConductionOperator::SetMMatrix(std::shared_ptr<mfem::HypreParMatrix> M_mat,
                                           std::shared_ptr<mfem::HypreParMatrix> M_e_mat)
{
  // Set the mass matrix
  m_M_mat   = M_mat;
  m_M_e_mat = M_e_mat;
}

void DynamicConductionOperator::SetKMatrix(std::shared_ptr<mfem::HypreParMatrix> K_mat,
                                           std::shared_ptr<mfem::HypreParMatrix> K_e_mat)
{
  // Set the stiffness matrix and RHS
  m_K_mat   = K_mat;
  m_K_e_mat = K_e_mat;
}

void DynamicConductionOperator::SetLoadVector(std::shared_ptr<mfem::Vector> rhs) { m_rhs = rhs; }

void DynamicConductionOperator::SetEssentialBCs(mfem::Coefficient *ess_bdr_coef, mfem::Array<int> &ess_bdr,
                                                mfem::Array<int> &ess_tdof_list)
{
  m_ess_bdr_coef  = ess_bdr_coef;
  m_ess_bdr       = ess_bdr;
  m_ess_tdof_list = ess_tdof_list;
}

// TODO: allow for changing thermal essential boundary conditions
void DynamicConductionOperator::Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::Mult!");
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!");

  m_y = u;
  m_M_solver->SetOperator(*m_M_mat);

  *m_bc_rhs = *m_rhs;
  mfem::EliminateBC(*m_K_mat, *m_K_e_mat, m_ess_tdof_list, m_y, *m_bc_rhs);

  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  m_K_mat->Mult(m_y, m_z);
  m_z.Neg();  // z = -zw  m_z.Add(1.0, *m_bc_rhs);
  m_z.Add(1.0, *m_bc_rhs);
  m_M_solver->Mult(m_z, du_dt);
}

void DynamicConductionOperator::ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &du_dt)
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!");
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!");

  m_y = u;

  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt
  if (dt != m_old_dt) {
    m_T_mat.reset(mfem::Add(1.0, *m_M_mat, dt, *m_K_mat));

    // Eliminate the essential DOFs from the T matrix
    m_T_e_mat.reset(m_T_mat->EliminateRowsCols(m_ess_tdof_list));
    m_T_solver->SetOperator(*m_T_mat);
  }

  // Apply the boundary conditions
  *m_bc_rhs = *m_rhs;
  m_x       = 0.0;

  if (m_ess_bdr_coef != nullptr) {
    m_ess_bdr_coef->SetTime(t);
    m_state_gf->SetFromTrueDofs(m_y);
    m_state_gf->ProjectBdrCoefficient(*m_ess_bdr_coef, m_ess_bdr);
    m_state_gf->GetTrueDofs(m_y);

    mfem::EliminateBC(*m_K_mat, *m_K_e_mat, m_ess_tdof_list, m_y, *m_bc_rhs);
  }

  m_K_mat->Mult(m_y, m_z);
  m_z.Neg();
  m_z.Add(1.0, *m_bc_rhs);
  m_T_solver->Mult(m_z, du_dt);

  // Save the dt used to compute the T matrix
  m_old_dt = dt;
}

DynamicConductionOperator::~DynamicConductionOperator()
{
  delete m_state_gf;
  delete m_bc_rhs;
}
