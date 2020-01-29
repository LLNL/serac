// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_solver.hpp"

ThermalSolver::ThermalSolver(int order, mfem::ParMesh *pmesh) :
  BaseSolver(), m_M_form(nullptr), m_K_form(nullptr), m_M_mat(nullptr), m_K_mat(nullptr),
  m_l_form(nullptr), m_rhs(nullptr), m_K_solver(nullptr), m_K_prec(nullptr), m_kappa(nullptr), m_source(nullptr),
  m_dyn_oper(nullptr), m_dynamic(true), m_gf_initialized(false)
{
  m_pmesh = pmesh;
  m_fecolls.SetSize(1);
  m_fespaces.SetSize(1);

  m_fecolls[0] = new mfem::H1_FECollection(order, pmesh->Dimension());
  m_fespaces[0] = new mfem::ParFiniteElementSpace(pmesh, m_fecolls[0]);

  m_state_gf.SetSize(1);
  m_state_gf[0] = new mfem::ParGridFunction(m_fespaces[0]);

  m_true_vec.SetSize(1);
  m_true_vec[0] = new mfem::HypreParVector(m_fespaces[0]);
}

void ThermalSolver::SetInitialState(mfem::Coefficient &temp)
{
  m_state_gf[0]->ProjectCoefficient(temp);
  m_gf_initialized = true;
}

void ThermalSolver::SetTemperatureBCs(mfem::Array<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef)
{
  SetEssentialBCs(ess_bdr, ess_bdr_coef);

  m_fespaces[0]->GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list);
  m_state_gf[0]->ProjectBdrCoefficient(*ess_bdr_coef, ess_bdr);
}

void ThermalSolver::SetFluxBCs(mfem::Array<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef)
{
  SetNaturalBCs(nat_bdr, nat_bdr_coef);
}

void ThermalSolver::SetConductivity(mfem::Coefficient &kappa)
{
  m_kappa = &kappa;
}

void ThermalSolver::SetSource(mfem::Coefficient &source)
{
  m_source = &source;
}

void ThermalSolver::SetLinearSolverParameters(const LinearSolverParameters &params)
{
  m_lin_params = params;
}

void ThermalSolver::CompleteSetup(const bool allow_dynamic)
{
  MFEM_ASSERT(m_kappa != nullptr, "Conductivity not set in ThermalSolver!");

  m_state_gf[0]->GetTrueDofs(*m_true_vec[0]);

  m_K_form = new mfem::ParBilinearForm(m_fespaces[0]);
  m_K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(*m_kappa));
  m_K_form->Assemble(0); // keep sparsity pattern of M and K the same
  m_K_form->Finalize();

  m_l_form = new mfem::ParLinearForm(m_fespaces[0]);
  if (m_source != nullptr && m_nat_bdr_coef == nullptr) {
    m_l_form->AddDomainIntegrator(new mfem::DomainLFIntegrator(*m_source));
    m_rhs = m_l_form->ParallelAssemble();
  } else {
    m_rhs = new mfem::HypreParVector(m_fespaces[0]);
    *m_rhs = 0.0;
  }

  m_K_mat = m_K_form->ParallelAssemble();
  m_K_mat->EliminateRowsCols(m_ess_tdof_list, *m_true_vec[0], *m_rhs);

  if (allow_dynamic) {
    m_M_mat = new mfem::HypreParMatrix;
    m_M_form = new mfem::ParBilinearForm(m_fespaces[0]);
    m_M_form->AddDomainIntegrator(new mfem::MassIntegrator());
    m_M_form->Assemble(0); // keep sparsity pattern of M and K the same
    m_M_form->Finalize();

    m_M_mat = m_M_form->ParallelAssemble();
    mfem::HypreParMatrix *Me = m_M_mat->EliminateRowsCols(m_ess_tdof_list);
    delete Me;

    m_dyn_oper = new DynamicConductionOperator(m_fespaces[0]->GetComm(), m_fespaces[0]->GetTrueVSize(), m_lin_params);
    m_dyn_oper->SetMMatrix(m_M_mat);
    m_dyn_oper->SetKMatrixAndRHS(m_K_mat, m_rhs);
    m_ode_solver->Init(*m_dyn_oper);
  }
}

void ThermalSolver::StaticSolve()
{
  m_K_solver = new mfem::CGSolver(m_fespaces[0]->GetComm());
  m_K_prec = new mfem::HypreSmoother();

  m_K_solver->iterative_mode = false;
  m_K_solver->SetRelTol(m_lin_params.rel_tol);
  m_K_solver->SetAbsTol(m_lin_params.abs_tol);
  m_K_solver->SetMaxIter(m_lin_params.max_iter);
  m_K_solver->SetPrintLevel(m_lin_params.print_level);
  m_K_prec->SetType(mfem::HypreSmoother::Jacobi);
  m_K_solver->SetPreconditioner(*m_K_prec);
  m_K_solver->SetOperator(*m_K_mat);

  m_K_solver->Mult(*m_rhs, *m_true_vec[0]);
  m_state_gf[0]->SetFromTrueDofs(*m_true_vec[0]);
}

void ThermalSolver::AdvanceTimestep(double dt)
{
  MFEM_ASSERT(m_dynamic == true, "Solver not setup with dynamic option!");

  m_ode_solver->Step(*m_true_vec[0], m_time, dt);
  m_state_gf[0]->SetFromTrueDofs(*m_true_vec[0]);

  m_time += dt;
  m_cycle += 1;
}

ThermalSolver::~ThermalSolver()
{
  delete m_K_form;
  delete m_l_form;
  delete m_rhs;
  delete m_K_mat;
  delete m_K_solver;
  delete m_K_prec;

  if (m_dynamic) {
    delete m_dyn_oper;
    delete m_M_form;
    delete m_M_mat;
  }
}

DynamicConductionOperator::DynamicConductionOperator(MPI_Comm comm, int height, LinearSolverParameters &params)
  : mfem::TimeDependentOperator(height, 0.0), m_M_solver(nullptr), m_T_solver(nullptr), m_M_prec(nullptr),
    m_T_prec(nullptr), m_M_mat(nullptr), m_K_mat(nullptr), m_T_mat(nullptr), m_true_rhs(nullptr), m_z(height)
{
  m_M_solver = new mfem::CGSolver(comm);
  m_M_prec = new mfem::HypreSmoother();
  m_M_solver->iterative_mode = false;
  m_M_solver->SetRelTol(params.rel_tol);
  m_M_solver->SetAbsTol(params.abs_tol);
  m_M_solver->SetMaxIter(params.max_iter);
  m_M_solver->SetPrintLevel(params.print_level);
  m_M_prec->SetType(mfem::HypreSmoother::Jacobi);
  m_M_solver->SetPreconditioner(*m_M_prec);

  m_T_solver = new mfem::CGSolver(comm);
  m_T_prec = new mfem::HypreSmoother();
  m_T_solver->iterative_mode = false;
  m_T_solver->SetRelTol(params.rel_tol);
  m_T_solver->SetAbsTol(params.abs_tol);
  m_T_solver->SetMaxIter(params.max_iter);
  m_T_solver->SetPrintLevel(params.print_level);
  m_T_solver->SetPreconditioner(*m_T_prec);

}

void DynamicConductionOperator::SetMMatrix(mfem::HypreParMatrix *M_mat)
{
  m_M_mat = M_mat;
  m_M_solver->SetOperator(*m_M_mat);
}

void DynamicConductionOperator::SetKMatrixAndRHS(mfem::HypreParMatrix *K_mat, mfem::Vector *rhs)
{
  m_K_mat = K_mat;
  m_true_rhs = rhs;
}

void DynamicConductionOperator::Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::Mult!");
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!");
  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  m_K_mat->Mult(u, m_z);
  m_z.Neg(); // z = -z
  m_z.Add(1.0, *m_true_rhs);
  m_M_solver->Mult(m_z, du_dt);
}

void DynamicConductionOperator::ImplicitSolve(const double dt,
    const mfem::Vector &u, mfem::Vector &du_dt)
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!");
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!");
  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt

  m_T_mat = new mfem::HypreParMatrix;
  m_T_mat = Add(1.0, *m_M_mat, dt, *m_K_mat);
  m_T_solver->SetOperator(*m_T_mat);

  m_K_mat->Mult(u, m_z);
  m_z.Neg();
  m_z.Add(1.0, *m_true_rhs);
  m_T_solver->Mult(m_z, du_dt);
  delete m_T_mat;
}

DynamicConductionOperator::~DynamicConductionOperator()
{
  delete m_M_solver;
  delete m_M_prec;
  delete m_T_solver;
  delete m_T_prec;
}
