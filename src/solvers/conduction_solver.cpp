// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "conduction_solver.hpp"

ThermalSolver::ThermalSolver(int order, mfem::ParMesh *pmesh, bool allow_dynamic) :
  BaseSolver(), m_M_form(nullptr), m_K_form(nullptr), m_M_mat(nullptr), m_T_mat(nullptr),
  m_dyn_oper(nullptr), m_dynamic(allow_dynamic), m_gf_initialized(false)
{
  m_fespace.Resize(1);
  
  mfem::H1_FECollection fe_coll(order, pmesh->Dimension);
  m_fespace[0] = new mfem::ParFinieElementSpace(pmesh, &fe_coll);

  m_stategf.Resize(1);

  m_stategf[0] = new mfem::ParGridFunction(m_fespace[0]);
}

void ThermalSolver::SetInitialState(const mfem::Coefficient &temp)
{
  m_stategf[0]->ProjectCoefficient(temp);
  m_gf_initialized = true;
}

void SetTemperatureBCs(const mfem::Array<int> &ess_bdr, const mfem::Coefficient *ess_bdr_coef)
{
  BaseSolver::SetEssentialBCs(ess_bdr)
  m_fespace[0]->GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list); 
  m_stategf[0]->ProjectBdrCoefficient(ess_bdr_coef, ess_bdr);
}

void SetFluxBCs(const mfem::Array<int> &nat_bdr, const mfem::Coefficient *nat_bdr_coef)
{
  BaseSolver::SetNaturalBCs(nat_bdr, nat_bdr_coef);
}

void SetConductivity(const mfem::Coefficient &kappa)
{
  m_kappa = kappa;
  m_conductivity_set = true;
}

void CompleteSetup() {
  m_K_form = new mfem::ParBilinearForm(m_fespace[0]);
  m_K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(m_kappa));
  m_K_form->Assemble(0); // keep sparsity pattern of M and K the same
  m_K_form->FormSystemMatrix(m_ess_tdof_list, m_K_mat);

  m_K_solver.iterative_mode = false;
  m_K_solver.SetRelTol(rel_tol);
  m_K_solver.SetAbsTol(0.0);
  m_K_solver.SetMaxIter(100);
  m_K_solver.SetPrintLevel(0);
  m_K_prec.SetType(mfem::HypreSmoother::Jacobi);
  m_K_solver.SetPreconditioner(m_K_prec);

  if (allow_dynamic) {
    m_M_form = new mfem::ParBilinearForm(m_fespaces[0]);
    m_M_form->AddDomainIntegrator(new mfem::MassIntegrator());
    m_M_form->Assemble(0); // keep sparsity pattern of M and K the same
    m_M_form->FormSystemMatrix(m_ess_tdof_list, m_M_mat);
   
    m_dyn_oper = new DynamicConductionOperator(m_fespaces[0].GetTrueVSize());
    m_dyn_oper->SetMMatrix(m_M_mat);
    m_dyn_oper->SetKMatrix(m_K_mat);
    m_ode_solver->Init(m_dyn_oper);
 }
} 

void ConductionSolver::StaticSolve()
{
  mfem::Vector zero;

  m_K_solver.Mult(zero, m_state_gf[0])
}

void AdvanceTimestep(const double dt)
{
  MFEM_ASSERT(m_dynamic == true, "Solver not setup with dynamic option!");
  
  m_ode_solver->Step(m_state_gf[0], m_time, dt);

  m_time += dt;
  m_cycle += 1;
}

DynamicConductionOperator::DynamicConductionOperator(int height)
  : mfem::TimeDependentOperator(height, 0.0), m_current_dt(0.0),
    m_z(height)
{
  m_M_solver.iterative_mode = false;
  m_M_solver.SetRelTol(rel_tol);
  m_M_solver.SetAbsTol(0.0);
  m_M_solver.SetMaxIter(100);
  m_M_solver.SetPrintLevel(0);
  m_M_prec.SetType(mfem::HypreSmoother::Jacobi);
  m_M_solver.SetPreconditioner(m_M_prec);

  m_T_solver.iterative_mode = false;
  m_T_solver.SetRelTol(rel_tol);
  m_T_solver.SetAbsTol(0.0);
  m_T_solver.SetMaxIter(100);
  m_T_solver.SetPrintLevel(0);
  m_T_solver.SetPreconditioner(m_T_prec);
 
}

void DynamicConductionOperator::SetMMatrix(mfem::HypreParMatrix *M_mat)
{
  m_M_mat = M_mat;
  m_M_solver.SetOperator(*m_M_mat);
}

void DynamicConductionOperator::SetKMatrix(mfem::HypreParMatrix *K_mat);
{
  m_K_mat = K_mat;
}

void DynamicConductionOperator::Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::Mult!"); 
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::Mult!"); 
  // Compute:
  //    du_dt = M^{-1}*-K(u)
  // for du_dt
  m_K_mat.Mult(u, m_z);
  m_z.Neg(); // z = -z
  m_M_solver.Mult(m_z, du_dt);
}

void DynamicConductionOperator::ImplicitSolve(const double dt,
                                     const mfem::Vector &u, mfem::Vector &du_dt)
{
  MFEM_ASSERT(m_M_mat != nullptr, "Mass matrix not set in ConductionSolver::ImplicitSolve!"); 
  MFEM_ASSERT(m_K_mat != nullptr, "Stiffness matrix not set in ConductionSolver::ImplicitSolve!"); 
  // Solve the equation:
  //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
  // for du_dt

  m_T_mat = Add(1.0, m_M_mat, dt, m_K_mat);
  m_T_solver.SetOperator(*m_T_mat);

  m_K_mat.Mult(u, m_z);
  m_z.Neg();
  m_T_solver.Mult(m_z, du_dt);
  delete m_T_mat;
}

