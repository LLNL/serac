// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "nonlinear_solid_solver.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"
#include "integrators/inc_hyperelastic_integrator.hpp"

NonlinearSolidSolver::NonlinearSolidSolver(int order, mfem::ParMesh *pmesh) :
  BaseSolver(), m_H_form(nullptr), m_nonlinear_oper(nullptr), m_newton_solver(pmesh->GetComm()), m_J_solver(nullptr),
  m_J_prec(nullptr), m_model(nullptr)
{
  m_state.SetSize(2);
  m_state[0].mesh = pmesh;
  m_state[1].mesh = pmesh;

  // Use vector-valued H1 nodal basis functions for the displacement and velocity field
  m_state[0].coll = new mfem::H1_FECollection(order, pmesh->Dimension());
  m_state[0].space = new mfem::ParFiniteElementSpace(pmesh, m_state[0].coll, pmesh->Dimension(), mfem::Ordering::byVDIM);

  m_state[1].coll = new mfem::H1_FECollection(order, pmesh->Dimension());
  m_state[1].space = new mfem::ParFiniteElementSpace(pmesh, m_state[1].coll, pmesh->Dimension(), mfem::Ordering::byVDIM);

  // Initialize the grid functions
  m_state[0].gf = new mfem::ParGridFunction(m_state[0].space);
  *m_state[0].gf = 0.0;

  m_state[1].gf = new mfem::ParGridFunction(m_state[0].space);
  *m_state[1].gf = 0.0;

  // Initialize the true DOF vector
  m_state[0].true_vec = new mfem::HypreParVector(m_state[0].space);
  *m_state[0].true_vec = 0.0;

  m_state[1].true_vec = new mfem::HypreParVector(m_state[1].space);
  *m_state[1].true_vec = 0.0;
}

void NonlinearSolidSolver::SetDisplacementBCs(mfem::Array<int> &disp_bdr, mfem::VectorCoefficient *disp_bdr_coef)
{
  SetEssentialBCs(disp_bdr, disp_bdr_coef);

  // Get the list of essential DOFs
  m_state[0].space->GetEssentialTrueDofs(disp_bdr, m_ess_tdof_list);
}

void NonlinearSolidSolver::SetTractionBCs(mfem::Array<int> &trac_bdr, mfem::VectorCoefficient *trac_bdr_coef)
{
  SetNaturalBCs(trac_bdr, trac_bdr_coef);
}

void NonlinearSolidSolver::SetHyperelasticMaterialParameters(double mu, double K)
{
  delete m_model;
  m_model = new mfem::NeoHookeanModel(mu, K);
}

void NonlinearSolidSolver::SetInitialState(mfem::VectorCoefficient &disp_state, mfem::VectorCoefficient &velo_state)
{
  disp_state.SetTime(m_time);
  velo_state.SetTime(m_time);
  m_state[0].gf->ProjectCoefficient(disp_state);
  m_state[1].gf->ProjectCoefficient(velo_state);
  m_gf_initialized = true;
}

void NonlinearSolidSolver::SetLinearSolverParameters(const LinearSolverParameters &params)
{
  m_lin_params = params;
}

void NonlinearSolidSolver::CompleteSetup()
{
  // Define the nonlinear form
  m_H_form = new mfem::ParNonlinearForm(m_state[0].space);

  // Add the hyperelastic integrator
  if (m_timestepper == TimestepMethod::QuasiStatic) {
    m_H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(m_model));
  }
  else {
    m_H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(m_model));
  }

  // Add the traction integrator
  if (m_nat_bdr_vec_coef != nullptr) {
    m_H_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(*m_nat_bdr_vec_coef), m_nat_bdr);
  }

  // Add the essential boundary
  if (m_ess_bdr_vec_coef != nullptr) {
    m_H_form->SetEssentialBC(m_ess_bdr);
  }

  // If dynamic, create the mass and viscosity forms
  if (m_timestepper != TimestepMethod::QuasiStatic) {
    const double ref_density = 1.0; // density in the reference configuration
    mfem::ConstantCoefficient rho0(ref_density);

    m_M_form = new mfem::ParBilinearForm(&fes);

    m_M_form->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
    m_M_form->Assemble(0);
    m_M_form->Finalize(0);
    m_M_mat = m_M_form->ParallelAssemble();
    m_fe_space.GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list);
    mfem::HypreParMatrix *Me = m_M_mat->EliminateRowsCols(m_ess_tdof_list);
    delete Me;

    m_S_form = new mfem::ParBilinearForm(&fes);
    m_S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(m_viscosity));
    m_S_form->Assemble(0);
    m_S_form->Finalize(0);

    m_M_solver.iterative_mode = false;
    m_M_solver.SetRelTol(m_lin_params.rel_tol);
    m_M_solver.SetAbsTol(m_lin_params.abs_tol);
    m_M_solver.SetMaxIter(m_lin_params.max_iter);
    m_M_solver.SetPrintLevel(m_lin_params.print_level);
    m_M_prec.SetType(mfem::HypreSmoother::Jacobi);
    m_M_solver.SetPreconditioner(m_M_prec);
    m_M_solver.SetOperator(*m_M_mat); 
  }

  if (m_lin_params.prec == Preconditioner::BoomerAMG) {
    MFEM_VERIFY(m_state[0].space->GetOrdering() == mfem::Ordering::byVDIM, "Attempting to use BoomerAMG with nodal ordering.");
    mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
    prec_amg->SetPrintLevel(m_lin_params.print_level);
    prec_amg->SetElasticityOptions(m_state[0].space);
    m_J_prec = prec_amg;

    mfem::GMRESSolver *J_gmres = new mfem::GMRESSolver(m_state[0].space->GetComm());
    J_gmres->SetRelTol(m_lin_params.rel_tol);
    J_gmres->SetAbsTol(m_lin_params.abs_tol);
    J_gmres->SetMaxIter(m_lin_params.max_iter);
    J_gmres->SetPrintLevel(m_lin_params.print_level);
    J_gmres->SetPreconditioner(*m_J_prec);
    m_J_solver = J_gmres;  } else {
  } else {
    mfem::HypreSmoother *J_hypreSmoother = new mfem::HypreSmoother;
    J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
    J_hypreSmoother->SetPositiveDiagonal(true);
    m_J_prec = J_hypreSmoother;

    mfem::MINRESSolver *J_minres = new mfem::MINRESSolver(m_state[0].space->GetComm());
    J_minres->SetRelTol(m_lin_params.rel_tol);
    J_minres->SetAbsTol(0.0);
    J_minres->SetMaxIter(m_lin_params.max_iter);
    J_minres->SetPrintLevel(m_lin_params.print_level);
    J_minres->SetPreconditioner(*m_J_prec);
    m_J_solver = J_minres;
 }

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    m_nonlinear_oper = new NonlinearSolidQuasiStaticOperator(m_H_form);
  }
  else {
    m_nonlinear_oper = new NonlinearSolidDynamicOperator(m_H_form, m_M_form, m_S_form, m_ess_tdof_list);
  }

  // Set the newton solve parameters
  m_newton_solver.iterative_mode = true;
  m_newton_solver.SetSolver(*m_J_solver);
  m_newton_solver.SetOperator(*m_nonlinear_oper);
  m_newton_solver.SetPrintLevel(m_lin_params.print_level);
  m_newton_solver.SetRelTol(m_lin_params.rel_tol);
  m_newton_solver.SetAbsTol(m_lin_params.abs_tol);
  m_newton_solver.SetMaxIter(m_lin_params.max_iter);

}

// Solve the Quasi-static Newton system
void NonlinearSolidSolver::QuasiStaticSolve()
{
  mfem::Vector zero;
  m_newton_solver.Mult(zero, *m_state[0].true_vec);
}

// Advance the timestep
void NonlinearSolidSolver::AdvanceTimestep(__attribute__((unused)) double &dt)
{
  // Initialize the true vector
  m_state[0].gf->GetTrueDofs(*m_state[0].true_vec);

  if (m_timestepper == TimestepMethod::QuasiStatic) {
    QuasiStaticSolve();
  } else {
    mfem::mfem_error("Only quasistatics implemented for nonlinear solid mechanics!");
  }

  // Distribute the shared DOFs
  m_state[0].gf->SetFromTrueDofs(*m_state[0].true_vec);
  m_cycle += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver() 
{
  delete m_H_form;
  delete m_nonlinear_oper;
  delete m_J_solver;
  delete m_J_prec;
  delete m_model;
}

NonlinearSolidQuasiStaticOperator::NonlinearSolidQuasiStaticOperator(mfem::ParNonlinearForm *H_form)
  : mfem::Operator(H_form->FESpace()->GetTrueVSize())
{
  m_H_form = H_form;
}

// compute: y = H(x,p)
void NonlinearSolidQuasiStaticOperator::Mult(const mfem::Vector &k, mfem::Vector &y) const
{
  // Apply the nonlinear form
  m_H_form->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
mfem::Operator &NonlinearSolidQuasiStaticOperator::GetGradient(const mfem::Vector &x) const
{
  m_Jacobian = &m_H_form->GetGradient(x);
  return *m_Jacobian;
}

// destructor
NonlinearSolidQuasiStaticOperator::~NonlinearSolidQuasiStaticOperator()
{}

