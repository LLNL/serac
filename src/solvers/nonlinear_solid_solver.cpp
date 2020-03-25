// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "solvers/nonlinear_solid_solver.hpp"

#include "integrators/inc_hyperelastic_integrator.hpp"
#include "integrators/hyperelastic_traction_integrator.hpp"

constexpr int num_fields = 2;

NonlinearSolidSolver::NonlinearSolidSolver(int order, mfem::ParMesh *pmesh)
   : BaseSolver(num_fields)
   , displacement(m_states[0])
   , velocity(m_states[1])
   , m_H_form(nullptr)
   , m_M_form(nullptr)
   , m_S_form(nullptr)
   , m_nonlinear_oper(nullptr)
   , m_timedep_oper(nullptr)
   , m_newton_solver(pmesh->GetComm())
   , m_J_solver(nullptr)
   , m_J_prec(nullptr)
   , m_viscosity(nullptr)
   , m_model(nullptr)
{

   displacement.mesh  = pmesh;
   displacement.coll  = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
   displacement.space = std::make_shared<mfem::ParFiniteElementSpace>(pmesh,
                                                                      displacement.coll.get(),
                                                                      pmesh->Dimension(),
                                                                      mfem::Ordering::byVDIM);
   displacement.gf    = std::make_shared<mfem::ParGridFunction>(displacement.space.get());
   *displacement.gf   = 0.0;
   displacement.name  = "displacement";

   velocity.mesh  = pmesh;
   velocity.coll  = std::make_shared<mfem::H1_FECollection>(order, pmesh->Dimension());
   velocity.space = std::make_shared<mfem::ParFiniteElementSpace>(pmesh,
                                                                  velocity.coll.get(),
                                                                  pmesh->Dimension(),
                                                                  mfem::Ordering::byVDIM);
   velocity.gf    = std::make_shared<mfem::ParGridFunction>(velocity.space.get());
   *velocity.gf   = 0.0;
   velocity.name  = "velocity";

   // Initialize the true DOF vector
   int true_size = velocity.space->TrueVSize();
   mfem::Array<int> true_offset(3);
   true_offset[0] = 0;
   true_offset[1] = true_size;
   true_offset[2] = 2 * true_size;
   m_block        = new mfem::BlockVector(true_offset);

   m_block->GetBlockView(1, displacement.true_vec);
   displacement.true_vec = 0.0;

   m_block->GetBlockView(0, velocity.true_vec);
   velocity.true_vec = 0.0;

}

void NonlinearSolidSolver::SetDisplacementBCs(mfem::Array<int> &disp_bdr,
                                              mfem::VectorCoefficient *disp_bdr_coef)
{
   SetEssentialBCs(disp_bdr, disp_bdr_coef);

   // Get the list of essential DOFs
   displacement.space->GetEssentialTrueDofs(disp_bdr, m_ess_tdof_list);
}

void NonlinearSolidSolver::SetTractionBCs(mfem::Array<int> &trac_bdr,
                                          mfem::VectorCoefficient *trac_bdr_coef)
{
   SetNaturalBCs(trac_bdr, trac_bdr_coef);
}

void NonlinearSolidSolver::SetHyperelasticMaterialParameters(double mu, double K)
{
   delete m_model;
   m_model = new mfem::NeoHookeanModel(mu, K);
}

void NonlinearSolidSolver::SetViscosity(mfem::Coefficient *visc) { m_viscosity = visc; }

void NonlinearSolidSolver::SetInitialState(mfem::VectorCoefficient &disp_state,
                                           mfem::VectorCoefficient &velo_state)
{
   disp_state.SetTime(m_time);
   velo_state.SetTime(m_time);
   displacement.gf->ProjectCoefficient(disp_state);
   velocity.gf->ProjectCoefficient(velo_state);
   m_gf_initialized = true;
}

void NonlinearSolidSolver::SetSolverParameters(const LinearSolverParameters &lin_params,
                                               const NonlinearSolverParameters &nonlin_params)
{
   m_lin_params    = lin_params;
   m_nonlin_params = nonlin_params;
}

void NonlinearSolidSolver::CompleteSetup()
{
   // Define the nonlinear form
   m_H_form = new mfem::ParNonlinearForm(displacement.space.get());

   // Add the hyperelastic integrator
   if (m_timestepper == TimestepMethod::QuasiStatic)
   {
      m_H_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(m_model));
   }
   else
   {
      m_H_form->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(m_model));
   }

   // Add the traction integrator
   if (m_nat_bdr_vec_coef != nullptr)
   {
      m_H_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(*m_nat_bdr_vec_coef),
                                     m_nat_bdr);
   }

   // Add the essential boundary
   if (m_ess_bdr_vec_coef != nullptr)
   {
      m_H_form->SetEssentialBC(m_ess_bdr);
   }

   // If dynamic, create the mass and viscosity forms
   if (m_timestepper != TimestepMethod::QuasiStatic)
   {
      const double ref_density = 1.0;  // density in the reference configuration
      mfem::ConstantCoefficient rho0(ref_density);

      m_M_form = new mfem::ParBilinearForm(velocity.space.get());

      m_M_form->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho0));
      m_M_form->Assemble(0);
      m_M_form->Finalize(0);

      m_S_form = new mfem::ParBilinearForm(velocity.space.get());
      m_S_form->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(*m_viscosity));
      m_S_form->Assemble(0);
      m_S_form->Finalize(0);
   }

   // Set up the jacbian solver based on the linear solver options
   if (m_lin_params.prec == Preconditioner::BoomerAMG)
   {
      MFEM_VERIFY(displacement.space->GetOrdering() == mfem::Ordering::byVDIM,
                  "Attempting to use BoomerAMG with nodal ordering.");
      mfem::HypreBoomerAMG *prec_amg = new mfem::HypreBoomerAMG();
      prec_amg->SetPrintLevel(m_lin_params.print_level);
      prec_amg->SetElasticityOptions(displacement.space.get());
      m_J_prec = prec_amg;

      mfem::GMRESSolver *J_gmres = new mfem::GMRESSolver(displacement.space->GetComm());
      J_gmres->SetRelTol(m_lin_params.rel_tol);
      J_gmres->SetAbsTol(m_lin_params.abs_tol);
      J_gmres->SetMaxIter(m_lin_params.max_iter);
      J_gmres->SetPrintLevel(m_lin_params.print_level);
      J_gmres->SetPreconditioner(*m_J_prec);
      m_J_solver = J_gmres;
   }
   else
   {
      mfem::HypreSmoother *J_hypreSmoother = new mfem::HypreSmoother;
      J_hypreSmoother->SetType(mfem::HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      m_J_prec = J_hypreSmoother;

      mfem::MINRESSolver *J_minres = new mfem::MINRESSolver(displacement.space->GetComm());
      J_minres->SetRelTol(m_lin_params.rel_tol);
      J_minres->SetAbsTol(m_lin_params.abs_tol);
      J_minres->SetMaxIter(m_lin_params.max_iter);
      J_minres->SetPrintLevel(m_lin_params.print_level);
      J_minres->SetPreconditioner(*m_J_prec);
      m_J_solver = J_minres;
   }

   // Set the newton solve parameters
   m_newton_solver.iterative_mode = true;
   m_newton_solver.SetSolver(*m_J_solver);
   m_newton_solver.SetPrintLevel(m_nonlin_params.print_level);
   m_newton_solver.SetRelTol(m_nonlin_params.rel_tol);
   m_newton_solver.SetAbsTol(m_nonlin_params.abs_tol);
   m_newton_solver.SetMaxIter(m_nonlin_params.max_iter);

   // Set the MFEM abstract operators for use with the internal MFEM solvers
   if (m_timestepper == TimestepMethod::QuasiStatic)
   {
      m_nonlinear_oper = new NonlinearSolidQuasiStaticOperator(m_H_form);
      m_newton_solver.SetOperator(*m_nonlinear_oper);
   }
   else
   {
      m_timedep_oper = new NonlinearSolidDynamicOperator(m_H_form,
                                                         m_S_form,
                                                         m_M_form,
                                                         m_ess_tdof_list,
                                                         &m_newton_solver,
                                                         m_lin_params);
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
   displacement.gf->GetTrueDofs(displacement.true_vec);
   velocity.gf->GetTrueDofs(velocity.true_vec);

   if (m_timestepper == TimestepMethod::QuasiStatic)
   {
      QuasiStaticSolve();
   }
   else
   {
      m_ode_solver->Step(*m_block, m_time, dt);
   }

   // Distribute the shared DOFs
   displacement.gf->SetFromTrueDofs(displacement.true_vec);
   velocity.gf->SetFromTrueDofs(velocity.true_vec);
   m_cycle += 1;
}

NonlinearSolidSolver::~NonlinearSolidSolver()
{
   delete m_H_form;
   delete m_nonlinear_oper;
   delete m_timedep_oper;
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
NonlinearSolidQuasiStaticOperator::~NonlinearSolidQuasiStaticOperator() {}

NonlinearSolidDynamicOperator::NonlinearSolidDynamicOperator(mfem::ParNonlinearForm *H_form,
                                                             mfem::ParBilinearForm *S_form,
                                                             mfem::ParBilinearForm *M_form,
                                                             const mfem::Array<int> &ess_tdof_list,
                                                             mfem::NewtonSolver *newton_solver,
                                                             LinearSolverParameters lin_params)
   : mfem::TimeDependentOperator(M_form->ParFESpace()->TrueVSize() * 2)
   , m_M_form(M_form)
   , m_S_form(S_form)
   , m_H_form(H_form)
   , m_M_mat(nullptr)
   , m_reduced_oper(nullptr)
   , m_newton_solver(newton_solver)
   , m_ess_tdof_list(ess_tdof_list)
   , m_lin_params(lin_params)
   , m_z(height / 2)
{
   // Assemble the mass matrix and eliminate the fixed DOFs
   m_M_mat                  = m_M_form->ParallelAssemble();
   mfem::HypreParMatrix *Me = m_M_mat->EliminateRowsCols(m_ess_tdof_list);
   delete Me;

   // Set the mass matrix solver options
   m_M_solver.iterative_mode = false;
   m_M_solver.SetRelTol(m_lin_params.rel_tol);
   m_M_solver.SetAbsTol(m_lin_params.abs_tol);
   m_M_solver.SetMaxIter(m_lin_params.max_iter);
   m_M_solver.SetPrintLevel(m_lin_params.print_level);
   m_M_prec.SetType(mfem::HypreSmoother::Jacobi);
   m_M_solver.SetPreconditioner(m_M_prec);
   m_M_solver.SetOperator(*m_M_mat);

   // Construct the reduced system operator and initialize the newton solver with it
   m_reduced_oper = new NonlinearSolidReducedSystemOperator(H_form, S_form, M_form, ess_tdof_list);
   m_newton_solver->SetOperator(*m_reduced_oper);
}

void NonlinearSolidDynamicOperator::Mult(const mfem::Vector &vx, mfem::Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height / 2;
   mfem::Vector v(vx.GetData() + 0, sc);
   mfem::Vector x(vx.GetData() + sc, sc);
   mfem::Vector dv_dt(dvx_dt.GetData() + 0, sc);
   mfem::Vector dx_dt(dvx_dt.GetData() + sc, sc);

   m_H_form->Mult(x, m_z);
   m_S_form->TrueAddMult(v, m_z);
   m_z.SetSubVector(m_ess_tdof_list, 0.0);
   m_z.Neg();  // z = -z
   m_M_solver.Mult(m_z, dv_dt);

   dx_dt = v;
}

void NonlinearSolidDynamicOperator::ImplicitSolve(const double dt,
                                                  const mfem::Vector &vx,
                                                  mfem::Vector &dvx_dt)
{
   int sc = height / 2;
   mfem::Vector v(vx.GetData() + 0, sc);
   mfem::Vector x(vx.GetData() + sc, sc);
   mfem::Vector dv_dt(dvx_dt.GetData() + 0, sc);
   mfem::Vector dx_dt(dvx_dt.GetData() + sc, sc);

   // By eliminating kx from the coupled system:
   //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // m_reduced_oper. This equation is solved with the m_newton_solver
   // object (using m_J_solver and m_J_prec internally).
   m_reduced_oper->SetParameters(dt, &v, &x);
   mfem::Vector zero;  // empty vector is interpreted as zero r.h.s. by NewtonSolver
   m_newton_solver->Mult(zero, dv_dt);
   MFEM_VERIFY(m_newton_solver->GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}

// destructor
NonlinearSolidDynamicOperator::~NonlinearSolidDynamicOperator() { delete m_M_mat; }

NonlinearSolidReducedSystemOperator::NonlinearSolidReducedSystemOperator(
   mfem::ParNonlinearForm *H_form,
   mfem::ParBilinearForm *S_form,
   mfem::ParBilinearForm *M_form,
   const mfem::Array<int> &ess_tdof_list)
   : mfem::Operator(M_form->ParFESpace()->TrueVSize())
   , m_M_form(M_form)
   , m_S_form(S_form)
   , m_H_form(H_form)
   , m_jacobian(nullptr)
   , m_dt(0.0)
   , m_v(nullptr)
   , m_x(nullptr)
   , m_w(height)
   , m_z(height)
   , m_ess_tdof_list(ess_tdof_list)
{}

void NonlinearSolidReducedSystemOperator::SetParameters(double dt,
                                                        const mfem::Vector *v,
                                                        const mfem::Vector *x)
{
   m_dt = dt;
   m_v  = v;
   m_x  = x;
}

void NonlinearSolidReducedSystemOperator::Mult(const mfem::Vector &k, mfem::Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*m_v, m_dt, k, m_w);
   add(*m_x, m_dt, m_w, m_z);
   m_H_form->Mult(m_z, y);
   m_M_form->TrueAddMult(k, y);
   m_S_form->TrueAddMult(m_w, y);
   y.SetSubVector(m_ess_tdof_list, 0.0);
}

mfem::Operator &NonlinearSolidReducedSystemOperator::GetGradient(const mfem::Vector &k) const
{
   delete m_jacobian;
   // Form the gradient of the complete nonlinear operator
   mfem::SparseMatrix *localJ = Add(1.0, m_M_form->SpMat(), m_dt, m_S_form->SpMat());
   add(*m_v, m_dt, k, m_w);
   add(*m_x, m_dt, m_w, m_z);
   localJ->Add(m_dt * m_dt, m_H_form->GetLocalGradient(m_z));
   m_jacobian = m_M_form->ParallelAssemble(localJ);
   delete localJ;

   // Eliminate the fixed boundary DOFs
   //
   // This call eliminates the appropriate DOFs in m_jacobian and returns the
   // eliminated DOFs in Je. We don't need this so it gets deleted.
   mfem::HypreParMatrix *Je = m_jacobian->EliminateRowsCols(m_ess_tdof_list);
   delete Je;
   return *m_jacobian;
}

NonlinearSolidReducedSystemOperator::~NonlinearSolidReducedSystemOperator() { delete m_jacobian; }
