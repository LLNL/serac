// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#include "dynamic_solver.hpp"

using namespace mfem;

DynamicSolver::DynamicSolver(ParFiniteElementSpace &fes,
                             Array<int> &ess_bdr,
                             double mu,
                             double K,
                             double visc,
                             double rel_tol,
                             double abs_tol,
                             int iter,
                             bool gmres,
                             bool slu)
   : TimeDependentOperator(2*fes.TrueVSize(), 0.0), m_fe_space(fes), m_viscosity(visc),
     m_M_solver(fes.GetComm()), m_newton_solver(fes.GetComm()),
     m_z(height/2)
{   
   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);

   m_M_form = new ParBilinearForm(&fes);
   
   m_M_form->AddDomainIntegrator(new VectorMassIntegrator(rho0));
   m_M_form->Assemble(0);
   m_M_form->Finalize(0);
   m_M_mat = m_M_form->ParallelAssemble();
   m_fe_space.GetEssentialTrueDofs(ess_bdr, m_ess_tdof_list);
   HypreParMatrix *Me = m_M_mat->EliminateRowsCols(m_ess_tdof_list);
   delete Me;

   m_S_form = new ParBilinearForm(&fes);
   ConstantCoefficient visc_coeff(m_viscosity);
   m_S_form->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   m_S_form->Assemble(0);
   m_S_form->Finalize(0);
   
   m_M_solver.iterative_mode = false;
   m_M_solver.SetRelTol(rel_tol);
   m_M_solver.SetAbsTol(abs_tol);
   m_M_solver.SetMaxIter(300);
   m_M_solver.SetPrintLevel(0);
   m_M_prec.SetType(HypreSmoother::Jacobi);
   m_M_solver.SetPreconditioner(m_M_prec);
   m_M_solver.SetOperator(*m_M_mat);

   // Define the parallel nonlinear form 
   m_H_form = new ParNonlinearForm(&fes);
   m_model = new NeoHookeanModel(mu, K);   
   m_H_form->AddDomainIntegrator(new HyperelasticNLFIntegrator(m_model));
   m_H_form->SetEssentialTrueDofs(m_ess_tdof_list);
   
   m_reduced_oper = new ReducedSystemOperator(m_M_form, m_S_form, m_H_form, m_ess_tdof_list);

   if (gmres) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&m_fe_space);
      m_J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(m_fe_space.GetComm());
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*m_J_prec);
      m_J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      SuperLUSolver *superlu = nullptr;
      superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      
      m_J_solver = superlu;
      m_J_prec = nullptr;
   }
   else {
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      m_J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(m_fe_space.GetComm());
      J_minres->SetRelTol(rel_tol);
      J_minres->SetAbsTol(0.0);
      J_minres->SetMaxIter(300);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*m_J_prec);
      m_J_solver = J_minres;

   }

   // Set the newton solve parameters
   m_newton_solver.iterative_mode = false;
   m_newton_solver.SetSolver(*m_J_solver);
   m_newton_solver.SetOperator(*m_reduced_oper);
   m_newton_solver.SetPrintLevel(1); 
   m_newton_solver.SetRelTol(rel_tol);
   m_newton_solver.SetAbsTol(abs_tol);
   m_newton_solver.SetMaxIter(iter);
}

void DynamicSolver::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   m_H_form->Mult(x, m_z);
   if (m_viscosity != 0.0)
   {
      m_S_form->TrueAddMult(v, m_z);
      m_z.SetSubVector(m_ess_tdof_list, 0.0);
   }
   m_z.Neg(); // z = -z
   m_M_solver.Mult(m_z, dv_dt);

   dx_dt = v;
}

void DynamicSolver::ImplicitSolve(const double dt,
                                  const Vector &vx, Vector &dvx_dt)
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   // By eliminating kx from the coupled system:
   //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // m_reduced_oper. This equation is solved with the m_newton_solver
   // object (using m_J_solver and m_J_prec internally).
   m_reduced_oper->SetParameters(dt, &v, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   m_newton_solver.Mult(zero, dv_dt);
   MFEM_VERIFY(m_newton_solver.GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}


DynamicSolver::~DynamicSolver()
{
   delete m_J_solver;
   if (m_J_prec != nullptr) {
      delete m_J_prec;
   }
   delete m_reduced_oper;
   delete m_model;
   delete m_M_mat;
   delete m_M_form;
   delete m_H_form;
   delete m_S_form;
}


ReducedSystemOperator::ReducedSystemOperator(
   ParBilinearForm *M, ParBilinearForm *S, ParNonlinearForm *H,
   const Array<int> &ess_tdof_list)
   : Operator(M->ParFESpace()->TrueVSize()), m_M_form(M), m_S_form(S), m_H_form(H),
     m_jacobian(nullptr), m_dt(0.0), m_v(nullptr), m_x(nullptr), m_w(height), m_z(height),
     m_ess_tdof_list(ess_tdof_list)
{ }

void ReducedSystemOperator::SetParameters(double dt, const Vector *v,
                                          const Vector *x)
{   
   m_dt = dt;  m_v = v;  m_x = x;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*m_v, m_dt, k, m_w);
   add(*m_x, m_dt, m_w, m_z);
   m_H_form->Mult(m_z, y);
   m_M_form->TrueAddMult(k, y);
   m_S_form->TrueAddMult(m_w, y);
   y.SetSubVector(m_ess_tdof_list, 0.0);
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   delete m_jacobian;
   SparseMatrix *localJ = Add(1.0, m_M_form->SpMat(), m_dt, m_S_form->SpMat());
   add(*m_v, m_dt, k, m_w);
   add(*m_x, m_dt, m_w, m_z);
   localJ->Add(m_dt*m_dt, m_H_form->GetLocalGradient(m_z));
   m_jacobian = m_M_form->ParallelAssemble(localJ);
   delete localJ;
   HypreParMatrix *Je = m_jacobian->EliminateRowsCols(m_ess_tdof_list);
   delete Je;
   return *m_jacobian;   
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete m_jacobian;
}
