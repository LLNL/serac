// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "conduction_solver.hpp"

ConductionSolver::ConductionSolver(mfem::ParFiniteElementSpace &f, double kap)
   : mfem::TimeDependentOperator(f.GetTrueVSize(), 0.0), m_fe_space(f), m_M_form(nullptr), m_K_form(nullptr),
     m_T_mat(nullptr), m_current_dt(0.0),
     m_M_solver(f.GetComm()), m_T_solver(f.GetComm()), m_z(height)
{
   const double rel_tol = 1e-8;
   m_kappa = kap;

   m_M_form = new mfem::ParBilinearForm(&m_fe_space);
   m_M_form->AddDomainIntegrator(new mfem::MassIntegrator());
   m_M_form->Assemble(0); // keep sparsity pattern of M and K the same
   m_M_form->FormSystemMatrix(m_ess_tdof_list, m_M_mat);

   mfem::ConstantCoefficient kappa_coef(m_kappa);

   m_K_form = new mfem::ParBilinearForm(&m_fe_space);
   m_K_form->AddDomainIntegrator(new mfem::DiffusionIntegrator(kappa_coef));
   m_K_form->Assemble(0); // keep sparsity pattern of M and K the same
   m_K_form->FormSystemMatrix(m_ess_tdof_list, m_K_mat);

   m_M_solver.iterative_mode = false;
   m_M_solver.SetRelTol(rel_tol);
   m_M_solver.SetAbsTol(0.0);
   m_M_solver.SetMaxIter(100);
   m_M_solver.SetPrintLevel(0);
   m_M_prec.SetType(mfem::HypreSmoother::Jacobi);
   m_M_solver.SetPreconditioner(m_M_prec);
   m_M_solver.SetOperator(m_M_mat);

   m_T_solver.iterative_mode = false;
   m_T_solver.SetRelTol(rel_tol);
   m_T_solver.SetAbsTol(0.0);
   m_T_solver.SetMaxIter(100);
   m_T_solver.SetPrintLevel(0);
   m_T_solver.SetPreconditioner(m_T_prec);

}

void ConductionSolver::Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-K(u)
   // for du_dt
   m_K_mat.Mult(u, m_z);
   m_z.Neg(); // z = -z
   m_M_solver.Mult(m_z, du_dt);
}

void ConductionSolver::ImplicitSolve(const double dt,
                                       const mfem::Vector &u, mfem::Vector &du_dt)
{
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

ConductionSolver::~ConductionSolver()
{
   delete m_M_form;
   delete m_K_form;
}


