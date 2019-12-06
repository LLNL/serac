// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#ifndef CONDUCTION_SOLVER
#define CONDUCTION_SOLVER

#include "mfem.hpp"

// After spatial discretization, the conduction model can be written as:
//
//     du/dt = M^{-1}(-Ku)
//
//  where u is the vector representing the temperature, M is the mass matrix,
//  and K is the diffusion opeperator.
//
//  Class ConductionSolver represents the right-hand side of the above ODE.
 
class ConductionSolver : public mfem::TimeDependentOperator
{
protected:
   mfem::ParFiniteElementSpace &m_fe_space;
   mfem::Array<int> m_ess_tdof_list;
   mfem::ParBilinearForm *m_M_form;
   mfem::ParBilinearForm *m_K_form;

   mfem::HypreParMatrix m_M_mat;
   mfem::HypreParMatrix m_K_mat;
   mfem::HypreParMatrix *m_T_mat; // T = M + dt K
   double m_current_dt;

   mfem::CGSolver m_M_solver;    // Krylov solver for inverting the mass matrix M
   mfem::HypreSmoother m_M_prec; // Preconditioner for the mass matrix M

   mfem::CGSolver m_T_solver;    // Implicit solver for T = M + dt K
   mfem::HypreSmoother m_T_prec; // Preconditioner for the implicit solver

   double m_kappa;

   mutable mfem::Vector m_z; // auxiliary vector

   
public:
   ConductionSolver(mfem::ParFiniteElementSpace &f, double kappa);

   virtual void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const mfem::Vector &u, mfem::Vector &k);

   virtual ~ConductionSolver();
   
   
};


#endif
