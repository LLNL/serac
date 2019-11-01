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
                             double rel_tol,
                             double abs_tol,
                             int iter,
                             bool gmres,
                             bool slu)
   : TimeDependentOperator(2*fes.TrueVSize(), 0.0), fe_space(fes),
     newton_solver(fes.GetComm()),
     z(height/2)
{

   Array<int> ess_tdof_list;
   
   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);

   Mform = new ParBilinearForm(&fes);
   
   Mform->AddDomainIntegrator(new VectorMassIntegrator(rho0));
   Mform->Assemble();
   Mform->Finalize();
   Mmat = Mform->ParallelAssemble();
   fe_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   HypreParMatrix *Me = Mmat->EliminateRowsCols(ess_tdof_list);
   delete Me;

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(300);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*Mmat);

   model = new NeoHookeanModel(mu, K);
   Hform->AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
   Hform->SetEssentialTrueDofs(ess_tdof_list);

   reduced_oper = new ReducedSystemOperator(Mform, Hform, ess_tdof_list);

   if (gmres) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      prec_amg->SetPrintLevel(0);
      prec_amg->SetElasticityOptions(&fe_space);
      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      J_gmres->SetRelTol(rel_tol);
      J_gmres->SetAbsTol(1e-12);
      J_gmres->SetMaxIter(300);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres; 

   } 
   // retain super LU solver capabilities
   else if (slu) { 
      SuperLUSolver *superlu = NULL;
      superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(false);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      
      J_solver = superlu;
      J_prec = NULL;
   }
   else {
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(rel_tol);
      J_minres->SetAbsTol(0.0);
      J_minres->SetMaxIter(300);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;

   }

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

void DynamicSolver::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   Hform->Mult(x, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

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
   //    kv = -M^{-1}*[H(x + dt*kx)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // reduced_oper. This equation is solved with the newton_solver
   // object (using J_solver and J_prec internally).
   reduced_oper->SetParameters(dt, &v, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton_solver.Mult(zero, dv_dt);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");
   add(v, dt, dv_dt, dx_dt);
}


DynamicSolver::~DynamicSolver()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}


ReducedSystemOperator::ReducedSystemOperator(
   ParBilinearForm *M_, ParNonlinearForm *H_,
   const Array<int> &ess_tdof_list_)
   : Operator(M_->ParFESpace()->TrueVSize()), M(M_), H(H_),
     Jacobian(NULL), dt(0.0), v(NULL), x(NULL), w(height), z(height),
     ess_tdof_list(ess_tdof_list_)
{ }

void ReducedSystemOperator::SetParameters(double dt_, const Vector *v_,
                                          const Vector *x_)
{
   dt = dt_;  v = v_;  x = x_;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   H->Mult(z, y);
   M->TrueAddMult(k, y);
   y.SetSubVector(ess_tdof_list, 0.0);
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   delete Jacobian;
   SparseMatrix *localJ = &M->SpMat();
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   localJ->Add(dt*dt, H->GetLocalGradient(z));
   Jacobian = M->ParallelAssemble(localJ);
   delete localJ;
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);
   delete Je;
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Jacobian;
}
