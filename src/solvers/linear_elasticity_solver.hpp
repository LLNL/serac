// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef LINEARELASTIC_SOLVER
#define LINEARELASTIC_SOLVER

#include "mfem.hpp"

class LinearElasticSolver
{
protected:
  mfem::ParFiniteElementSpace &m_fe_space;
  mfem::ParBilinearForm *m_Kform;
  mfem::ParLinearForm *m_lform;

  /// Solver for the stiffness matrix
  mfem::Solver *m_K_solver;
  /// Preconditioner for the stiffness
  mfem::Solver *m_K_prec;

  mfem::Array<int> m_ess_tdof_list;

public:
  LinearElasticSolver(mfem::ParFiniteElementSpace &fes,
                      mfem::Array<int> &ess_bdr,
                      mfem::Array<int> &trac_bdr,
                      mfem::Coefficient &mu,
                      mfem::Coefficient &K,
                      mfem::VectorCoefficient &trac,
                      double rel_tol,
                      double abs_tol,
                      int iter,
                      bool gmres,
                      bool slu);

  /// Driver for the solver
  bool Solve(mfem::Vector &x) const;

  /// Get FE space
  const mfem::ParFiniteElementSpace *GetFESpace()
  {
    return &m_fe_space;
  }

  virtual ~LinearElasticSolver();
};

#endif
