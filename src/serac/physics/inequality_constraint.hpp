// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics.hpp
 *
 * @brief An object for representing inequality constraints
 */

#pragma once

#include "mfem.hpp"

#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"

namespace serac {

struct LevelSet {
  double y_height = 0.1;

  // negative mean interior
  double evaluate(const mfem::Vector& x) const { return x[1] - y_height; }

  mfem::Vector gradient(const mfem::Vector& x) const
  {
    mfem::Vector grad = x;
    grad              = 0.0;
    grad[1]           = 1.0;
    return grad;
  }

  mfem::Vector hess_vec(const mfem::Vector& x, const mfem::Vector& v) const
  {
    mfem::Vector hv = x;
    hv              = v;
    hv              = 0.0;
    return hv;
  }
};

template <int order, int dim>
struct InequalityConstraint {
  InequalityConstraint(std::string physics_name, std::string mesh_tag)
      : constraint_(StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint"), mesh_tag)),
        constraint_multiplier_(
            StateManager::newDual(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_multiplier"), mesh_tag)),
        constraint_penalty_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_penalty"), mesh_tag)),
        constraint_npc_error_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_npc_error"), mesh_tag)),
        constraint_diagonal_stiffness_(StateManager::newState(
            H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_diagonal_stiffness"), mesh_tag))
  {
    constraint_                    = 0.0;
    constraint_multiplier_         = 0.0;
    constraint_penalty_            = 0.0;
    constraint_npc_error_          = 0.0;
    constraint_diagonal_stiffness_ = 0.0;
  }

  void updateMultipliers(const FiniteElementVector& x_current)
  {
    const int rows     = x_current.Size();
    const int numNodes = rows / dim;
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x_current[dim * n + i];
      }
      double c       = levelSet_.evaluate(currentCoords);
      constraint_[n] = c;

      double lam = constraint_multiplier_[n];
      double k   = constraint_penalty_[n];

      // update multiplier
      constraint_multiplier_[n] = std::max(lam - k * c, 0.0);
    }

    // kappa = alObjective.kappa
    // alObjective.lam = np.maximum(alObjective.lam-kappa*c, 0.0)

    // ncpError = np.abs( alObjective.ncp(x) )
    //# check if each constraint is making progress, or if they are already small relative to the specificed constraint
    //tolerances poorProgress = ncpError > np.maximum(alSettings.target_constraint_decrease_factor * ncpErrorOld, 10 *
    // alSettings.tol / np.sqrt(len(ncpError)))
  }

  void sumConstraintResidual(const FiniteElementVector& x_current, mfem::Vector& res)
  {
    const int sz       = x_current.Size();
    const int numNodes = sz / dim;
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x_current[dim * n + i];
      }
      double c       = levelSet_.evaluate(currentCoords);
      constraint_[n] = c;
      double lam     = constraint_multiplier_[n];
      double k       = constraint_penalty_[n];
      auto   gradC   = levelSet_.gradient(currentCoords);

      double* resVec = &res[dim * n];
      if (lam >= k * c) {
        for (int i = 0; i < dim; ++i) {
          resVec[i] += gradC[i] * (-lam + k * c);
        }
      }
    }
  }

  std::unique_ptr<mfem::HypreParMatrix> sumConstraintJacobian(const FiniteElementVector&            x_current,
                                                              std::unique_ptr<mfem::HypreParMatrix> J)
  {
    const int sz       = x_current.Size();
    const int numNodes = sz / dim;
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x_current[dim * n + i];
      }
      double c         = levelSet_.evaluate(currentCoords);
      constraint_[n]   = c;
      double  lam      = constraint_multiplier_[n];
      double  k        = constraint_penalty_[n];
      auto    gradC    = levelSet_.gradient(currentCoords);
      double* diagHess = &constraint_diagonal_stiffness_[dim * n];

      if (lam >= k * c) {
        for (int i = 0; i < dim; ++i) {
          diagHess[i] += k * gradC[i] * gradC[i];  // ignore the actually hessian term for now * (-lam + k * c);
        }
      }
    }

    hypre_ParCSRMatrix* J_hype(*J);

    int   size       = J->Height();
    auto* Adiag_data = hypre_CSRMatrixData(J_hype->diag);
    auto* Adiag_i    = hypre_CSRMatrixI(J_hype->diag);

    printf("f\n");

    for (int i = 0; i < size; ++i) {
      auto diagIndex = Adiag_i[i];
      Adiag_data[diagIndex] += constraint_diagonal_stiffness_[i];
    }

    printf("g\n");

    // printf("f\n");
    // auto diagSparseMat = std::make_unique<mfem::SparseMatrix>(constraint_diagonal_stiffness_);
    // printf("g\n");
    // auto diagHypreMat = std::make_unique<mfem::HypreParMatrix>(constraint_diagonal_stiffness_.comm(),
    // J.GetGlobalNumRows(), J.GetRowStarts(), diagSparseMat.get()); printf("h\n"); printf("i,j\n"); printf("k\n");
    return J;
  }

protected:
  LevelSet levelSet_;

  FiniteElementState constraint_;
  FiniteElementDual  constraint_multiplier_;
  FiniteElementState constraint_penalty_;
  FiniteElementState constraint_npc_error_;
  FiniteElementState constraint_diagonal_stiffness_;
};

}  // namespace serac