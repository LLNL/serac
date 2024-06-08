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
  double y_height = 0.85;

  static constexpr double xAlpha = 0.1;

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const mfem::Vector& x) const { return y_height - x[1] - xAlpha * x[0]; }

  mfem::Vector gradient(const mfem::Vector& x) const
  {
    mfem::Vector grad = x;
    grad              = 0.0;
    grad[0]           = -xAlpha;
    grad[1]           = -1.0;
    return grad;
  }

  mfem::Vector hess_vec(const mfem::Vector& x, const mfem::Vector&) const
  {
    mfem::Vector hv = x;
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
            H1<order, dim * dim>{}, detail::addPrefix(physics_name, "constraint_diagonal_stiffness"), mesh_tag))
  {
    reset();
  }

  void reset()
  {
    constraint_            = 0.0;
    constraint_multiplier_ = 0.0;
    constraint_penalty_    = 0.125;
    constraint_npc_error_  = 0.1 * std::numeric_limits<double>::max();
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

      const double c           = levelSet_.evaluate(currentCoords);
      constraint_[n]           = c;
      const double       lam   = constraint_multiplier_[n];
      const double       k     = constraint_penalty_[n];
      const mfem::Vector gradC = levelSet_.gradient(currentCoords);

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

    constraint_diagonal_stiffness_ = 0.0;

    mfem::Vector currentCoords(dim);  // switch to stack vectors eventually
    mfem::Vector xyz_dirs(dim);

    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x_current[dim * n + i];
      }
      const double c   = levelSet_.evaluate(currentCoords);
      constraint_[n]   = c;
      const double lam = constraint_multiplier_[n];
      const double k   = constraint_penalty_[n];

      double* diagHess = &constraint_diagonal_stiffness_[dim * dim * n];

      if (lam >= k * c) {
        const mfem::Vector gradC = levelSet_.gradient(currentCoords);
        for (int i = 0; i < dim; ++i) {
          xyz_dirs                 = 0.0;
          xyz_dirs[i]              = 1.0;
          const mfem::Vector hessI = levelSet_.hess_vec(currentCoords, xyz_dirs);
          for (int j = 0; j < dim; ++j) {
            diagHess[dim * i + j] += k * gradC[i] * gradC[j] + hessI[j] * (-lam + k * c);
          }
        }
      }
    }

    hypre_ParCSRMatrix* J_hype(*J);

    auto*       Jdiag_data = hypre_CSRMatrixData(J_hype->diag);
    const auto* Jdiag_i    = hypre_CSRMatrixI(J_hype->diag);
    const auto* Jdiag_j    = hypre_CSRMatrixJ(J_hype->diag);

    J->Print("first.txt");

    std::array<int, dim> nodalCols;
    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        nodalCols[i] = dim * n + i;
      }

      double* diagHess = &constraint_diagonal_stiffness_[dim * dim * n];

      for (int i = 0; i < dim; ++i) {
        int  row      = dim * n + i;
        auto rowStart = Jdiag_i[row];
        auto rowEnd   = Jdiag_i[row + 1];
        for (auto colInd = rowStart; colInd < rowEnd; ++colInd) {
          int   col = Jdiag_j[colInd];
          auto& val = Jdiag_data[colInd];
          for (int j = 0; j < dim; ++j) {
            if (col == nodalCols[j]) {
              val += diagHess[dim * i + j];
            }
          }
        }
      }
    }
    J->Print("second.txt");

    // auto diagSparseMat = std::make_unique<mfem::SparseMatrix>(constraint_diagonal_stiffness_);
    // auto diagHypreMat = std::make_unique<mfem::HypreParMatrix>(constraint_diagonal_stiffness_.comm(),
    // J.GetGlobalNumRows(), J.GetRowStarts(), diagSparseMat.get()); printf("h\n"); printf("i,j\n"); printf("k\n");
    return J;
  }

  void updateMultipliers(const FiniteElementVector& x_current)
  {
    printf("updating multipliers\n");
    const int rows     = x_current.Size();
    const int numNodes = rows / dim;
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    double target_decrease_factor = 0.8;

    auto fischer_burmeister_npc_error = [](double c, double lam, double k) {
      double ck = c * k;
      return std::sqrt(ck * ck + lam * lam) - ck - lam;
    };

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x_current[dim * n + i];
      }
      const double c = levelSet_.evaluate(currentCoords);
      constraint_[n] = c;

      const double lam = constraint_multiplier_[n];
      const double k   = constraint_penalty_[n];
      // update multiplier
      constraint_multiplier_[n] = std::max(lam - k * c, 0.0);

      double oldError          = constraint_npc_error_[n];
      double newError          = std::abs(fischer_burmeister_npc_error(c, lam, k));
      constraint_npc_error_[n] = newError;

      bool poorProgress = newError > target_decrease_factor * oldError;

      if (poorProgress) constraint_penalty_[n] *= 1.1;

      // if (np.any(poorProgress) and solverSuccess):
      //   print('Poor progress on ncp detected, increasing some penalty parameters')
      //   alObjective.kappa = kappa.at[poorProgress].set(alSettings.penalty_scaling*kappa[poorProgress])
    }

    std::cout << "npc error = " << constraint_npc_error_.Norml2() << std::endl;

    // ncpError = np.abs( alObjective.ncp(x) )
    //# check if each constraint is making progress, or if they are already small relative to the specificed constraint
    // tolerances poorProgress = ncpError > np.maximum(alSettings.target_constraint_decrease_factor * ncpErrorOld, 10 *
    // alSettings.tol / np.sqrt(len(ncpError)))
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