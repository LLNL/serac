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
  double y_height = 0.9;//1.9;

  static constexpr double xAlpha = 0.0; //1.0;
  static constexpr double yAlpha = 1.0;

  // positive means constraint is satisfied, i.e., c(x) >= 0
  double evaluate(const mfem::Vector& x) const { return y_height - xAlpha * x[0] - yAlpha * x[1] ; }

  mfem::Vector gradient(const mfem::Vector& x) const
  {
    mfem::Vector grad = x;
    grad              = 0.0;
    grad[0]           = -xAlpha;
    grad[1]           = -yAlpha;
    return grad;
  }

  mfem::Vector hess_vec(const mfem::Vector& x, const mfem::Vector&) const
  {
    mfem::Vector hv = x;
    hv              = 0.0;
    return hv;
  }
};


template <int dim>
struct View {
  View(mfem::Vector& v_) : v(v_), offset(v.Size()/dim) {
  }

  double& operator[] (int i) { return v[i]; }
  double& operator() (int i) { return v[i]; }
  const double& operator[] (int i) const { return v[i]; }
  const double& operator() (int i) const { return v[i]; }

  //double& operator() (int i, int j) { return v[i + offset*j]; }
  //const double& operator() (int i, int j) const { return v[i + offset*j]; }

  double& operator() (int i, int j) { return v[dim * i + j]; }
  const double& operator() (int i, int j) const { return v[dim * i + j]; }

  int numNodes() const { return offset; }

  private:
  mfem::Vector& v;
  int offset;
};


template <int dim>
struct ConstView {
  ConstView(const mfem::Vector& v_) : v(v_), offset(v.Size()/dim) {
  }

  const double& operator[] (int i) const { return v[i]; }
  const double& operator() (int i) const { return v[i]; }
  //const double& operator() (int i, int j) const { return v[i + offset*j]; }
  const double& operator() (int i, int j) const { return v[dim * i + j]; }

  int numNodes() const { return offset; }

  private:
  const mfem::Vector& v;
  int offset;
};

template <int order, int dim>
struct InequalityConstraint {
  InequalityConstraint(std::string physics_name, std::string mesh_tag)
      : constraint_(StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint"), mesh_tag)),
        constraint_multiplier_(
            StateManager::newDual(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_multiplier"), mesh_tag)),
        constraint_penalty_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_penalty"), mesh_tag)),
        constraint_ncp_error_(
            StateManager::newState(H1<order, 1>{}, detail::addPrefix(physics_name, "constraint_ncp_error"), mesh_tag)),
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
    constraint_ncp_error_  = 0.1 * std::numeric_limits<double>::max();
  }

  void outputStateToDisk() const {
    StateManager::updateState(constraint_);
    StateManager::updateDual(constraint_multiplier_);
    StateManager::updateState(constraint_penalty_);
    StateManager::updateState(constraint_ncp_error_);
    StateManager::updateState(constraint_diagonal_stiffness_);
  }

  void sumConstraintResidual(const FiniteElementVector& x_current, mfem::Vector& res)
  {
    const int sz       = x_current.Size();
    const int numNodes = sz / dim;
    SLIC_ERROR_ROOT_IF(numNodes != constraint_.Size(), "Constraint size does not match system size.");

    View<1> constraint(constraint_);
    View<1> constraint_multiplier(constraint_multiplier_);
    View<1> constraint_penalty(constraint_penalty_);
    ConstView<dim> x(x_current);
    View<dim> residual(res);

    mfem::Vector currentCoords(dim);
    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x(n,i);
      }

      const double c   = levelSet_.evaluate(currentCoords);
      constraint[n]    = c;
      const double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];

      const mfem::Vector gradC = levelSet_.gradient(currentCoords);

      if (lam >= k * c) {
        for (int i = 0; i < dim; ++i) {
          residual(n,i) += gradC[i] * (-lam + k * c);
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

    View<1> constraint(constraint_);
    View<1> constraint_multiplier(constraint_multiplier_);
    View<1> constraint_penalty(constraint_penalty_);
    ConstView<dim> x(x_current);
    View<dim*dim> constraint_diagonal_stiffness(constraint_diagonal_stiffness_);

    mfem::Vector currentCoords(dim);  // switch to stack vectors eventually
    mfem::Vector xyz_dirs(dim);

    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x(n,i);
      }
      const double c   = levelSet_.evaluate(currentCoords);
      constraint[n]   = c;
      const double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];

      if (lam >= k * c) {
        const mfem::Vector gradC = levelSet_.gradient(currentCoords);
        for (int i = 0; i < dim; ++i) {
          xyz_dirs                 = 0.0;
          xyz_dirs[i]              = 1.0;
          const mfem::Vector hessI = levelSet_.hess_vec(currentCoords, xyz_dirs);
          for (int j = 0; j < dim; ++j) {
            constraint_diagonal_stiffness(n, dim*i+j) += k * gradC[i] * gradC[j] + hessI[j] * (-lam + k * c);
          }
        }
      }
    }

    hypre_ParCSRMatrix* J_hype(*J);

    auto*       Jdiag_data = hypre_CSRMatrixData(J_hype->diag);
    const auto* Jdiag_i    = hypre_CSRMatrixI(J_hype->diag);
    const auto* Jdiag_j    = hypre_CSRMatrixJ(J_hype->diag);

    // J->Print("first.txt");

    using array = std::array<int, dim>;
    using arrayInt = typename array::size_type;

    std::array<int, dim> nodalCols;
    for (int n = 0; n < numNodes; ++n) {
      for (int i = 0; i < dim; ++i) {
        //nodalCols[static_cast<arrayInt>(i)] = n + i * numNodes;
        nodalCols[static_cast<arrayInt>(i)] = dim * n + i;
      }

      for (int i = 0; i < dim; ++i) {
        int  row      = dim * n + i;
        auto rowStart = Jdiag_i[row];
        auto rowEnd   = Jdiag_i[row + 1];
        for (auto colInd = rowStart; colInd < rowEnd; ++colInd) {
          int   col = Jdiag_j[colInd];
          auto& val = Jdiag_data[colInd];
          for (int j = 0; j < dim; ++j) {
            if (col == nodalCols[static_cast<arrayInt>(j)]) {
              val += constraint_diagonal_stiffness(n, dim*i+j);
            }
          }
        }
      }
    }

    // J->Print("second.txt");

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

    double target_decrease_factor = 0.75;

    auto fischer_burmeister_ncp_error = [](double c, double lam, double k) {
      double ck = c * k;
      return std::sqrt(ck * ck + lam * lam) - ck - lam;
    };

    View<1> constraint(constraint_);
    View<1> constraint_multiplier(constraint_multiplier_);
    View<1> constraint_penalty(constraint_penalty_);
    View<1> constraint_ncp_error(constraint_ncp_error_);
    ConstView<dim> x(x_current);

    for (int n = 0; n < numNodes; ++n) {
      mfem::Vector currentCoords(dim);
      for (int i = 0; i < dim; ++i) {
        currentCoords[i] = x(n,i);
      }
      const double c = levelSet_.evaluate(currentCoords);
      constraint[n] = c;

      const double lam = constraint_multiplier[n];
      const double k   = constraint_penalty[n];
      // update multiplier
      constraint_multiplier[n] = std::max(lam - k * c, 0.0);

      double oldError         = constraint_ncp_error[n];
      double newError         = std::abs(fischer_burmeister_ncp_error(c, lam, k));
      constraint_ncp_error[n] = newError;

      bool poorProgress = newError > target_decrease_factor * oldError;

      if (poorProgress) constraint_penalty[n] *= 1.5;
    }

    std::cout << "ncp error = " << constraint_ncp_error_.Norml2() << std::endl;

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
  FiniteElementState constraint_ncp_error_;
  FiniteElementState constraint_diagonal_stiffness_;
};

}  // namespace serac