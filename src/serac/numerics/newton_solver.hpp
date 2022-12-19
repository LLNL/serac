// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file newton_solver.hpp
 *
 * @brief This file contains the declaration of an modified version of MFEM's newton solver
 */

#include "mfem.hpp"

namespace serac::mfem_ext {

/// Newton's method for solving F(x)=b for a given operator F.
/** This is a slightly modified version of MFEM's native Newton
 * solver. By default, it forces one linear solve as opposed to only performing a
 * residual calculation if the initial residual is low.
 */
class NewtonSolver : public mfem::NewtonSolver {
public:
  /**
   * @brief Construct a new Newton Solver object
   *
   * @param communicator The MPI communicator for the global reductions in the solver
   * @param force_linear_solve Flag to force the newton solver to perform at least one linear solve
   */
  NewtonSolver(MPI_Comm communicator, bool force_linear_solve = true)
      : mfem::NewtonSolver(communicator), force_linear_solve_(force_linear_solve)
  {
  }

  /// Solve the nonlinear system with right-hand side @a b.
  /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
  void Mult(const mfem::Vector& b, mfem::Vector& x) const override;

protected:
  /// @brief Flag denoting whether to force a single linear solve even if the residual is low at iteration zero
  bool force_linear_solve_;
};

}  // namespace serac::mfem_ext