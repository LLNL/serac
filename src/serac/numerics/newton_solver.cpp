// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/numerics/newton_solver.hpp"
#include "serac/infrastructure/logger.hpp"

namespace serac::mfem_ext {

void NewtonSolver::Mult(const mfem::Vector& b, mfem::Vector& x) const
{
  MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
  MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

  int        it;
  double     norm0, norm, norm_goal;
  const bool have_b = (b.Size() == Height());

  if (!iterative_mode) {
    x = 0.0;
  }

  ProcessNewState(x);

  oper->Mult(x, r);
  if (have_b) {
    r -= b;
  }

  norm0 = norm = Norm(r);
  if (print_options.first_and_last && !print_options.iterations) {
    mfem::out << "Newton iteration " << std::setw(2) << 0 << " : ||r|| = " << norm << "...\n";
  }
  norm_goal = std::max(rel_tol * norm, abs_tol);

  prec->iterative_mode = false;

  // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
  for (it = 0; true; it++) {
    SLIC_ERROR_ROOT_IF(!mfem::IsFinite(norm), axom::fmt::format("norm = {}", norm));
    if (print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << it << " : ||r|| = " << norm;
      if (it > 0) {
        mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
      }
      mfem::out << '\n';
    }
    Monitor(it, norm, r, x);

    if (norm <= norm_goal) {
      // Ensure that at least 1 linear solve occurs for each Newton solve
      if (it > 0 || !force_linear_solve_) {
        converged = true;
        break;
      }
    }

    if (it >= max_iter) {
      converged = false;
      break;
    }

    grad = &oper->GetGradient(x);
    prec->SetOperator(*grad);

    if (lin_rtol_type) {
      AdaptiveLinRtolPreSolve(x, it, norm);
    }

    prec->Mult(r, c);  // c = [DF(x_i)]^{-1} [F(x_i)-b]

    if (lin_rtol_type) {
      AdaptiveLinRtolPostSolve(c, r, it, norm);
    }

    const double c_scale = ComputeScalingFactor(x, b);
    if (c_scale == 0.0) {
      converged = false;
      break;
    }
    add(x, -c_scale, c, x);

    ProcessNewState(x);

    oper->Mult(x, r);
    if (have_b) {
      r -= b;
    }
    norm = Norm(r);
  }

  final_iter = it;
  final_norm = norm;

  if (print_options.summary || (!converged && print_options.warnings) || print_options.first_and_last) {
    mfem::out << "Newton: Number of iterations: " << final_iter << '\n' << "   ||r|| = " << final_norm << '\n';
  }
  if (!converged && (print_options.summary || print_options.warnings)) {
    mfem::out << "Newton: No convergence!\n";
  }
}

}  // namespace serac::mfem_ext
