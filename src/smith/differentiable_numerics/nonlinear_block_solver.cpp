// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/solver_with_preconditioner.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "mfem.hpp"

namespace smith {

using smith::FiniteElementDual;
using smith::FiniteElementState;

/// @brief Utility to compute the matrix norm
double matrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
  mfem::HypreParMatrix* H = K.get();
  hypre_ParCSRMatrix* Hhypre = static_cast<hypre_ParCSRMatrix*>(*H);
  double Hfronorm;
  hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
  return Hfronorm;
}

/// @brief Utility to compute 0.5*norm(K-K.T)
double skewMatrixNorm(std::unique_ptr<mfem::HypreParMatrix>& K)
{
  auto K_T = std::unique_ptr<mfem::HypreParMatrix>(K->Transpose());
  K_T->Add(-1.0, *K);
  (*K_T) *= 0.5;
  mfem::HypreParMatrix* H = K_T.get();
  hypre_ParCSRMatrix* Hhypre = static_cast<hypre_ParCSRMatrix*>(*H);
  double Hfronorm;
  hypre_ParCSRMatrixNormFro(Hhypre, &Hfronorm);
  return Hfronorm;
}

std::unique_ptr<mfem::SparseMatrix> buildElementDofMap(const std::vector<std::shared_ptr<FiniteElementState>>& unknowns,
                                                       const mfem::Array<int>& block_offsets)
{
  SLIC_ERROR_IF(unknowns.empty(), "buildElementDofMap: received an empty field set");
  SLIC_ERROR_IF(block_offsets.Size() != static_cast<int>(unknowns.size()) + 1,
                "buildElementDofMap: Field count does not match block offsets");

  for (const auto& unknown : unknowns) {
    SLIC_ERROR_IF(!unknown, "buildElementDofMap: Null unknown field encountered");
    SLIC_ERROR_IF(dynamic_cast<const mfem::L2_FECollection*>(unknown->space().FEColl()) == nullptr,
                  "buildElementDofMap: Requires element-wise L2 discontinuous fields");
  }

  const auto* pmesh = unknowns[0]->space().GetParMesh();
  const int num_elements = unknowns[0]->space().GetNE();
  const int total_true_dofs = block_offsets.Last();

  auto element_dof_map = std::make_unique<mfem::SparseMatrix>(num_elements, total_true_dofs);

  for (size_t uid = 0; uid < unknowns.size(); ++uid) {
    const auto& space = unknowns[uid]->space();
    SLIC_ERROR_IF(space.GetParMesh() != pmesh, "buildElementDofMap: All fields must have the same mesh");
    SLIC_ERROR_IF(space.GetNE() != num_elements,
                  "buildElementDofMap: All fields must have the same number of elements");
    SLIC_ERROR_IF(space.TrueVSize() != block_offsets[static_cast<int>(uid + 1)] - block_offsets[static_cast<int>(uid)],
                  "buildElementDofMap: Field true vector size does not match block offsets");

    for (int e = 0; e < num_elements; ++e) {
      mfem::Array<int> element_vdofs;
      space.GetElementVDofs(e, element_vdofs);

      for (int vdof : element_vdofs) {
        SLIC_ERROR_IF(vdof < 0,
                      "buildElementDofMap: Expect non-negative dof index for element-wise discontinuous fields");

        const int combined_dof = block_offsets[static_cast<int>(uid)] + vdof;
        element_dof_map->Set(e, combined_dof, 1.0);
      }
    }
  }

  element_dof_map->Finalize();
  return element_dof_map;
}

NonlinearBlockSolver::NonlinearBlockSolver(std::unique_ptr<EquationSolver> s, MPI_Comm comm, double abs_tol,
                                           double rel_tol,
                                           std::optional<NonlinearSolverOptions> retained_nonlinear_options,
                                           std::optional<LinearSolverOptions> retained_linear_options)
    : nonlinear_solver_(std::move(s)),
      comm_(comm),
      abs_tol_(abs_tol),
      rel_tol_(rel_tol),
      retained_nonlinear_options_(std::move(retained_nonlinear_options)),
      retained_linear_options_(std::move(retained_linear_options))
{
}

std::shared_ptr<NonlinearBlockSolver> NonlinearBlockSolver::cloneFresh() const
{
  if (!retained_nonlinear_options_ || !retained_linear_options_) {
    return nullptr;
  }

  auto nonlinear_opts = *retained_nonlinear_options_;
  const auto linear_opts = *retained_linear_options_;

  auto solver = std::make_unique<EquationSolver>(nonlinear_opts, linear_opts, comm_);
  return std::make_shared<NonlinearBlockSolver>(std::move(solver), comm_, nonlinear_opts.absolute_tol,
                                                nonlinear_opts.relative_tol, nonlinear_opts, linear_opts);
}

void NonlinearBlockSolver::completeSetup(const std::vector<FieldPtr>& us) const
{
  if (us.empty() || !nonlinear_solver_) return;

  if (retained_linear_options_ && retained_linear_options_->linear_solver == LinearSolver::ElementBlock) {
    mfem::Array<int> block_offsets;
    block_offsets.SetSize(static_cast<int>(us.size()) + 1);
    block_offsets[0] = 0;
    for (size_t row_i = 0; row_i < us.size(); ++row_i) {
      block_offsets[static_cast<int>(row_i) + 1] = us[row_i]->space().TrueVSize();
    }
    block_offsets.PartialSum();

    auto* element_block_solver = dynamic_cast<DirectElementBlockSolver*>(&nonlinear_solver_->linearSolver());
    SLIC_ERROR_IF(!element_block_solver, "NonlinearBlockSolver::completeSetup requires DirectElementBlockSolver");
    element_block_solver->setElementDofMap(buildElementDofMap(us, block_offsets));
  }

  if (!retained_linear_options_ || retained_linear_options_->preconditioner == Preconditioner::None) return;

  // Pick the field with the highest vdim (typically the displacement field for vector problems).
  SLIC_ERROR_IF(!us[0], "NonlinearBlockSolver::completeSetup received a null field");
  const FieldT* best = us[0].get();
  for (const auto& u : us) {
    SLIC_ERROR_IF(!u, "NonlinearBlockSolver::completeSetup received a null field");
    if (u->space().GetVDim() > best->space().GetVDim()) best = u.get();
  }

  auto* mfem_solver = &nonlinear_solver_->preconditioner();

  auto configure_amg = [](mfem::Solver* s, int vdim) {
    if (!s) {
      return;
    }
    if (auto* amg = dynamic_cast<mfem::HypreBoomerAMG*>(s)) {
      amg->SetSystemsOptions(vdim, smith::ordering == mfem::Ordering::byNODES);
      return;
    }
    if (auto* wrapped = dynamic_cast<smith::SolverWithPreconditioner*>(s)) {
      if (auto* amg = dynamic_cast<mfem::HypreBoomerAMG*>(wrapped->preconditioner())) {
        amg->SetSystemsOptions(vdim, smith::ordering == mfem::Ordering::byNODES);
      }
    }
  };

  // Top-level AMG preconditioner
  configure_amg(mfem_solver, best->space().GetVDim());

  // Block preconditioners: configure per-block AMG with each block's vdim.
  if (auto* block_preconditioner = dynamic_cast<smith::BlockPreconditioner*>(mfem_solver)) {
    for (int i = 0; i < block_preconditioner->numSubSolvers() && static_cast<size_t>(i) < us.size(); ++i) {
      configure_amg(block_preconditioner->subSolver(i), us[static_cast<size_t>(i)]->space().GetVDim());
    }
  }

#ifdef SMITH_USE_PETSC
  auto* space_dep_pc = dynamic_cast<smith::mfem_ext::PetscPreconditionerSpaceDependent*>(mfem_solver);
  if (space_dep_pc) {
    // This call sets the displacement ParFiniteElementSpace used to get the spatial coordinates and to
    // generate the near null space for the PCGAMG preconditioner
    auto* space = const_cast<mfem::ParFiniteElementSpace*>(&best->space());
    space_dep_pc->SetFESpace(space);
  }
#endif
}

ConvergenceStatus NonlinearBlockSolver::convergenceStatus(double tolerance_multiplier,
                                                          const std::vector<mfem::Vector>& residuals,
                                                          NonlinearConvergenceContext& context) const
{
  auto block_norms = computeResidualBlockNorms(residuals, comm_);
  return evaluateResidualConvergence(tolerance_multiplier, abs_tol_, rel_tol_, block_norms, context);
}

void NonlinearBlockSolver::primeConvergenceContext(const std::vector<mfem::Vector>& residuals,
                                                   NonlinearConvergenceContext& context) const
{
  static_cast<void>(convergenceStatus(1.0, residuals, context));
}

std::vector<NonlinearBlockSolverBase::FieldPtr> NonlinearBlockSolver::solve(
    const std::vector<FieldPtr>& u_guesses,
    std::function<std::vector<mfem::Vector>(const std::vector<FieldPtr>&)> residual_funcs,
    std::function<std::vector<std::vector<MatrixPtr>>(const std::vector<FieldPtr>&)> jacobian_funcs) const
{
  SMITH_MARK_FUNCTION;
  SLIC_ERROR_IF(!nonlinear_solver_, "NonlinearBlockSolver requires an EquationSolver instance before solve()");

  nonlinear_solver_->resetConvergenceState();

  int num_rows = static_cast<int>(u_guesses.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  mfem::Array<int> block_offsets;
  block_offsets.SetSize(num_rows + 1);
  block_offsets[0] = 0;
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_offsets[row_i + 1] = u_guesses[static_cast<size_t>(row_i)]->space().TrueVSize();
  }
  block_offsets.PartialSum();
  auto block_u = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_u->GetBlock(row_i) = *u_guesses[static_cast<size_t>(row_i)];
  }

  auto block_r = std::make_unique<mfem::BlockVector>(block_offsets);

  if (!is_setup_) {
    is_setup_ = true;
    completeSetup(u_guesses);
  }

  auto residual_op_ = std::make_unique<mfem_ext::StdFunctionOperator>(
      block_u->Size(),
      [&residual_funcs, num_rows, &u_guesses, &block_r, &block_offsets](const mfem::Vector& u_, mfem::Vector& r_) {
        const mfem::BlockVector* u = dynamic_cast<const mfem::BlockVector*>(&u_);
        mfem::BlockVector u_block_wrapper;

        if (!u) {
          u_block_wrapper.Update(const_cast<double*>(u_.GetData()), block_offsets);
          u = &u_block_wrapper;
        }
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          *u_guesses[static_cast<size_t>(row_i)] = u->GetBlock(row_i);
        }

        auto residuals = residual_funcs(u_guesses);
        SLIC_ERROR_IF(!block_r, "Invalid residual block cast to an mfem::BlockVector");
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          auto r = residuals[static_cast<size_t>(row_i)];
          block_r->GetBlock(row_i) = r;
        }
        r_ = *block_r;
      },
      [this, &block_offsets, &u_guesses, jacobian_funcs, num_rows](const mfem::Vector& u_) -> mfem::Operator& {
        const mfem::BlockVector* u = dynamic_cast<const mfem::BlockVector*>(&u_);
        mfem::BlockVector u_block_wrapper;
        if (!u) {
          u_block_wrapper.Update(const_cast<double*>(u_.GetData()), block_offsets);
          u = &u_block_wrapper;
        }
        for (int row_i = 0; row_i < num_rows; ++row_i) {
          *u_guesses[static_cast<size_t>(row_i)] = u->GetBlock(row_i);
        }
        matrix_of_jacs_ = jacobian_funcs(u_guesses);
        if (num_rows == 1) {
          auto& J = matrix_of_jacs_[0][0];
          SLIC_ERROR_IF(!J, "Jacobian block (0,0) is null in single-block solve");
          return *J;
        }
        block_jac_ = std::make_unique<mfem::BlockOperator>(block_offsets);
        for (int i = 0; i < num_rows; ++i) {
          for (int j = 0; j < num_rows; ++j) {
            auto& J = matrix_of_jacs_[static_cast<size_t>(i)][static_cast<size_t>(j)];
            if (J) {
              block_jac_->SetBlock(i, j, J.get());
            }
          }
        }
        return *block_jac_;
      });
  nonlinear_solver_->setConvergenceTolerances(inner_tol_multiplier_ * abs_tol_, inner_tol_multiplier_ * rel_tol_,
                                              comm_);
  nonlinear_solver_->setOperator(*residual_op_);
  nonlinear_solver_->solve(*block_u);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_guesses[static_cast<size_t>(row_i)] = block_u->GetBlock(row_i);
  }

  return u_guesses;
}

std::vector<NonlinearBlockSolverBase::FieldPtr> NonlinearBlockSolver::solveAdjoint(
    const std::vector<DualPtr>& u_bars, std::vector<std::vector<MatrixPtr>>& jacobian_transposed) const
{
  SMITH_MARK_FUNCTION;

  int num_rows = static_cast<int>(u_bars.size());
  SLIC_ERROR_IF(num_rows < 0, "Number of residual rows must be non-negative");

  std::vector<NonlinearBlockSolverBase::FieldPtr> u_duals(static_cast<size_t>(num_rows));
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    u_duals[static_cast<size_t>(row_i)] = std::make_shared<NonlinearBlockSolverBase::FieldT>(
        u_bars[static_cast<size_t>(row_i)]->space(), "u_dual_" + std::to_string(row_i));
  }

  auto& linear_solver = nonlinear_solver_->linearSolver();
  if (auto* element_block_solver = dynamic_cast<DirectElementBlockSolver*>(&linear_solver)) {
    element_block_solver->resetFactorizationForAdjointSolve();
  }

  // If it's a 1x1 system, pass the operator directly to avoid potential BlockOperator issues with smoothers
  if (num_rows == 1) {
    linear_solver.SetOperator(*jacobian_transposed[0][0]);
    linear_solver.Mult(*u_bars[0], *u_duals[0]);
    return u_duals;
  }

  mfem::Array<int> block_offsets;
  block_offsets.SetSize(num_rows + 1);
  block_offsets[0] = 0;
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_offsets[row_i + 1] = u_bars[static_cast<size_t>(row_i)]->space().TrueVSize();
  }
  block_offsets.PartialSum();

  auto block_ds = std::make_unique<mfem::BlockVector>(block_offsets);
  *block_ds = 0.0;

  auto block_r = std::make_unique<mfem::BlockVector>(block_offsets);
  for (int row_i = 0; row_i < num_rows; ++row_i) {
    block_r->GetBlock(row_i) = *u_bars[static_cast<size_t>(row_i)];
  }

  auto block_jac = std::make_unique<mfem::BlockOperator>(block_offsets);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_rows; ++j) {
      auto& J = jacobian_transposed[static_cast<size_t>(i)][static_cast<size_t>(j)];
      if (J) {
        block_jac->SetBlock(i, j, J.get());
      }
    }
  }

  linear_solver.SetOperator(*block_jac);
  linear_solver.Mult(*block_r, *block_ds);

  for (int row_i = 0; row_i < num_rows; ++row_i) {
    *u_duals[static_cast<size_t>(row_i)] = block_ds->GetBlock(row_i);
  }

  return u_duals;
}

std::shared_ptr<NonlinearBlockSolver> buildNonlinearBlockSolver(NonlinearSolverOptions nonlinear_opts,
                                                                LinearSolverOptions linear_opts,
                                                                const smith::Mesh& mesh)
{
  auto solid_solver = std::make_unique<EquationSolver>(nonlinear_opts, linear_opts, mesh.getComm());
  return std::make_shared<NonlinearBlockSolver>(std::move(solid_solver), mesh.getComm(), nonlinear_opts.absolute_tol,
                                                nonlinear_opts.relative_tol, nonlinear_opts, linear_opts);
}

}  // namespace smith
