// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "common/boundary_condition.hpp"

#include <algorithm>

#include "common/logger.hpp"

namespace serac {

BoundaryCondition::BoundaryCondition(Coef coef, const int component, const std::set<int>& attrs, const int num_attrs)
    : coef_(coef), component_(component), markers_(num_attrs)
{
  markers_ = 0;
  for (const int attr : attrs) {
    SLIC_ASSERT_MSG(attr <= num_attrs, "Attribute specified larger than what is found in the mesh.");
    markers_[attr - 1] = 1;
  }
}

BoundaryCondition::BoundaryCondition(Coef coef, const int component, const mfem::Array<int>& true_dofs)
    : coef_(coef), component_(component), markers_(0), true_dofs_(true_dofs)
{
}

void BoundaryCondition::setTrueDofs(const mfem::Array<int> dofs) { true_dofs_ = dofs; }

void BoundaryCondition::setTrueDofs(FiniteElementState& state)
{
  true_dofs_.emplace(0);
  state_ = &state;
  state.space().GetEssentialTrueDofs(markers_, *true_dofs_, component_);
}

void BoundaryCondition::project(FiniteElementState& state) const
{
  SLIC_ERROR_IF(!true_dofs_, "Only essential boundary conditions can be projected over all DOFs.");
  // Value semantics for convenience
  auto tdofs = *true_dofs_;
  auto size  = tdofs.Size();
  if (size) {
    // Generate the scalar dof list from the vector dof list
    SLIC_ASSERT_MSG(space, "Only BCs associated with a space can be projected.");
    mfem::Array<int> dof_list(size);
    std::transform(&tdofs[0], &tdofs[0] + size, &dof_list[0],
                   [&space = std::as_const(state.space())](int tdof) { return space.VDofToDof(tdof); });

    if (component_ == -1) {
      // If it contains all components, project the vector
      SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                      "Essential boundary condition contained all components but had a non-vector coefficient.");
      state.gridFunc().ProjectCoefficient(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(coef_), dof_list);
    } else {
      // If it is only a single component, project the scalar
      SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef_),
                      "Essential boundary condition contained a single component but had a non-scalar coefficient.");
      state.gridFunc().ProjectCoefficient(*std::get<std::shared_ptr<mfem::Coefficient>>(coef_), dof_list, component_);
    }
  }
}

void BoundaryCondition::project() const
{
  SLIC_ERROR_IF(!state_, "Boundary condition must be associated with a FiniteElementState.");
  project(*state_);
}

void BoundaryCondition::projectBdr(mfem::ParGridFunction& gf, const double time, const bool should_be_scalar) const
{
  if (should_be_scalar) {
    SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef_),
                    "Boundary condition should have been an mfem::Coefficient");
  } else {
    SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                    "Boundary condition should have been an mfem::VectorCoefficient");
  }

  // markers_ should be const param but it's not
  std::visit(
      [&gf, &markers = const_cast<mfem::Array<int>&>(markers_), time](auto&& coef) {
        coef->SetTime(time);
        gf.ProjectBdrCoefficient(*coef, markers);
      },
      coef_);
}

void BoundaryCondition::projectBdr(FiniteElementState& state, const double time, const bool should_be_scalar) const
{
  projectBdr(state.gridFunc(), time, should_be_scalar);
}

void BoundaryCondition::projectBdr(const double time, const bool should_be_scalar) const
{
  SLIC_ERROR_IF(!state_, "Boundary condition must be associated with a FiniteElementState.");
  projectBdr(*state_, time, should_be_scalar);
}

void BoundaryCondition::eliminateFromMatrix(mfem::HypreParMatrix& k_mat)
{
  SLIC_ERROR_IF(!true_dofs_, "Can only eliminate essential boundary conditions.");
  eliminated_matrix_entries_.reset(k_mat.EliminateRowsCols(*true_dofs_));
}

void BoundaryCondition::eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln,
                                       mfem::Vector& rhs)
{
  SLIC_ERROR_IF(!true_dofs_, "Can only eliminate essential boundary conditions.");
  SLIC_ERROR_IF(!eliminated_matrix_entries_,
                "Must set eliminated matrix entries with eliminateFrom before applying to RHS.");
  mfem::EliminateBC(k_mat_post_elim, *eliminated_matrix_entries_, *true_dofs_, soln, rhs);
}

void BoundaryCondition::apply(mfem::HypreParMatrix& k_mat_post_elim, mfem::Vector& rhs, FiniteElementState& state,
                              const double time, const bool should_be_scalar)
{
  projectBdr(state, time, should_be_scalar);
  state.initializeTrueVec();
  eliminateToRHS(k_mat_post_elim, state.trueVec(), rhs);
}

}  // namespace serac
