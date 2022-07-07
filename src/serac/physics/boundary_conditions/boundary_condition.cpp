// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>

#include "serac/physics/boundary_conditions/boundary_condition.hpp"

namespace serac {

BoundaryCondition::BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                                     const std::set<int>& attrs, const int num_attrs, FiniteElementState* state)
    : coef_(coef), component_(component), markers_(num_attrs), state_(state)
{
  if (get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_)) {
    SLIC_ERROR_ROOT_IF(component_, "A vector coefficient must be applied to all components");
  }

  markers_ = 0;
  for (const int attr : attrs) {
    SLIC_ASSERT_MSG(attr <= num_attrs, "Attribute specified larger than what is found in the mesh.");
    markers_[attr - 1] = 1;
  }

  // If a finite element state is provided, set the dofs from the associated finite element space
  if (state) {
    setDofs();
  }
}

BoundaryCondition::BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                                     const mfem::Array<int>& true_dofs, FiniteElementState* state)
    : coef_(coef), component_(component), markers_(0), state_(state)
{
  if (get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_)) {
    SLIC_ERROR_IF(component_, "A vector coefficient must be applied to all components");
  }
  setTrueDofs(true_dofs);
}

void BoundaryCondition::setTrueDofs(const mfem::Array<int> true_dofs)
{
  SLIC_ERROR_ROOT_IF(!state_, "A finite element state must exist to set the boundary condition DOFs.");
  true_dofs_ = true_dofs;
  state_->space().GetRestrictionMatrix()->BooleanMultTranspose(*true_dofs_, *local_dofs_);
}

void BoundaryCondition::setLocalDofs(const mfem::Array<int> local_dofs)
{
  SLIC_ERROR_ROOT_IF(!state_, "A finite element state must exist to set the boundary condition DOFs.");
  local_dofs_ = local_dofs;
  state_->space().GetRestrictionMatrix()->BooleanMult(*local_dofs_, *true_dofs_);
}

void BoundaryCondition::setDofs()
{
  SLIC_ERROR_ROOT_IF(!state_, "A finite element state must exist to set the boundary condition DOFs.");
  true_dofs_.emplace(0);
  local_dofs_.emplace(0);

  if (component_) {
    mfem::Array<int> dof_markers;

    state_->space().GetEssentialTrueDofs(markers_, *true_dofs_, *component_);
    state_->space().GetEssentialVDofs(markers_, dof_markers, *component_);

    // The VDof call actually returns a marker array, so we need to transform it to a list of indices
    state_->space().MarkerToList(dof_markers, *local_dofs_);

  } else {
    mfem::Array<int> dof_markers;

    state_->space().GetEssentialTrueDofs(markers_, *true_dofs_);
    state_->space().GetEssentialVDofs(markers_, *local_dofs_);

    // The VDof call actually returns a marker array, so we need to transform it to a list of indices
    state_->space().MarkerToList(dof_markers, *local_dofs_);
  }
}

void BoundaryCondition::project(FiniteElementState& state, const double time) const
{
  SLIC_ERROR_ROOT_IF(!true_dofs_, "Only essential boundary conditions can be projected over all DOFs.");
  // Value semantics for convenience
  auto local_dofs = *local_dofs_;
  auto size       = local_dofs.Size();

  // Generate the scalar dof list from the vector dof list
  mfem::Array<int> dof_list(size);
  std::transform(local_dofs.begin(), local_dofs.end(), dof_list.begin(),
                 [&space = std::as_const(state.space())](int ldof) { return space.VDofToDof(ldof); });

  // the only reason to store a VectorCoefficient is to act on all components
  if (is_vector_valued(coef_)) {
    auto vec_coef = get<std::shared_ptr<mfem::VectorCoefficient>>(coef_);
    vec_coef->SetTime(time);
    state.project(*vec_coef, dof_list);
  } else {
    // an mfem::Coefficient could be used to describe a scalar-valued function, or
    // a single component of a vector-valued function
    auto scalar_coef = get<std::shared_ptr<mfem::Coefficient>>(coef_);
    scalar_coef->SetTime(time);
    if (component_) {
      state.project(*scalar_coef, dof_list, *component_);

    } else {
      state.projectBdr(*scalar_coef, markers_);
    }
  }
}

void BoundaryCondition::project(const double time) const
{
  SLIC_ERROR_ROOT_IF(!state_, "Boundary condition must be associated with a FiniteElementState.");
  project(*state_, time);
}

void BoundaryCondition::projectToDofs(mfem::Vector& dof_values, const double time) const
{
  SLIC_ERROR_ROOT_IF(!state_, "Boundary condition must be associated with a FiniteElementState.");
  FiniteElementState state_copy(*state_);
  project(state_copy, time);
  dof_values = state_copy;
}

void BoundaryCondition::eliminateFromMatrix(mfem::HypreParMatrix& k_mat) const
{
  SLIC_ERROR_ROOT_IF(!true_dofs_, "Can only eliminate essential boundary conditions.");
  eliminated_matrix_entries_.reset(k_mat.EliminateRowsCols(*true_dofs_));
}

void BoundaryCondition::eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln,
                                       mfem::Vector& rhs) const
{
  SLIC_ERROR_ROOT_IF(!true_dofs_, "Can only eliminate essential boundary conditions.");
  SLIC_ERROR_ROOT_IF(!eliminated_matrix_entries_,
                     "Must set eliminated matrix entries with eliminateFrom before applying to RHS.");
  mfem::EliminateBC(k_mat_post_elim, *eliminated_matrix_entries_, *true_dofs_, soln, rhs);
}

void BoundaryCondition::apply(mfem::HypreParMatrix& k_mat_post_elim, mfem::Vector& rhs, FiniteElementState& state,
                              const double time) const
{
  project(state, time);
  eliminateToRHS(k_mat_post_elim, state, rhs);
}

const mfem::Coefficient& BoundaryCondition::scalarCoefficient() const
{
  auto scalar_coef = get_if<std::shared_ptr<mfem::Coefficient>>(&coef_);
  SLIC_ERROR_ROOT_IF(!scalar_coef,
                     "Asking for a scalar coefficient on a BoundaryCondition that contains a vector coefficient.");
  return **scalar_coef;
}

mfem::Coefficient& BoundaryCondition::scalarCoefficient()
{
  auto scalar_coef = get_if<std::shared_ptr<mfem::Coefficient>>(&coef_);
  SLIC_ERROR_ROOT_IF(!scalar_coef,
                     "Asking for a scalar coefficient on a BoundaryCondition that contains a vector coefficient.");
  return **scalar_coef;
}

const mfem::VectorCoefficient& BoundaryCondition::vectorCoefficient() const
{
  auto vec_coef = get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_);
  SLIC_ERROR_ROOT_IF(!vec_coef,
                     "Asking for a vector coefficient on a BoundaryCondition that contains a scalar coefficient.");
  return **vec_coef;
}

mfem::VectorCoefficient& BoundaryCondition::vectorCoefficient()
{
  auto vec_coef = get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_);
  SLIC_ERROR_ROOT_IF(!vec_coef,
                     "Asking for a vector coefficient on a BoundaryCondition that contains a scalar coefficient.");
  return **vec_coef;
}

}  // namespace serac
