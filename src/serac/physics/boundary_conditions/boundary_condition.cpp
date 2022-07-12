// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>

#include "serac/physics/boundary_conditions/boundary_condition.hpp"

namespace serac {

BoundaryCondition::BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                                     const mfem::ParFiniteElementSpace& space, const std::set<int>& attrs)
    : coef_(coef), component_(component), markers_(space.GetMesh()->bdr_attributes.Max()), space_(space)
{
  if (get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_)) {
    SLIC_ERROR_ROOT_IF(component_, "A vector coefficient must be applied to all components");
  }

  markers_ = 0;

  for (const int attr : attrs) {
    SLIC_ASSERT_MSG(attr <= markers_.Size(), "Attribute specified larger than what is found in the mesh.");
    markers_[attr - 1] = 1;
  }

  setDofListsFromMarkers();
}

BoundaryCondition::BoundaryCondition(GeneralCoefficient coef, const std::optional<int> component,
                                     const mfem::ParFiniteElementSpace& space, const mfem::Array<int>& true_dofs)
    : coef_(coef), component_(component), markers_(0), space_(space)
{
  if (get_if<std::shared_ptr<mfem::VectorCoefficient>>(&coef_)) {
    SLIC_ERROR_IF(component_, "A vector coefficient must be applied to all components");
  }
  setTrueDofList(true_dofs);
}

void BoundaryCondition::setTrueDofList(const mfem::Array<int>& true_dofs)
{
  true_dofs_ = true_dofs;
  space_.GetRestrictionMatrix()->BooleanMultTranspose(true_dofs_, local_dofs_);
}

void BoundaryCondition::setLocalDofList(const mfem::Array<int>& local_dofs)
{
  local_dofs_ = local_dofs;
  space_.GetRestrictionMatrix()->BooleanMult(local_dofs_, true_dofs_);
}

void BoundaryCondition::setDofListsFromMarkers()
{
  auto& mutable_space = const_cast<mfem::ParFiniteElementSpace&>(space_);

  if (component_) {
    mfem::Array<int> dof_markers;

    mutable_space.GetEssentialTrueDofs(markers_, true_dofs_, *component_);
    space_.GetEssentialVDofs(markers_, dof_markers, *component_);

    // The VDof call actually returns a marker array, so we need to transform it to a list of indices
    space_.MarkerToList(dof_markers, local_dofs_);

  } else {
    mfem::Array<int> dof_markers;

    mutable_space.GetEssentialTrueDofs(markers_, true_dofs_);
    space_.GetEssentialVDofs(markers_, local_dofs_);

    // The VDof call actually returns a marker array, so we need to transform it to a list of indices
    space_.MarkerToList(dof_markers, local_dofs_);
  }
}

void BoundaryCondition::setDofs(mfem::Vector& vector, const double time) const
{
  SLIC_ERROR_IF(space_.GetTrueVSize() != vector.Size(),
                "State to project and boundary condition space are not compatible.");

  FiniteElementState state(*space_.GetParMesh(), space_);

  // Generate the scalar dof list from the vector dof list
  mfem::Array<int> dof_list(local_dofs_.Size());
  std::transform(local_dofs_.begin(), local_dofs_.end(), dof_list.begin(),
                 [&space = space_](int ldof) { return space.VDofToDof(ldof); });

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
      state.projectOnBoundary(*scalar_coef, markers_);
    }
  }

  for (int i : true_dofs_) {
    vector(i) = state(i);
  }
}

void BoundaryCondition::apply(mfem::HypreParMatrix& k_mat, mfem::Vector& rhs, mfem::Vector& state) const
{
  std::unique_ptr<mfem::HypreParMatrix> eliminated_entries(k_mat.EliminateRowsCols(true_dofs_));
  mfem::EliminateBC(k_mat, *eliminated_entries, true_dofs_, state, rhs);
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
