// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/utilities/boundary_condition.hpp"

#include <algorithm>

namespace serac {

const mfem::Coefficient& GeneralCoefficientWrapper::scalarCoefficient() const
{
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef_),
                "Asking for a scalar coefficient on a GeneralCoefficientWrapper that contains a vector coefficient.");
  return *std::get<std::shared_ptr<mfem::Coefficient>>(coef_);
}

mfem::Coefficient& GeneralCoefficientWrapper::scalarCoefficient()
{
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef_),
                "Asking for a scalar coefficient on a GeneralCoefficientWrapper that contains a vector coefficient.");
  return *std::get<std::shared_ptr<mfem::Coefficient>>(coef_);
}

const mfem::VectorCoefficient& GeneralCoefficientWrapper::vectorCoefficient() const
{
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                "Asking for a vector coefficient on a GeneralCoefficientWrapper that contains a scalar coefficient.");
  return *std::get<std::shared_ptr<mfem::VectorCoefficient>>(coef_);
}

mfem::VectorCoefficient& GeneralCoefficientWrapper::vectorCoefficient()
{
  SLIC_ERROR_IF(!std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                "Asking for a vector coefficient on a GeneralCoefficientWrapper that contains a scalar coefficient.");
  return *std::get<std::shared_ptr<mfem::VectorCoefficient>>(coef_);
}

mfem::Array<int> BoundaryCondition::makeMarkers(const std::set<int>& attrs, const int num_attrs)
{
  mfem::Array<int> markers(num_attrs);
  markers = 0;
  for (const int attr : attrs) {
    SLIC_ASSERT_MSG(attr <= num_attrs, "Attribute specified larger than what is found in the mesh.");
    markers[attr - 1] = 1;
  }

  return markers;
}

NaturalBoundaryCondition::NaturalBoundaryCondition(GeneralCoefficient coef, const int component,
                                                   mfem::Array<int>&& markers)
    : coef_(coef), component_(component), markers_(std::move(markers))
{
}

EssentialBoundaryCondition::EssentialBoundaryCondition(GeneralCoefficient coef, const int component,
                                                       mfem::Array<int>&& markers)
    : coef_(coef), component_(component), markers_(std::move(markers)), true_dofs_(0)
{
}

EssentialBoundaryCondition::EssentialBoundaryCondition(GeneralCoefficient coef, const int component)
    : coef_(coef), component_(component), markers_(0), true_dofs_(0)
{
}

void EssentialBoundaryCondition::setTrueDofs(const mfem::Array<int>& dofs)
{
  true_dofs_              = dofs;
  dofs_fully_initialized_ = true;
}

void EssentialBoundaryCondition::setTrueDofs(FiniteElementState& state)
{
  if (markers_.Size() > 0) {
    state.space().GetEssentialTrueDofs(markers_, true_dofs_, component_);
    dofs_fully_initialized_ = true;
  }
}

void EssentialBoundaryCondition::project(FiniteElementState& state) const
{
  SLIC_ERROR_IF(!dofs_fully_initialized_, "DOFS have not been initialized.");
  // Value semantics for convenience
  auto tdofs = true_dofs_;
  auto size  = tdofs.Size();
  if (size) {
    // Generate the scalar dof list from the vector dof list
    mfem::Array<int> dof_list(size);
    std::transform(tdofs.begin(), tdofs.end(), dof_list.begin(),
                   [&space = std::as_const(state.space())](int tdof) { return space.VDofToDof(tdof); });

    if (component_ == -1) {
      // If it contains all components, project the vector
      state.gridFunc().ProjectCoefficient(coef_.vectorCoefficient(), dof_list);
    } else {
      // If it is only a single component, project the scalar
      state.gridFunc().ProjectCoefficient(coef_.scalarCoefficient(), dof_list, component_);
    }
  }
}

void EssentialBoundaryCondition::projectBdr(mfem::ParGridFunction& gf, const double time,
                                            const bool should_be_scalar) const
{
  // markers_ should be const param but it's not
  auto& non_const_markers = const_cast<mfem::Array<int>&>(markers_);

  if (should_be_scalar) {
    auto& coef = coef_.scalarCoefficient();
    coef.SetTime(time);
    gf.ProjectBdrCoefficient(coef, non_const_markers);
  } else {
    auto& coef = coef_.vectorCoefficient();
    coef.SetTime(time);
    gf.ProjectBdrCoefficient(coef, non_const_markers);
  }
}

void EssentialBoundaryCondition::projectBdr(FiniteElementState& state, const double time,
                                            const bool should_be_scalar) const
{
  projectBdr(state.gridFunc(), time, should_be_scalar);
}

void EssentialBoundaryCondition::eliminateFromMatrix(mfem::HypreParMatrix& k_mat) const
{
  SLIC_ERROR_IF(!dofs_fully_initialized_, "DOFS have not been initialized.");
  eliminated_matrix_entries_.reset(k_mat.EliminateRowsCols(true_dofs_));
}

void EssentialBoundaryCondition::eliminateToRHS(mfem::HypreParMatrix& k_mat_post_elim, const mfem::Vector& soln,
                                                mfem::Vector& rhs) const
{
  SLIC_ERROR_IF(!dofs_fully_initialized_, "DOFS have not been initialized.");
  SLIC_ERROR_IF(!eliminated_matrix_entries_,
                "Must set eliminated matrix entries with eliminateFrom before applying to RHS.");
  mfem::EliminateBC(k_mat_post_elim, *eliminated_matrix_entries_, true_dofs_, soln, rhs);
}

void EssentialBoundaryCondition::apply(mfem::HypreParMatrix& k_mat_post_elim, mfem::Vector& rhs,
                                       FiniteElementState& state, const double time, const bool should_be_scalar) const
{
  projectBdr(state, time, should_be_scalar);
  state.initializeTrueVec();
  eliminateToRHS(k_mat_post_elim, state.trueVec(), rhs);
}

}  // namespace serac
