// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"

#include <algorithm>
#include <iterator>

#include "serac/infrastructure/logger.hpp"

namespace serac {

void BoundaryConditionManager::addEssential(const std::set<int>& ess_bdr, serac::GeneralCoefficient ess_bdr_coef,
                                            mfem::ParFiniteElementSpace& space, const std::optional<int> component)
{
  std::set<int> filtered_attrs;
  std::set_difference(ess_bdr.begin(), ess_bdr.end(), attrs_in_use_.begin(), attrs_in_use_.end(),
                      std::inserter(filtered_attrs, filtered_attrs.begin()));

  // Check if anything was removed
  if (filtered_attrs.size() < ess_bdr.size()) {
    SLIC_WARNING_ROOT("Multiple definition of essential boundary! Using first definition given.");
  }

  ess_bdr_.emplace_back(ess_bdr_coef, component, space, filtered_attrs);
  attrs_in_use_.insert(ess_bdr.begin(), ess_bdr.end());
  all_dofs_valid_ = false;
}

void BoundaryConditionManager::addEssential(const mfem::Array<int>&                  true_dofs,
                                            std::shared_ptr<mfem::VectorCoefficient> ess_bdr_coef,
                                            mfem::ParFiniteElementSpace&             space)
{
  ess_bdr_.emplace_back(ess_bdr_coef, std::nullopt, space, true_dofs);
  all_dofs_valid_ = false;
}

void BoundaryConditionManager::updateAllDofs() const
{
  all_true_dofs_.DeleteAll();
  all_local_dofs_.DeleteAll();
  for (const auto& bc : ess_bdr_) {
    all_true_dofs_.Append(bc.getTrueDofList());
    all_local_dofs_.Append(bc.getLocalDofList());
  }
  all_true_dofs_.Sort();
  all_local_dofs_.Sort();
  all_true_dofs_.Unique();
  all_local_dofs_.Unique();
  all_dofs_valid_ = true;
}

}  // namespace serac
