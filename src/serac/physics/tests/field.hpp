// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file field.hpp
 *
 * @brief Containers for fields and their duals
 */

#pragma once

#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/physics/common.hpp"

namespace serac {

struct Field {

  Field() = default;
  Field(const Field& f) = default;
  Field& operator=(const Field& f) = default;

  template <typename function_space>
  static Field create(function_space space, const std::string& name, const std::string& mesh_tag, bool make_dual=true) {
    Field f;
    f.field = std::make_shared<FiniteElementState>(StateManager::newState(space, name, mesh_tag));
    if (make_dual) {
      f.dual = std::make_shared<FiniteElementDual>(StateManager::newDual(space, detail::addPrefix(name,"dual"), mesh_tag));
    }
    return f;
  }

  FiniteElementState& get() const {
    return *field;
  }

  FiniteElementDual& getDual() const {
    SLIC_ASSERT_MSG(dual, "Cannot get dual from a field constructed in forward mode only");
    return *dual;
  }

  const mfem::ParFiniteElementSpace& space() const { return field->space(); }

  std::shared_ptr<FiniteElementState> field;
  std::shared_ptr<FiniteElementDual> dual;
};


/// @brief needs work
struct Resultant {

  Resultant() = default;
  Resultant(const Resultant& f) = default;
  Resultant& operator=(const Resultant& f) = default;

  template <typename function_space>
  static Resultant create(function_space space, const std::string& name, const std::string& mesh_tag, bool make_dual=true) {
    Resultant f;
    f.resultant = std::make_shared<FiniteElementDual>(StateManager::newDual(space, name, mesh_tag));
    if (make_dual) {
      f.resultantDual = std::make_shared<FiniteElementState>(StateManager::newState(space, detail::addPrefix(name,"dual"), mesh_tag));
    }
    return f;
  }

  FiniteElementDual& get() {
    return *resultant;
  }

  FiniteElementState& getDual() const {
    SLIC_ASSERT_MSG(dual, "Cannot get dual from a field constructed in forward mode only");
    return *resultantDual;
  }

  const mfem::ParFiniteElementSpace& space() const { return resultant->space(); }

  std::shared_ptr<FiniteElementDual> resultant;
  std::shared_ptr<FiniteElementState> resultantDual;
};

}