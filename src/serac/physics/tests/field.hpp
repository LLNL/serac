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

#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

struct Field {
  FiniteElementState& get() {
    return *field;
  }

  FiniteElementState& getDual() {
    SLIC_ASSERT_MSG(dual, "Cannot get dual from a field constructed in forward mode only");
    return *dual;
  }

  const mfem::ParFiniteElementSpace& space() const { return field->space(); }

  Field create()

  private:

  FiniteElementState* field;
  FiniteElementDual* dual;
};


/// @brief needs work
struct FieldDual {
  FiniteElementDual& get() {
    return *dual;
  }
  std::shared_ptr<FiniteElementDual> dual;
  std::shared_ptr<FiniteElementState> dualOfDual;
};

}