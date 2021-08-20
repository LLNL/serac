// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_dual.hpp
 *
 * @brief This contains a class that represents the space of topological duals sampled
 * on a discrete finite element basis, i.e. the space of residuals and sensitivities.
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

/**
 * @brief Class for encapsulating the topological dual space of a finite element space (i.e. the
 * space of linear forms) as applied to a specific basis set
 *
 * @note While a grid function is provided by this class, interpolation operations are not
 * defined in this representation of the dual space and should be used with caution
 *
 */
class FiniteElementDual : public FiniteElementState {
public:
  /**
   * Main constructor for building a new dual object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, Options&& options = {.order      = 1,
                                                              .vector_dim = 1,
                                                              .coll       = {},
                                                              .ordering   = mfem::Ordering::byVDIM,
                                                              .name       = "",
                                                              .alloc_gf   = true})
      : FiniteElementState(mesh, std::forward<Options>(options)){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "")
      : FiniteElementState(mesh, gf, name){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] space The space to use for the finite element state. This space is deep copied into the new FE state
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& space, const std::string& name = "")
      : FiniteElementState(mesh, space, name){};

  /**
   * @brief Minimal constructor for a FiniteElementDual given an already-existing state
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] fe_state The state for the new state to copy
   * @param[in] name The name of the field
   */
  FiniteElementDual(mfem::ParMesh& mesh, FiniteElementState& fe_state, const std::string& name = "")
      : FiniteElementState(mesh, fe_state, name){};

  /**
   * @brief Set a finite element dual to a constant value
   *
   * @param value The constant to set the finite element dual to
   * @return The modified finite element dual
   * @note This sets the true degrees of freedom and then broadcasts to the shared grid function entries. This means
   * that if a different value is given on different processors, a shared DOF will be set to the owning processor value.
   */
  FiniteElementDual& operator=(const double value)
  {
    auto& true_vec = trueVec();
    true_vec       = value;
    distributeSharedDofs();
    return *this;
  }
};

}  // namespace serac
