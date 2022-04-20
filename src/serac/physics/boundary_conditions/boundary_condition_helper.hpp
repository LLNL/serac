// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file boundary_condition_helper.hpp
 *
 * @brief This file contains the declation of helper methods for boundary conditions
 */

#pragma once

#include "mfem.hpp"

namespace serac::mfem_ext {

void GetEssentialTrueDofsFromElementAttribute(
    const mfem::ParFiniteElementSpace &fespace,
    const mfem::Array<int> &elem_attr_is_ess, 
    mfem::Array<int> &ess_tdof_list, 
    int component);

void GetEssentialVDofsFromElementAttribute(
    const mfem:ParFiniteElementSpace &fespace,
    const mfem::Array<int> &elem_attr_is_ess,
    mfem::Array<int> &ess_vdofs,
    int component);

}  // namespace serac::mfem_ext