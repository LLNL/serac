// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef LOADING_FUNCTIONS
#define LOADING_FUNCTIONS

#include "mfem.hpp"

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const mfem::Vector &x, mfem::Vector &y);
void InitialDeformation(const mfem::Vector &x, mfem::Vector &y);

#endif
