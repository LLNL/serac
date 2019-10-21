// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause) 

#ifndef LOADING_FUNCTIONS
#define LOADING_FUNCTIONS

#include "mfem.hpp"

using namespace mfem;

// set kinematic functions and boundary condition functions
void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);

#endif
