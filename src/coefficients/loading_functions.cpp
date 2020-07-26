// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "loading_functions.hpp"

void referenceConfiguration(const mfem::Vector& x, mfem::Vector& y)
{
  // set the reference, stress free, configuration
  y = x;
}

void initialDeformation(const mfem::Vector&, mfem::Vector& y) { y = 0.0; }
