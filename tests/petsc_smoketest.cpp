// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <iostream>

#include "petsc.h"

TEST(petsc_test, petsc_smoketest)
{
  char buf[100];
  PetscGetVersion(buf, 100);
  std::cout << buf << std::endl;
}
