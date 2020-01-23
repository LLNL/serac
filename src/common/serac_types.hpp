// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

enum class OutputType {
  GLVis,
  VisIt
};

enum class TimestepMethod {
  BackwardEuler,
  SDIRK33,
  ForwardEuler,
  RK2,
  RK3SSP,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34
};

#endif
