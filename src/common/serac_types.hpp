// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include "mfem.hpp"

// Option bundling enums

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
  RK4,
  GeneralizedAlpha,
  ImplicitMidpoint,
  SDIRK23,
  SDIRK34,
  QuasiStatic
};

enum class LinearSolver {
  CG,
  GMRES,
  MINRES
};

enum class Preconditioner {
  Jacobi,
  BoomerAMG
};

// Parameter bundles

struct LinearSolverParameters {
  double rel_tol;
  double abs_tol;
  int print_level;
  int max_iter;
  LinearSolver lin_solver;
  Preconditioner prec;
};

// Finite element information bundle
struct FiniteElementState {
  mfem::ParFiniteElementSpace* space = nullptr;
  mfem::FiniteElementCollection* coll = nullptr;
  mfem::ParGridFunction* gf = nullptr;
  mfem::ParMesh* mesh = nullptr;
  mfem::Vector* true_vec = nullptr;
  std::string name = "";
};

#endif
