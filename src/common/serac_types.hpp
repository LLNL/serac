// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>

#include "mfem.hpp"

namespace serac {

// Option bundling enums

enum class OutputType
{
  GLVis,
  VisIt
};

enum class TimestepMethod
{
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

enum class LinearSolver
{
  CG,
  GMRES,
  MINRES
};

enum class Preconditioner
{
  Jacobi,
  BoomerAMG
};

// Parameter bundles

struct LinearSolverParameters {
  double         rel_tol;
  double         abs_tol;
  int            print_level;
  int            max_iter;
  LinearSolver   lin_solver;
  Preconditioner prec;
};

struct NonlinearSolverParameters {
  double rel_tol;
  double abs_tol;
  int    max_iter;
  int    print_level;
};

// Finite element information bundle
struct FiniteElementState {
  std::shared_ptr<mfem::ParFiniteElementSpace>   space;
  std::shared_ptr<mfem::FiniteElementCollection> coll;
  std::shared_ptr<mfem::ParGridFunction>         gf;
  mfem::Vector                                   true_vec;
  std::shared_ptr<mfem::ParMesh>                 mesh;
  std::string                                    name = "";
};

// Boundary condition information
struct BoundaryCondition {
  mfem::Array<int>                         markers;
  mfem::Array<int>                         true_dofs;
  int                                      component;
  std::shared_ptr<mfem::Coefficient>       scalar_coef;
  std::shared_ptr<mfem::VectorCoefficient> vec_coef;
  std::unique_ptr<mfem::HypreParMatrix>    eliminated_matrix_entries;
};

} // namespace serac

#endif
