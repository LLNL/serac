// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>
#include <optional>
#include <set>
#include <variant>

#include "mfem.hpp"

#include "common/logger.hpp"

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

enum class CouplingScheme
{
  OperatorSplit,
  FixedPoint,
  FullyCoupled
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
  std::shared_ptr<mfem::Vector>                  true_vec;
  std::shared_ptr<mfem::ParMesh>                 mesh;
  std::string                                    name = "";
};

// Boundary condition information
// struct BoundaryCondition {
//   using Coef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;
//   mfem::Array<int>                      markers;
//   mfem::Array<int>                      true_dofs;
//   int                                   component;
//   Coef                                  coef;
//   std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries;
// };

class BoundaryCondition {
  public:
    using Coef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

    BoundaryCondition(Coef coef, const int component, const std::set<int>& attrs, const int num_attrs = 0);

    BoundaryCondition(Coef coef, const int component);

    const mfem::Array<int>& getMarkers() const { return markers_; }

    mfem::Array<int>& getMarkers() { return markers_; }

    void setTrueDofs(const mfem::Array<int> dofs);

    void setTrueDofs(const mfem::ParFiniteElementSpace& fes);

    // FIXME: Assert that this is an essential BC
    const mfem::Array<int>& getTrueDofs() const { return *true_dofs_; }

    // FIXME: Temporary way of maintaining single definition of essential bdr
    // until single class created to encapsulate all BCs
    void removeAttr(const int attr) { markers_[attr-1] = 0;}

    void project(mfem::ParGridFunction& gf, const mfem::ParFiniteElementSpace& space) const;

    void projectBdr(mfem::ParGridFunction& gf, const double time, bool should_be_scalar = true) const;

    template <typename Integrator>
    Integrator* newIntegrator() const;

    // TODO: Check if MFEM should let this be const
    void eliminate(mfem::HypreParMatrix& k_mat);

    mfem::HypreParMatrix& getEliminated() { return *eliminated_matrix_entries_; }

  private:
    Coef                                  coef_;
    int                                   component_;
    mfem::Array<int> markers_;
    std::optional<mfem::Array<int>>       true_dofs_; // Only if essential
    std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries_;
};

template <typename Integrator>
Integrator* BoundaryCondition::newIntegrator() const
{
  // FIXME: So far this is only used for traction integrators...
  // will this always be used with VectorCoef in the general case??
  SLIC_ASSERT_MSG(std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef_),
                    "Boundary condition had a non-vector coefficient when constructing an integrator.");
  // FIXME: Can we do any better with this from a memory management standpoint?
  return new Integrator(*std::get<std::shared_ptr<mfem::VectorCoefficient>>(coef_));
}

}  // namespace serac

#endif
