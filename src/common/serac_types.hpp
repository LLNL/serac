// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SERAC_TYPES
#define SERAC_TYPES

#include <memory>
#include <optional>
#include <variant>

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
// struct FiniteElementState {
//   std::shared_ptr<mfem::ParFiniteElementSpace>   space;
//   std::shared_ptr<mfem::FiniteElementCollection> coll;
//   std::shared_ptr<mfem::ParGridFunction>         gf;
//   std::shared_ptr<mfem::Vector>                  true_vec;
//   std::shared_ptr<mfem::ParMesh>                 mesh;
//   std::string                                    name = "";
// };

// Git will git confused if the order of the classes are switched, so leave this in until merge
using BCCoef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

class FiniteElementState {
 public:
  FiniteElementState() = default;

  template <typename Collection = mfem::H1_FECollection>
  FiniteElementState(const int order, std::shared_ptr<mfem::ParMesh> pmesh, const std::string& name,
                     const mfem::Ordering::Type ordering = mfem::Ordering::byNODES,
                     std::optional<int>         mesh_dim = std::nullopt);

  MPI_Comm comm() { return space_->GetComm(); }

  mfem::ParGridFunction* gridFunc() { return gf_.get(); }

  mfem::ParMesh* mesh() { return mesh_.get(); }

  std::shared_ptr<mfem::ParFiniteElementSpace> space() { return space_; }

  mfem::Vector* trueVec() { return true_vec_.get(); }

  std::string name() { return name_; }

  void project(const BCCoef& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    std::visit([this](auto&& concrete_coef) { gf_->ProjectCoefficient(*concrete_coef); }, coef);
  }

  void project(mfem::Coefficient& coef) { gf_->ProjectCoefficient(coef); }

  void project(mfem::VectorCoefficient& coef) { gf_->ProjectCoefficient(coef); }

  void initializeTrueVec() { gf_->GetTrueDofs(*true_vec_); }

  void distributeSharedDofs() { gf_->SetFromTrueDofs(*true_vec_); }

  template <typename Tensor>
  std::unique_ptr<Tensor> createTensorOnSpace()
  {
    return std::make_unique<Tensor>(space_.get());
  }

 private:
  std::shared_ptr<mfem::ParMesh>                 mesh_;
  std::shared_ptr<mfem::FiniteElementCollection> coll_;
  std::shared_ptr<mfem::ParFiniteElementSpace>   space_;
  std::shared_ptr<mfem::ParGridFunction>         gf_;
  std::shared_ptr<mfem::Vector>                  true_vec_;
  std::string                                    name_ = "";
};

template <typename Collection = mfem::H1_FECollection>
FiniteElementState::FiniteElementState(const int order, std::shared_ptr<mfem::ParMesh> pmesh, const std::string& name,
                                       const mfem::Ordering::Type ordering, std::optional<int> mesh_dim)
    : mesh_(pmesh),
      coll_(std::make_shared<Collection>(order, pmesh->Dimension())),
      space_(std::make_shared<mfem::ParFiniteElementSpace>(pmesh.get(), coll_.get(),
                                                           (mesh_dim) ? *mesh_dim : pmesh->Dimension(), ordering)),
      gf_(std::make_shared<mfem::ParGridFunction>(space_.get())),
      true_vec_(std::make_shared<mfem::HypreParVector>(space_.get())),
      name_(name)
{
  *gf_       = 0.0;
  *true_vec_ = 0.0;
}

// Boundary condition information
struct BoundaryCondition {
  using Coef = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;
  mfem::Array<int>                      markers;
  mfem::Array<int>                      true_dofs;
  int                                   component;
  Coef                                  coef;
  std::unique_ptr<mfem::HypreParMatrix> eliminated_matrix_entries;
};

}  // namespace serac

#endif
