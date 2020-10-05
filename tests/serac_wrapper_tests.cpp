// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include <gtest/gtest.h>

#include <memory>

#include "coefficients/stdfunction_coefficient.hpp"
#include "integrators/wrapper_integrator.hpp"
#include "mfem.hpp"

using namespace mfem;
using namespace serac;

class WrapperTests : public ::testing::Test {
protected:
  void SetUp()
  {
    // Set up mesh
    dim_    = 3;
    int nex = 4;
    int ney = 4;
    int nez = 4;

    Mesh mesh(nex, ney, nez, mfem::Element::HEXAHEDRON, true);
    pmesh_  = std::shared_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, mesh));
    pfes_   = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh_.get(), new H1_FECollection(1, dim_, BasisType::GaussLobatto), 1, Ordering::byNODES));
    pfes_v_ = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh_.get(), new H1_FECollection(1, dim_, BasisType::GaussLobatto), dim_, Ordering::byNODES));

    pfes_l2_ = std::shared_ptr<ParFiniteElementSpace>(
        new ParFiniteElementSpace(pmesh_.get(), new L2_FECollection(0, dim_), 1, Ordering::byNODES));
  }

  void TearDown() {}

  int                                    dim_;
  std::shared_ptr<ParMesh>               pmesh_;
  std::shared_ptr<ParFiniteElementSpace> pfes_;
  std::shared_ptr<ParFiniteElementSpace> pfes_v_;
  std::shared_ptr<ParFiniteElementSpace> pfes_l2_;
};

void SolveLinear(std::shared_ptr<ParFiniteElementSpace> pfes_, Array<int>& ess_tdof_list, ParGridFunction& temp)
{
  ConstantCoefficient one(1.);

  ParBilinearForm A_lin(pfes_.get());
  A_lin.AddDomainIntegrator(new DiffusionIntegrator(one));
  A_lin.Assemble(0);
  A_lin.Finalize(0);
  std::unique_ptr<HypreParMatrix> A = std::unique_ptr<HypreParMatrix>(A_lin.ParallelAssemble());

  ConstantCoefficient coeff_zero(0.0);
  ParLinearForm       f_lin(pfes_.get());
  f_lin.AddDomainIntegrator(new DomainLFIntegrator(coeff_zero));
  f_lin.Assemble();
  std::unique_ptr<HypreParVector> F = std::unique_ptr<HypreParVector>(f_lin.ParallelAssemble());

  std::unique_ptr<HypreParVector> T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

  A->EliminateRowsCols(ess_tdof_list, *T, *F);

  HyprePCG       solver(*A);
  HypreBoomerAMG hypre_amg(*A);
  hypre_amg.SetPrintLevel(0);
  solver.SetTol(1.e-14);
  solver.SetPreconditioner(hypre_amg);
  solver.Mult(*F, *T);

  temp = *T;
}

// Solve the same linear system using a newton solver
void SolveNonlinear(std::shared_ptr<ParFiniteElementSpace> pfes_, Array<int>& ess_tdof_list, ParGridFunction& temp)
{
  ConstantCoefficient one(1.);

  ParNonlinearForm A_nonlin(pfes_.get());

  auto diffusion = std::make_shared<DiffusionIntegrator>(one);

  A_nonlin.AddDomainIntegrator(new BilinearToNonlinearFormIntegrator(diffusion));
  A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

  ConstantCoefficient coeff_zero(0.0);

  auto zero_integrator = std::make_shared<DomainLFIntegrator>(coeff_zero);
  A_nonlin.AddDomainIntegrator(new LinearToNonlinearFormIntegrator(zero_integrator, pfes_));

  // The temperature solution vector already contains the essential boundary condition values
  std::unique_ptr<HypreParVector> T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

  GMRESSolver solver(pfes_->GetComm());

  NewtonSolver newton_solver(pfes_->GetComm());
  newton_solver.SetSolver(solver);
  newton_solver.SetOperator(A_nonlin);

  Vector zero;
  newton_solver.Mult(zero, *T);

  temp = *T;
}

// Solve the same linear system using a newton solver but by using the MixedIntegrator calls
void SolveMixedNonlinear(std::shared_ptr<ParFiniteElementSpace> pfes_, Array<int>& ess_tdof_list, ParGridFunction& temp)
{
  ConstantCoefficient one(1.);

  ParNonlinearForm A_nonlin(pfes_.get());

  auto diffusion = std::make_shared<DiffusionIntegrator>(one);

  A_nonlin.AddDomainIntegrator(new MixedBilinearToNonlinearFormIntegrator(diffusion, pfes_));
  A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

  // The temperature solution vector already contains the essential boundary condition values
  std::unique_ptr<HypreParVector> T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

  GMRESSolver solver(pfes_->GetComm());

  NewtonSolver newton_solver(pfes_->GetComm());
  newton_solver.SetSolver(solver);
  newton_solver.SetOperator(A_nonlin);

  Vector zero;
  newton_solver.Mult(zero, *T);

  temp = *T;
}

/// Solve a simple laplacian problem on a cube mesh
TEST_F(WrapperTests, nonlinear_linear_thermal)
{
  // Create a coefficient that indicates the x == 0 border of the cube
  StdFunctionCoefficient x_zero([](const Vector& x) {
    if (x[0] < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  // Create a coefficient that indicates the x == 1 border of the cube
  StdFunctionCoefficient x_one([](const Vector& x) {
    if ((1. - x[0]) < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  Array<int> bdr_attr_list_zero = serac::makeBdrAttributeList(*pmesh_, x_zero);
  Array<int> bdr_attr_list_one  = serac::makeBdrAttributeList(*pmesh_, x_one);

  // Set x_zero to be attribute 2 and x_one to be attribute 3
  Array<int> bdr_attr_list(pfes_->GetNBE());
  for (int be = 0; be < pfes_->GetNBE(); be++) {
    bdr_attr_list[be] = (bdr_attr_list_zero[be] - 1) + (bdr_attr_list_one[be] - 1) * 2 + 1;
  }

  for (int be = 0; be < pfes_->GetNBE(); be++) {
    pmesh_->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }

  // Update attribute data structures
  pmesh_->SetAttributes();

  Array<int> bdr_attr_is_ess(3);
  bdr_attr_is_ess[0] = 0;
  bdr_attr_is_ess[1] = 1;  //< This is an attribute we are looking for
  bdr_attr_is_ess[2] = 1;  //< This is an attribute we are looking for

  // Get all the essential degrees of freedom
  Array<int> ess_tdof_list;
  pfes_->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Boundary conditions evaluation
  ParGridFunction t_ess(pfes_.get());
  t_ess = 0.;
  t_ess.ProjectBdrCoefficient(x_one, bdr_attr_is_ess);

  // Solve a simple static thermal problem with standard linear integrators
  ParGridFunction t_lin(pfes_.get());
  t_lin = t_ess;
  SolveLinear(pfes_, ess_tdof_list, t_lin);

  // Solve the same problem using a wrapped nonlinear integrators
  ParGridFunction t_nonlin(pfes_.get());
  t_nonlin = t_ess;
  SolveNonlinear(pfes_, ess_tdof_list, t_nonlin);

  // Compare the linear and the nonlinear solution
  for (int i = 0; i < t_lin.Size(); i++) {
    EXPECT_NEAR(t_lin[i], t_nonlin[i], 1.e-12);
  }

  // Solve the same nonlinear problem with the MixedBilinearToNonlinearformIntegrator
  ParGridFunction t_mixed_nonlin(pfes_.get());
  t_mixed_nonlin = t_ess;
  SolveMixedNonlinear(pfes_, ess_tdof_list, t_mixed_nonlin);

  // Compare the mixed nonlinear linear and the nonlinear solution
  for (int i = 0; i < t_nonlin.Size(); i++) {
    EXPECT_NEAR(t_mixed_nonlin[i], t_nonlin[i], 1.e-12);
  }
}

TEST_F(WrapperTests, Transformed)
{
  // Setup problem

  // Create a coefficient that indicates the x == 0 border of the cube
  StdFunctionCoefficient x_zero([](const Vector& x) {
    if (x[0] < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  // Create a coefficient that indicates the x == 1 border of the cube
  StdFunctionCoefficient x_one([](const Vector& x) {
    if ((1. - x[0]) < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  Array<int> bdr_attr_list_zero = serac::makeBdrAttributeList(*pmesh_, x_zero);
  Array<int> bdr_attr_list_one  = serac::makeBdrAttributeList(*pmesh_, x_one);

  // Set x_zero to be attribute 2 and x_one to be attribute 3
  Array<int> bdr_attr_list(pfes_->GetNBE());
  for (int be = 0; be < pfes_->GetNBE(); be++) {
    bdr_attr_list[be] = (bdr_attr_list_zero[be] - 1) + (bdr_attr_list_one[be] - 1) * 2 + 1;
  }

  for (int be = 0; be < pfes_->GetNBE(); be++) {
    pmesh_->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }

  // Update attribute data structures
  pmesh_->SetAttributes();

  Array<int> bdr_attr_is_ess(3);
  bdr_attr_is_ess[0] = 0;
  bdr_attr_is_ess[1] = 1;  //< This is an attribute we are looking for
  bdr_attr_is_ess[2] = 1;  //< This is an attribute we are looking for

  // Get all the essential degrees of freedom
  Array<int> ess_tdof_list;
  pfes_->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Boundary conditions evaluation
  ParGridFunction t_ess(pfes_.get());
  t_ess = 0.;
  t_ess.ProjectBdrCoefficient(x_one, bdr_attr_is_ess);

  // Solve Nonlinear

  ConstantCoefficient one(1.);

  auto   diffusion           = std::make_shared<DiffusionIntegrator>(one);
  auto   nonlinear_diffusion = std::make_unique<BilinearToNonlinearFormIntegrator>(diffusion);
  double multiplier          = 2.;
  double offset              = 1.;
  auto   transform           = [=](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::Vector& x) {
    auto v = std::make_shared<mfem::Vector>(x);
    (*v) *= multiplier;
    (*v) += offset;
    return v;
  };
  auto transform_grad = [=](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::DenseMatrix& x) {
    auto m = std::make_shared<mfem::DenseMatrix>(x);
    (*m) *= multiplier;
    return m;
  };
  auto transformed_diffusion = std::make_unique<TransformedNonlinearFormIntegrator>(
      std::make_unique<BilinearToNonlinearFormIntegrator>(diffusion), transform, transform_grad);

  ParGridFunction temp(pfes_.get());
  {
    ParNonlinearForm A_nonlin(pfes_.get());
    A_nonlin.AddDomainIntegrator(nonlinear_diffusion.release());
    A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

    // The temperature solution vector already contains the essential boundary condition values
    temp = t_ess;

    auto T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

    GMRESSolver solver(pfes_->GetComm());

    NewtonSolver newton_solver(pfes_->GetComm());
    newton_solver.SetSolver(solver);
    newton_solver.SetOperator(A_nonlin);

    Vector zero;
    newton_solver.Mult(zero, *T);

    temp = *T;
  }

  ParGridFunction temp2(pfes_.get());
  {
    ParNonlinearForm A_nonlin(pfes_.get());
    A_nonlin.AddDomainIntegrator(transformed_diffusion.release());
    A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

    // The temperature solution vector already contains the essential boundary condition values
    temp2 = t_ess;
    temp2 -= offset;
    temp2 *= 1. / multiplier;

    auto T = std::unique_ptr<HypreParVector>(temp2.GetTrueDofs());

    GMRESSolver solver(pfes_->GetComm());

    NewtonSolver newton_solver(pfes_->GetComm());
    newton_solver.SetSolver(solver);
    newton_solver.SetOperator(A_nonlin);

    Vector zero;
    newton_solver.Mult(zero, *T);

    temp2 = *T;
  }

  for (int i = 0; i < temp.Size(); i++) {
    EXPECT_NEAR(temp[i], temp2[i] * multiplier + offset, 1.e-8);
  }
  temp.Print();
  std::cout << multiplier << std::endl;
  temp2.Print();
}

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
