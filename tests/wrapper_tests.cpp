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

int main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}

class WrapperTests : public ::testing::Test {
 protected:
  void SetUp()
  {
    // Set up mesh
    dim     = 3;
    int nex = 4;
    int ney = 4;
    int nez = 4;

    Mesh mesh(nex, ney, nez, mfem::Element::HEXAHEDRON, true);
    pmesh  = std::shared_ptr<ParMesh>(new ParMesh(MPI_COMM_WORLD, mesh));
    pfes   = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh.get(), new H1_FECollection(1, dim, BasisType::GaussLobatto), 1, Ordering::byNODES));
    pfes_v = std::shared_ptr<ParFiniteElementSpace>(new ParFiniteElementSpace(
        pmesh.get(), new H1_FECollection(1, dim, BasisType::GaussLobatto), dim, Ordering::byNODES));

    pfes_l2 = std::shared_ptr<ParFiniteElementSpace>(
        new ParFiniteElementSpace(pmesh.get(), new L2_FECollection(0, dim), 1, Ordering::byNODES));
  }

  void TearDown() {}

  int                                    dim;
  std::shared_ptr<ParMesh>               pmesh;
  std::shared_ptr<ParFiniteElementSpace> pfes;
  std::shared_ptr<ParFiniteElementSpace> pfes_v;
  std::shared_ptr<ParFiniteElementSpace> pfes_l2;
};

void SolveLinear(std::shared_ptr<ParFiniteElementSpace> pfes, Array<int> &ess_tdof_list, ParGridFunction &temp)
{
  ConstantCoefficient one(1.);

  ParBilinearForm A_lin(pfes.get());
  A_lin.AddDomainIntegrator(new DiffusionIntegrator(one));
  A_lin.Assemble(0);
  A_lin.Finalize(0);
  std::unique_ptr<HypreParMatrix> A = std::unique_ptr<HypreParMatrix>(A_lin.ParallelAssemble());

  ConstantCoefficient coeff_zero(0.0);
  ParLinearForm       f_lin(pfes.get());
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
void SolveNonlinear(std::shared_ptr<ParFiniteElementSpace> pfes, Array<int> &ess_tdof_list, ParGridFunction &temp)
{
  ConstantCoefficient one(1.);

  ParNonlinearForm A_nonlin(pfes.get());
  auto             diffusion = std::make_shared<DiffusionIntegrator>(one);

  A_nonlin.AddDomainIntegrator(new BilinearToNonlinearFormIntegrator(diffusion));
  A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

  ConstantCoefficient coeff_zero(0.0);
  auto                zero_integrator = std::make_shared<DomainLFIntegrator>(coeff_zero);
  A_nonlin.AddDomainIntegrator(new LinearToNonlinearFormIntegrator(zero_integrator, pfes));

  // The temperature solution vector already contains the essential boundary condition values
  std::unique_ptr<HypreParVector> T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

  GMRESSolver solver(pfes->GetComm());

  NewtonSolver newton_solver(pfes->GetComm());
  newton_solver.SetSolver(solver);
  newton_solver.SetOperator(A_nonlin);

  Vector zero;
  newton_solver.Mult(zero, *T);

  temp = *T;
}

// Solve the same linear system using a newton solver but by using the MixedIntegrator calls
void SolveMixedNonlinear(std::shared_ptr<ParFiniteElementSpace> pfes, Array<int> &ess_tdof_list, ParGridFunction &temp)
{
  ConstantCoefficient one(1.);

  ParNonlinearForm A_nonlin(pfes.get());
  auto             diffusion = std::make_shared<DiffusionIntegrator>(one);

  A_nonlin.AddDomainIntegrator(new MixedBilinearToNonlinearFormIntegrator(diffusion, pfes));
  A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

  ParLinearForm f_lin(pfes.get());
  f_lin                             = 0.;
  std::unique_ptr<HypreParVector> F = std::unique_ptr<HypreParVector>(f_lin.ParallelAssemble());

  // The temperature solution vector already contains the essential boundary condition values
  std::unique_ptr<HypreParVector> T = std::unique_ptr<HypreParVector>(temp.GetTrueDofs());

  GMRESSolver solver(pfes->GetComm());

  NewtonSolver newton_solver(pfes->GetComm());
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
  StdFunctionCoefficient x_zero([](Vector &x) {
    if (x[0] < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  // Create a coefficient that indicates the x == 1 border of the cube
  StdFunctionCoefficient x_one([](Vector &x) {
    if ((1. - x[0]) < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  Array<int> bdr_attr_list_zero;
  MakeBdrAttributeList(*pmesh, bdr_attr_list_zero, x_zero);

  Array<int> bdr_attr_list_one;
  MakeBdrAttributeList(*pmesh, bdr_attr_list_one, x_one);

  // Set x_zero to be attribute 2 and x_one to be attribute 3
  Array<int> bdr_attr_list(pfes->GetNBE());
  for (int be = 0; be < pfes->GetNBE(); be++) {
    bdr_attr_list[be] = (bdr_attr_list_zero[be] - 1) + (bdr_attr_list_one[be] - 1) * 2 + 1;
  }

  for (int be = 0; be < pfes->GetNBE(); be++) {
    pmesh->GetBdrElement(be)->SetAttribute(bdr_attr_list[be]);
  }

  // Update attribute data structures
  pmesh->SetAttributes();

  Array<int> bdr_attr_is_ess(3);
  bdr_attr_is_ess[0] = 0;
  bdr_attr_is_ess[1] = 1;  //< This is an attribute we are looking for
  bdr_attr_is_ess[2] = 1;  //< This is an attribute we are looking for

  // Get all the essential degrees of freedom
  Array<int> ess_tdof_list;
  pfes->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

  // Boundary conditions evaluation
  ParGridFunction t_ess(pfes.get());
  t_ess = 0.;
  t_ess.ProjectBdrCoefficient(x_one, bdr_attr_is_ess);

  // Solve a simple static thermal problem with standard linear integrators
  ParGridFunction t_lin(pfes.get());
  t_lin = t_ess;
  SolveLinear(pfes, ess_tdof_list, t_lin);

  // Solve the same problem using a wrapped nonlinear integrators
  ParGridFunction t_nonlin(pfes.get());
  t_nonlin = t_ess;
  SolveNonlinear(pfes, ess_tdof_list, t_nonlin);

  // Compare the linear and the nonlinear solution
  for (int i = 0; i < t_lin.Size(); i++) {
    EXPECT_NEAR(t_lin[i], t_nonlin[i], 1.e-12);
  }

  // Solve the same nonlinear problem with the MixedBilinearToNonlinearformIntegrator
  ParGridFunction t_mixed_nonlin(pfes.get());
  t_mixed_nonlin = t_ess;
  SolveMixedNonlinear(pfes, ess_tdof_list, t_mixed_nonlin);

  // Compare the mixed nonlinear linear and the nonlinear solution
  for (int i = 0; i < t_nonlin.Size(); i++) {
    EXPECT_NEAR(t_mixed_nonlin[i], t_nonlin[i], 1.e-12);
  }
}
