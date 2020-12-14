// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include <gtest/gtest.h>

#include <memory>

#include "mfem.hpp"
#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/integrators/wrapper_integrator.hpp"

using namespace mfem;
using namespace serac;

class WrapperTests : public ::testing::Test {
protected:
  void SetUp() override
  {
    // Set up mesh
    dim_    = 3;
    int nex = 4;
    int ney = 4;
    int nez = 4;

    Mesh mesh(nex, ney, nez, mfem::Element::HEXAHEDRON, true);
    pmesh_ = std::make_shared<ParMesh>(MPI_COMM_WORLD, mesh);
    fec_   = std::make_unique<H1_FECollection>(1, dim_, BasisType::GaussLobatto);
    pfes_  = std::make_shared<ParFiniteElementSpace>(pmesh_.get(), fec_.get(), 1, Ordering::byNODES);

    fec_v_  = std::make_unique<H1_FECollection>(1, dim_, BasisType::GaussLobatto);
    pfes_v_ = std::make_shared<ParFiniteElementSpace>(pmesh_.get(), fec_v_.get(), dim_, Ordering::byNODES);

    fec_l2_  = std::make_unique<L2_FECollection>(0, dim_);
    pfes_l2_ = std::make_shared<ParFiniteElementSpace>(pmesh_.get(), fec_l2_.get(), 1, Ordering::byNODES);
  }

  void TearDown() override {}

  int                                    dim_;
  std::shared_ptr<ParMesh>               pmesh_;
  std::shared_ptr<ParFiniteElementSpace> pfes_;
  std::shared_ptr<ParFiniteElementSpace> pfes_v_;
  std::shared_ptr<ParFiniteElementSpace> pfes_l2_;

private:
  std::unique_ptr<FiniteElementCollection> fec_;
  std::unique_ptr<FiniteElementCollection> fec_v_;
  std::unique_ptr<FiniteElementCollection> fec_l2_;
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

  A_nonlin.AddDomainIntegrator(new mfem_ext::BilinearToNonlinearFormIntegrator(diffusion));
  A_nonlin.SetEssentialTrueDofs(ess_tdof_list);

  ConstantCoefficient coeff_zero(0.0);

  auto zero_integrator = std::make_shared<DomainLFIntegrator>(coeff_zero);
  A_nonlin.AddDomainIntegrator(new mfem_ext::LinearToNonlinearFormIntegrator(zero_integrator, pfes_));

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

  A_nonlin.AddDomainIntegrator(new mfem_ext::MixedBilinearToNonlinearFormIntegrator(diffusion, pfes_));
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
  mfem::FunctionCoefficient x_zero([](const Vector& x) {
    if (x[0] < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  // Create a coefficient that indicates the x == 1 border of the cube
  mfem::FunctionCoefficient x_one([](const Vector& x) {
    if ((1. - x[0]) < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  Array<int> bdr_attr_list_zero = mfem_ext::makeBdrAttributeList(*pmesh_, x_zero);
  Array<int> bdr_attr_list_one  = mfem_ext::makeBdrAttributeList(*pmesh_, x_one);

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
  mfem::FunctionCoefficient x_zero([](const Vector& x) {
    if (x[0] < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  // Create a coefficient that indicates the x == 1 border of the cube
  mfem::FunctionCoefficient x_one([](const Vector& x) {
    if ((1. - x[0]) < 1.e-12) {
      return 1.;
    }
    return 0.;
  });

  Array<int> bdr_attr_list_zero = mfem_ext::makeBdrAttributeList(*pmesh_, x_zero);
  Array<int> bdr_attr_list_one  = mfem_ext::makeBdrAttributeList(*pmesh_, x_one);

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
  auto   nonlinear_diffusion = std::make_unique<mfem_ext::BilinearToNonlinearFormIntegrator>(diffusion);
  double multiplier          = 2.;
  double offset              = 1.;
  auto   transform           = [=](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::Vector& x) {
    mfem::Vector v(x);
    v *= multiplier;
    v += offset;
    return v;
  };
  auto transform_grad = [=](const mfem::FiniteElement&, mfem::ElementTransformation&, const mfem::DenseMatrix& x) {
    mfem::DenseMatrix m(x);
    m *= multiplier;
    return m;
  };
  auto transformed_diffusion = std::make_unique<mfem_ext::TransformedNonlinearFormIntegrator>(
      std::make_unique<mfem_ext::BilinearToNonlinearFormIntegrator>(diffusion), transform, transform_grad);

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

TEST_F(WrapperTests, attribute_modifier_coef)
{
  mfem::ConstantCoefficient three_and_a_half(3.5);
  // All the attributes in the mesh are "1", but for whatever reason the restricted coefficient
  // operates on attribute "2"
  mfem::Array<int> restrict_to(2);
  restrict_to[0] = 0;  // Don't apply if the attr is 1
  restrict_to[1] = 1;  // But apply if the attr is 0
  mfem::RestrictedCoefficient restrict_coef(three_and_a_half, restrict_to);

  // Everything gets converted to 2
  mfem::Array<int> modified_attrs(pmesh_->GetNE());
  modified_attrs = 2;
  mfem_ext::AttributeModifierCoefficient am_coef(modified_attrs, restrict_coef);

  ParGridFunction gf(pfes_.get());
  gf.ProjectCoefficient(am_coef);
  EXPECT_NEAR(gf.ComputeL2Error(three_and_a_half), 0.0, 1.e-8);
}

TEST_F(WrapperTests, vector_transform_coef)
{
  mfem::Vector one_two_three(dim_);
  one_two_three[0]    = 1;
  one_two_three[1]    = 2;
  one_two_three[2]    = 3;
  auto first_vec_coef = std::make_shared<mfem::VectorConstantCoefficient>(one_two_three);

  mfem::Vector four_five_six(dim_);
  four_five_six[0]     = 4;
  four_five_six[1]     = 5;
  four_five_six[2]     = 6;
  auto second_vec_coef = std::make_shared<mfem::VectorConstantCoefficient>(four_five_six);

  // Verify the answer using an MFEM VectorSumCoefficient since the transformation is
  // just an addition
  mfem::VectorSumCoefficient sum_coef(*first_vec_coef, *second_vec_coef);

  // Both of these do the same thing, but we can test both the single- and dual-vector transformations
  // by capturing the operand
  mfem_ext::TransformedVectorCoefficient mono_tv_coef(first_vec_coef,
                                                      [&four_five_six](const mfem::Vector& in, mfem::Vector& out) {
                                                        out = in;
                                                        out += four_five_six;
                                                      });
  ParGridFunction                        mono_gf(pfes_v_.get());
  mono_gf.ProjectCoefficient(mono_tv_coef);
  EXPECT_NEAR(mono_gf.ComputeL2Error(sum_coef), 0.0, 1.e-8);

  mfem_ext::TransformedVectorCoefficient dual_tv_coef(
      first_vec_coef, second_vec_coef, [](const mfem::Vector& first, const mfem::Vector& second, mfem::Vector& out) {
        out = first;
        out += second;
      });
  ParGridFunction dual_gf(pfes_v_.get());
  dual_gf.ProjectCoefficient(dual_tv_coef);
  EXPECT_NEAR(dual_gf.ComputeL2Error(sum_coef), 0.0, 1.e-8);
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
