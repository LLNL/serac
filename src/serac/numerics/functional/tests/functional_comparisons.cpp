// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include <gtest/gtest.h>

using namespace serac;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

constexpr bool                 verbose = false;
std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

static constexpr double a = 1.7;
static constexpr double b = 2.1;

template <int dim>
struct hcurl_qfunction {
  template <typename x_t, typename vector_potential_t>
  SERAC_HOST_DEVICE auto operator()(x_t x, vector_potential_t vector_potential) const
  {
    auto [A, curl_A] = vector_potential;
    auto J_term      = a * A - tensor<double, dim>{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]};
    auto H_term      = b * curl_A;
    return serac::tuple{J_term, H_term};
  }
};

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  // by default, mfem uses a different integration rule than serac
  // so we manually specify the one that we use
  const mfem::FiniteElement&   el = *fespace.GetFE(0);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

  mfem::ParBilinearForm A(&fespace);

  // Add the mass term using the standard MFEM method
  mfem::ConstantCoefficient a_coef(a);
  auto*                     mass = new mfem::MassIntegrator(a_coef);
  mass->SetIntRule(&ir);
  A.AddDomainIntegrator(mass);

  // Add the diffusion term using the standard MFEM method
  mfem::ConstantCoefficient b_coef(b);
  auto*                     diffusion = new mfem::DiffusionIntegrator(b_coef);
  diffusion->SetIntRule(&ir);
  A.AddDomainIntegrator(diffusion);

  // Assemble the bilinear form into a matrix
  A.Assemble(0);

  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J_mfem(A.ParallelAssemble());

  // Create a linear form for the load term using the standard MFEM method
  mfem::ParLinearForm       f(&fespace);
  mfem::FunctionCoefficient load_func([&](const mfem::Vector& coords) { return 100 * coords(0) * coords(1); });

  // Create and assemble the linear load term into a vector
  auto* load = new mfem::DomainLFIntegrator(load_func);
  load->SetIntRule(&ir);
  f.AddDomainIntegrator(load);
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  // Set up the same problem using functional

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](auto x, auto temperature) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = temperature;
        auto source     = a * u - (100 * x[0] * x[1]);
        auto flux       = b * du_dx;
        return serac::tuple{source, flux};
      },
      mesh);

  // Compute the residual using standard MFEM methods
  mfem::Vector r1 = (*J_mfem) * U - (*F);

  // Compute the residual using functional
  mfem::Vector r2 = residual(U);

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  // Test that the two residuals are equivalent
  EXPECT_NEAR(0.0, mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  // Compute the gradient using functional
  auto [r, drdU] = residual(differentiate_wrt(U));

  std::unique_ptr<mfem::HypreParMatrix> J_func = assemble(drdU);

  // Compute the gradient action using standard MFEM and functional
  mfem::Vector g1 = (*J_mfem) * U;
  mfem::Vector g2 = drdU * U;
  mfem::Vector g3 = (*J_func) * U;

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g3||: " << g3.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
    std::cout << "||g1-g3||/||g1||: " << mfem::Vector(g1 - g3).Norml2() / g1.Norml2() << std::endl;
  }

  // Ensure the two methods generate the same result
  EXPECT_NEAR(0.0, mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);
  EXPECT_NEAR(0.0, mfem::Vector(g1 - g3).Norml2() / g1.Norml2(), 1.e-14);
}

// this test sets up a toy "elasticity" problem where the residual includes contributions
// from a displacement-dependent body force term and an isotropically linear elastic stress response
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p, dim> test, H1<p, dim> trial, Dimension<dim>)
{
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec, dim);

  // by default, mfem uses a different integration rule than serac
  // so we manually specify the one that we use
  const mfem::FiniteElement&   el = *fespace.GetFE(0);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

  mfem::ParBilinearForm A(&fespace);

  mfem::ConstantCoefficient a_coef(a);
  auto*                     mass = new mfem::VectorMassIntegrator(a_coef);
  mass->SetIntRule(&ir);
  A.AddDomainIntegrator(mass);

  mfem::ConstantCoefficient lambda_coef(b);
  mfem::ConstantCoefficient mu_coef(b);
  auto*                     elasticity = new mfem::ElasticityIntegrator(lambda_coef, mu_coef);
  elasticity->SetIntRule(&ir);
  A.AddDomainIntegrator(elasticity);
  A.Assemble(0);
  A.Finalize();

  std::unique_ptr<mfem::HypreParMatrix> J_mfem(A.ParallelAssemble());

  mfem::ParLinearForm             f(&fespace);
  mfem::VectorFunctionCoefficient load_func(dim, [&](const mfem::Vector& /*coords*/, mfem::Vector& force) {
    force    = 0.0;
    force(0) = -1.0;
  });

  auto* load = new mfem::VectorDomainLFIntegrator(load_func);
  load->SetIntRule(&ir);
  f.AddDomainIntegrator(load);
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  [[maybe_unused]] static constexpr auto I = DenseIdentity<dim>();

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](auto /*x*/, auto displacement) {
        auto [u, du_dx] = displacement;
        auto body_force = a * u + I[0];
        auto strain     = 0.5 * (du_dx + transpose(du_dx));
        auto stress     = b * tr(strain) * I + 2.0 * b * strain;
        return serac::tuple{body_force, stress};
      },
      mesh);

  mfem::Vector r1 = (*J_mfem) * U - (*F);
  mfem::Vector r2 = residual(U);

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  auto [r, drdU] = residual(differentiate_wrt(U));

  std::unique_ptr<mfem::HypreParMatrix> J_func = assemble(drdU);

  mfem::Vector g1 = (*J_mfem) * U;
  mfem::Vector g2 = drdU * U;
  mfem::Vector g3 = (*J_func) * U;

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g3||: " << g3.Norml2() << std::endl;

    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
    std::cout << "||g1-g3||/||g1||: " << mfem::Vector(g1 - g3).Norml2() / g1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);
  EXPECT_NEAR(0., mfem::Vector(g1 - g3).Norml2() / g1.Norml2(), 1.e-14);
}

// this test sets up part of a toy "magnetic diffusion" problem where the residual includes contributions
// from a vector-potential-proportional J and an isotropically linear H
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, Hcurl<p> test, Hcurl<p> trial, Dimension<dim>)
{
  auto                        fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  // by default, mfem uses a different integration rule than serac
  // so we manually specify the one that we use
  const mfem::FiniteElement&   el = *fespace.GetFE(0);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

  mfem::ParBilinearForm B(&fespace);

  mfem::ConstantCoefficient a_coef(a);
  auto*                     mass = new mfem::VectorFEMassIntegrator(a_coef);
  mass->SetIntRule(&ir);
  B.AddDomainIntegrator(mass);

  mfem::ConstantCoefficient b_coef(b);
  auto*                     curlcurl = new mfem::CurlCurlIntegrator(b_coef);
  curlcurl->SetIntRule(&ir);
  B.AddDomainIntegrator(curlcurl);
  B.Assemble(0);
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J_mfem(B.ParallelAssemble());

  mfem::ParLinearForm             f(&fespace);
  mfem::VectorFunctionCoefficient load_func(dim, [&](const mfem::Vector& coords, mfem::Vector& output) {
    double x  = coords(0);
    double y  = coords(1);
    output    = 0.0;
    output(0) = 10 * x * y;
    output(1) = -5 * (x - y) * y;
  });

  auto* load = new mfem::VectorFEDomainLFIntegrator(load_func);
  load->SetIntRule(&ir);

  f.AddDomainIntegrator(load);
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, hcurl_qfunction<dim>{}, mesh);

  mfem::Vector r1 = (*J_mfem) * U - (*F);
  mfem::Vector r2 = residual(U);

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-13);

  auto [r, drdU] = residual(differentiate_wrt(U));

  std::unique_ptr<mfem::HypreParMatrix> J_func = assemble(drdU);

  mfem::Vector g1 = (*J_mfem) * U;
  mfem::Vector g2 = drdU * U;
  mfem::Vector g3 = (*J_func) * U;

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g3||: " << g3.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
    std::cout << "||g1-g3||/||g1||: " << mfem::Vector(g1 - g3).Norml2() / g1.Norml2() << std::endl;
  }
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-13);
  EXPECT_NEAR(0., mfem::Vector(g1 - g3).Norml2() / g1.Norml2(), 1.e-13);
}

TEST(Thermal, 2DLinear) { functional_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(Thermal, 2DQuadratic) { functional_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }
TEST(Thermal, 2DCubic) { functional_test(*mesh2D, H1<3>{}, H1<3>{}, Dimension<2>{}); }

TEST(Thermal, 3DLinear) { functional_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(Thermal, 3DQuadratic) { functional_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }
TEST(Thermal, 3DCubic) { functional_test(*mesh3D, H1<3>{}, H1<3>{}, Dimension<3>{}); }

// TODO: reenable these once hcurl implements of simplex elements is finished
// TEST(Hcurl, 2DLinear) { functional_test(*mesh2D, Hcurl<1>{}, Hcurl<1>{}, Dimension<2>{}); }
// TEST(Hcurl, 2DQuadratic) { functional_test(*mesh2D, Hcurl<2>{}, Hcurl<2>{}, Dimension<2>{}); }
// TEST(Hcurl, 2DCubic) { functional_test(*mesh2D, Hcurl<3>{}, Hcurl<3>{}, Dimension<2>{}); }
//
// TEST(Hcurl, 3DLinear) { functional_test(*mesh3D, Hcurl<1>{}, Hcurl<1>{}, Dimension<3>{}); }
// TEST(Hcurl, 3DQuadratic) { functional_test(*mesh3D, Hcurl<2>{}, Hcurl<2>{}, Dimension<3>{}); }
// TEST(Hcurl, 3DCubic) { functional_test(*mesh3D, Hcurl<3>{}, Hcurl<3>{}, Dimension<3>{}); }

TEST(Elasticity, 2DLinear) { functional_test(*mesh2D, H1<1, 2>{}, H1<1, 2>{}, Dimension<2>{}); }
TEST(Elasticity, 2DQuadratic) { functional_test(*mesh2D, H1<2, 2>{}, H1<2, 2>{}, Dimension<2>{}); }
TEST(Elasticity, 2DCubic) { functional_test(*mesh2D, H1<3, 2>{}, H1<3, 2>{}, Dimension<2>{}); }

TEST(Elasticity, 3DLinear) { functional_test(*mesh3D, H1<1, 3>{}, H1<1, 3>{}, Dimension<3>{}); }
TEST(Elasticity, 3DQuadratic) { functional_test(*mesh3D, H1<2, 3>{}, H1<2, 3>{}, Dimension<3>{}); }
TEST(Elasticity, 3DCubic) { functional_test(*mesh3D, H1<3, 3>{}, H1<3, 3>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&serial_refinement, "-r", "--ref", "");
  args.AddOption(&parallel_refinement, "-pr", "--pref", "");
  args.AddOption(&nsamples, "-n", "--n-samples", "Samples per test");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    exit(1);
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/patch2D_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);
  mesh2D->ExchangeFaceNbrData();

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);
  mesh3D->ExchangeFaceNbrData();

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
