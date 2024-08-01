// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

using namespace serac;

int num_procs, myid;

constexpr bool                 verbose = true;
std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int dim>
struct hcurl_qfunction {
  template <typename UnusedType1, typename UnusedType2, typename Position>
  SERAC_HOST_DEVICE auto operator()(UnusedType1, UnusedType2, Position X) const
  {
    auto dp_dX = serac::get<1>(X);
    auto I     = serac::Identity<dim>();
    return serac::tuple{serac::det(dp_dX + I), serac::zero{}};
  }
};

struct test_qfunction {
  template <typename P, typename Temperature>
  SERAC_HOST_DEVICE auto operator()(double, P position, Temperature temperature) const
  {
    static constexpr double a = 1.7;
    static constexpr double b = 0.0;
    // get the value and the gradient from the input tuple
    auto [X, dX_dxi] = position;
    auto [u, du_dx]  = temperature;
    auto source      = a * u - (100 * X[0] * X[1]);
    auto flux        = b * du_dx;
    return serac::tuple{source, flux};
  }
};

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, L2<p> test, L2<p> trial, Dimension<dim>)
{
  [[maybe_unused]] static constexpr double a = 1.7;
  [[maybe_unused]] static constexpr double b = 0.0;

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  // Create standard MFEM bilinear and linear forms on H1
  auto [fespace, fec] = serac::generateParFiniteElementSpace<test_space>(&mesh);

  mfem::ParBilinearForm A(fespace.get());

  // Add the mass term using the standard MFEM method
  mfem::ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new mfem::MassIntegrator(a_coef));

  // Assemble the bilinear form into a matrix
  A.Assemble(0);
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  // Create a linear form for the load term using the standard MFEM method
  mfem::ParLinearForm       f(fespace.get());
  mfem::FunctionCoefficient load_func([&](const mfem::Vector& coords) { return 100 * coords(0) * coords(1); });
  // FunctionCoefficient load_func([&]([[maybe_unused]] const Vector& coords) { return 1.0; });

  // Create and assemble the linear load term into a vector
  f.AddDomainIntegrator(new mfem::DomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(fespace.get());
  u_global.Randomize();

  mfem::Vector U(fespace->TrueVSize());
  u_global.GetTrueDofs(U);

  // Set up the same problem using weak form

  // Construct the new weak form object using the known test and trial spaces
#ifdef SERAC_USE_CUDA_KERNEL_EVALUATION
  Functional<test_space(trial_space), serac::ExecutionSpace::GPU> residual(&fespace, {&fespace});
#else
  Functional<test_space(trial_space), serac::ExecutionSpace::CPU> residual(&fespace, {&fespace});
#endif

  // Add the total domain residual term to the weak form
  residual.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, test_qfunction{}, mesh);

  // uncomment lines below to verify that compile-time error messages
  // explain L2 spaces are not currently supported in boundary integrals.
  //
  // residual.AddBoundaryIntegral(
  //    Dimension<dim-1>{},
  //    DependsOn<0>{},
  //    [&](double /*t*/, [[maybe_unused]] auto x, [[maybe_unused]] auto temperature) { return 1.0; },
  //    mesh);

  // Compute the residual using standard MFEM methods
  // mfem::Vector r1 = A * U - (*F);
  mfem::Vector r1(U.Size());
  A.Mult(U, r1);
  r1 -= (*F);

  // Compute the residual using weak form
  double t        = 0.0;
  auto [r2, drdU] = residual(t, differentiate_wrt(U));

  mfem::Vector diff(r1.Size());
  subtract(r1, r2, diff);

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << diff.Norml2() / r1.Norml2() << std::endl;
  }

  // Test that the two residuals are equivalent
  EXPECT_NEAR(0., diff.Norml2() / r1.Norml2(), 1.e-14);

  // Compute the gradient action using standard MFEM and Functional
  // mfem::Vector g1 = (*J) * U;
  mfem::Vector g1(U.Size());
  J->Mult(U, g1);

  mfem::Vector g2 = drdU(U);

  subtract(g1, g2, diff);

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << diff.Norml2() / g1.Norml2() << std::endl;
  }

  // Ensure the two methods generate the same result
  EXPECT_NEAR(0., diff.Norml2() / g1.Norml2(), 1.e-14);
}

TEST(L2, 2DConstant) { functional_test(*mesh2D, L2<0>{}, L2<0>{}, Dimension<2>{}); }
TEST(L2, 2DLinear) { functional_test(*mesh2D, L2<1>{}, L2<1>{}, Dimension<2>{}); }
TEST(L2, 2DQuadratic) { functional_test(*mesh2D, L2<2>{}, L2<2>{}, Dimension<2>{}); }
TEST(L2, 2DCubic) { functional_test(*mesh2D, L2<3>{}, L2<3>{}, Dimension<2>{}); }

TEST(L2, 3DLinear) { functional_test(*mesh3D, L2<1>{}, L2<1>{}, Dimension<3>{}); }
TEST(L2, 3DQuadratic) { functional_test(*mesh3D, L2<2>{}, L2<2>{}, Dimension<3>{}); }
TEST(L2, 3DCubic) { functional_test(*mesh3D, L2<3>{}, L2<3>{}, Dimension<3>{}); }

TEST(L2, 2DMixed)
{
  constexpr int dim = 2;
  using test_space  = L2<0>;
  using trial_space = H1<1, dim>;

  auto [L2fespace, L2fec] = serac::generateParFiniteElementSpace<test_space>(mesh2D.get());

  auto [H1fespace, H1fec] = serac::generateParFiniteElementSpace<trial_space>(mesh2D.get());

  serac::Functional<test_space(trial_space)> f(L2fespace.get(), {H1fespace.get()});
  f.AddDomainIntegral(serac::Dimension<dim>{}, serac::DependsOn<0>{}, hcurl_qfunction<dim>{}, *mesh2D);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
