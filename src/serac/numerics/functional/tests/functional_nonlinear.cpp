// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
#include <gtest/gtest.h>

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;

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

template <typename T>
void check_gradient(Functional<T>& f, mfem::Vector& U)
{
  int seed = 42;

  mfem::Vector dU(U.Size());
  dU.Randomize(seed);

  double epsilon = 1.0e-8;

  auto U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus.Add(-epsilon, dU);

  mfem::Vector df1 = f(U_plus);
  df1 -= f(U_minus);
  df1 /= (2 * epsilon);

  auto [value, dfdU] = f(differentiate_wrt(U));
  mfem::Vector df2   = dfdU(dU);

  mfem::HypreParMatrix* dfdU_matrix = dfdU;

  mfem::Vector df3 = (*dfdU_matrix) * dU;

  double relative_error1 = df1.DistanceTo(df2) / df1.Norml2();
  double relative_error2 = df1.DistanceTo(df3) / df1.Norml2();

  EXPECT_NEAR(0., relative_error1, 5.e-6);
  EXPECT_NEAR(0., relative_error2, 5.e-6);

  std::cout << relative_error1 << " " << relative_error2 << std::endl;

  delete dfdU_matrix;
}

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  std::string postfix = concat("_H1<", p, ">");
  serac::profiling::initializeCaliper();

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  // Set a random state to evaluate the residual
  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{},
      [=](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto source     = a * u * u - (100 * x[0] * x[1]);
        auto flux       = b * du_dx;
        return serac::tuple{source, flux};
      },
      mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{},
      [=](auto x, auto /*n*/, auto temperature) {
        auto u = get<0>(temperature);
        return x[0] + x[1] - cos(u);
      },
      mesh);

  check_gradient(residual, U);

  serac::profiling::terminateCaliper();
}

template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, H1<p, dim> test, H1<p, dim> trial, Dimension<dim>)
{
  std::string postfix = concat("_H1<", p, ",", dim, ">");
  serac::profiling::initializeCaliper();

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec, dim);

  // Set a random state to evaluate the residual
  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{},
      [=](auto /*x*/, auto displacement) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = displacement;
        auto source     = a * u * u[0];
        auto flux       = b * du_dx;
        return serac::tuple{source, flux};
      },
      mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{},
      [=](auto x, auto n, auto displacement) {
        auto u = get<0>(displacement);
        return (x[0] + x[1] - cos(u[0])) * n;
      },
      mesh);

  check_gradient(residual, U);

  serac::profiling::terminateCaliper();
}

TEST(thermal, 2D_linear) { functional_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(thermal, 2D_quadratic) { functional_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }
TEST(thermal, 2D_cubic) { functional_test(*mesh2D, H1<3>{}, H1<3>{}, Dimension<2>{}); }

TEST(thermal, 3D_linear) { functional_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(thermal, 3D_quadratic) { functional_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }
TEST(thermal, 3D_cubic) { functional_test(*mesh3D, H1<3>{}, H1<3>{}, Dimension<3>{}); }

TEST(elasticity, 2D_linear) { functional_test(*mesh2D, H1<1, 2>{}, H1<1, 2>{}, Dimension<2>{}); }
TEST(elasticity, 2D_quadratic) { functional_test(*mesh2D, H1<2, 2>{}, H1<2, 2>{}, Dimension<2>{}); }
TEST(elasticity, 2D_cubic) { functional_test(*mesh2D, H1<3, 2>{}, H1<3, 2>{}, Dimension<2>{}); }

TEST(elasticity, 3D_linear) { functional_test(*mesh3D, H1<1, 3>{}, H1<1, 3>{}, Dimension<3>{}); }
TEST(elasticity, 3D_quadratic) { functional_test(*mesh3D, H1<2, 3>{}, H1<2, 3>{}, Dimension<3>{}); }
TEST(elasticity, 3D_cubic) { functional_test(*mesh3D, H1<3, 3>{}, H1<3, 3>{}, Dimension<3>{}); }

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
