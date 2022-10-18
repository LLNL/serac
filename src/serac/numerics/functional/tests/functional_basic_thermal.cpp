// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

TEST(basic, nonlinear_thermal_test_2D)
{
  constexpr auto p   = 1;
  constexpr auto dim = 2;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/star.mesh";
  auto        mesh2D   = mesh::refineAndDistribute(buildMeshFromFile(meshfile));

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh2D.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  auto d00 = 1.0;
  auto d01 = 1.0 * make_tensor<dim>([](int i){ return i; });
  auto d10 = 1.0 * make_tensor<dim>([](int i){ return 2*i*i; });
  auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j){ return i + j*(j+1) + 1; });

  residual.AddAreaIntegral(
      DependsOn<0>{},
      [=](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto source     = d00 * u + dot(d01, du_dx) - 0.0 * (100 * x[0] * x[1]);
        auto flux       = d10 * u + dot(d11, du_dx);
        return serac::tuple{source, flux};
      },
      *mesh2D);

  // TODO: reenable surface integrals
  //residual.AddBoundaryIntegral(Dimension<1>{}, DependsOn<0>{}, [=](auto x, auto /*n*/, auto temperature) { 
  //      auto [u, du_dxi] = temperature;
  //      return x[0] + x[1] - cos(u); 
  //    }, 
  //    *mesh2D);

  check_gradient(residual, U);
}

#if 0
TEST(basic, nonlinear_thermal_test_3D)
{
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  constexpr auto p   = 2;
  constexpr auto dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  auto        mesh3D   = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh3D.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddVolumeIntegral(
      DependsOn<0>{},
      [=](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto source     = u * u - (100 * x[0] * x[1]);
        auto flux       = du_dx;
        return serac::tuple{source, flux};
      },
      *mesh3D);

  // TODO: reenable surface integrals
  residual.AddSurfaceIntegral(
      DependsOn<0>{}, 
      [=](auto x, auto /*n*/, auto temperature) { 
        auto [u, du_dxi] = temperature;
        return x[0] + x[1] - cos(u) + norm(du_dxi); 
      }, 
      *mesh3D);

  check_gradient(residual, U);
}
#endif

int main(int argc, char* argv[])
{
  int num_procs, myid;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
