// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>
#include "mfem.hpp"

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

TEST(FunctionalMultiphysics, NonlinearThermalTest3D)
{
  int serial_refinement   = 1;
  int parallel_refinement = 0;

  constexpr auto p   = 3;
  constexpr auto dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  auto        mesh3D   = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh3D.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  mfem::Vector dU_dt(fespace.TrueVSize());
  int seed = 0;
  U.Randomize(seed);
  dU_dt.Randomize(seed + 1);

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  double cp    = 1.0;
  double rho   = 1.0;
  double kappa = 1.0;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space, trial_space)> residual(&fespace, {&fespace, &fespace});

  residual.AddVolumeIntegral(
      DependsOn<0, 1>{},
      [=](auto x, auto temperature, auto dtemperature_dt) {
        auto [u, du_dx]      = temperature;
        auto [du_dt, unused] = dtemperature_dt;
        auto source          = rho * cp * du_dt * du_dt - (100 * x[0] * x[1]);
        auto flux            = kappa * du_dx;
        return serac::tuple{source, flux};
      },
      *mesh3D);

  residual.AddSurfaceIntegral(
      DependsOn<0, 1>{},
      [=](auto x, auto /*n*/, auto temperature, auto dtemperature_dt) {
        auto [u, _0]     = temperature;
        auto [du_dt, _1] = dtemperature_dt;
        return x[0] + x[1] - cos(u) * du_dt;
      },
      *mesh3D);

  mfem::Vector r = residual(U, dU_dt);

  check_gradient(residual, U, dU_dt);
}

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
