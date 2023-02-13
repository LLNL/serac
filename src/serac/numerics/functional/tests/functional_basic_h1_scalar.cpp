// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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

template <int ptest, int ptrial, int dim>
void thermal_test()
{
  std::string meshfile;
  if (dim == 2) {
    meshfile = SERAC_REPO_DIR "/data/meshes/patch2D.mesh";
  }
  if (dim == 3) {
    meshfile = SERAC_REPO_DIR "/data/meshes/patch3D.mesh";
  }

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        test_fec = mfem::H1_FECollection(ptest, dim);
  mfem::ParFiniteElementSpace test_fespace(mesh.get(), &test_fec);

  auto                        trial_fec = mfem::H1_FECollection(ptrial, dim);
  mfem::ParFiniteElementSpace trial_fespace(mesh.get(), &trial_fec);

  mfem::Vector U(trial_fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<ptest>;
  using trial_space = H1<ptrial>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&test_fespace, {&trial_fespace});

  auto d00 = 1.0;
  auto d01 = 1.0 * make_tensor<dim>([](int i) { return i; });
  auto d10 = 1.0 * make_tensor<dim>([](int i) { return 2 * i * i; });
  auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j) { return i + j * (j + 1) + 1; });

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto source     = d00 * u + dot(d01, du_dx) - 0.0 * (100 * x[0] * x[1]);
        auto flux       = d10 * u + dot(d11, du_dx);
        return serac::tuple{source, flux};
      },
      *mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](auto x, auto /*n*/, auto temperature) {
        auto [u, du_dxi] = temperature;
        return x[0] + x[1] - cos(u);
      },
      *mesh);

  check_gradient(residual, U);
}

TEST(basic, thermal_test_2D) { thermal_test<1, 1, 2>(); }
//TEST(basic, thermal_test_3D) { thermal_test<1, 1, 3>(); }

TEST(mixed, thermal_test_2D_0) { thermal_test<1, 2, 2>(); }
TEST(mixed, thermal_test_2D_1) { thermal_test<2, 1, 2>(); }

//TEST(mixed, thermal_test_3D_0) { thermal_test<1, 2, 3>(); }
//TEST(mixed, thermal_test_3D_1) { thermal_test<2, 1, 3>(); }

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
