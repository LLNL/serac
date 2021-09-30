// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"

using namespace serac;
using namespace serac::profiling;

template <typename T>
void check_gradient(Functional<T>& f, mfem::GridFunction& U)
{
  int                seed = 42;
  mfem::GridFunction dU   = U;
  dU.Randomize(seed);

  double epsilon = 1.0e-8;

  // grad(f) evaluates the gradient of f at the last evaluation,
  // so we evaluate f(U) before calling grad(f)
  f(U);

  auto& dfdU = grad(f);

  auto U_plus = U;
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus.Add(-epsilon, dU);

  mfem::Vector df1 = f(U_plus);
  df1 -= f(U_minus);
  df1 /= (2 * epsilon);

  mfem::Vector df2 = mfem::SparseMatrix(dfdU) * dU;
  mfem::Vector df3 = dfdU(dU);

  double relative_error1 = df1.DistanceTo(df2) / df1.Norml2();
  double relative_error2 = df1.DistanceTo(df3) / df1.Norml2();

  std::cout << relative_error1 << " " << relative_error2 << std::endl;

  EXPECT_NEAR(0., relative_error1, 1.e-6);
  EXPECT_NEAR(0., relative_error2, 1.e-6);
}

int main(int argc, char* argv[])
{
  int num_procs, myid;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  std::unique_ptr<mfem::ParMesh> mesh3D;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  constexpr auto p   = 2;
  constexpr auto dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D               = mesh::refineAndDistribute(buildMeshFromFile(meshfile), serial_refinement, parallel_refinement);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh3D.get(), &fec);

  // Set a random state to evaluate the residual
  mfem::GridFunction U(&fespace);
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddVolumeIntegral(
      [=](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto source     = u * u - (100 * x[0] * x[1]);
        auto flux       = du_dx;
        return serac::tuple{source, flux};
      },
      *mesh3D);

  residual.AddSurfaceIntegral([=](auto x, auto /*n*/, auto u) { return x[0] + x[1] - cos(u); }, *mesh3D);

  mfem::SparseMatrix K = grad(residual);

  check_gradient(residual, U);

  MPI_Finalize();
}
