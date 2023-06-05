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

TEST(basic, scalar_uniform) { 
  
  constexpr int dim = 2;
  constexpr int p = 2;
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch2D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Construct the new functional object using the specified test and trial spaces
  Functional<test_space(trial_space, double)> residual(&fespace, {&fespace, UniformSpace<double>{}});

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](auto /*x*/, auto temperature, auto param) {
        auto [u, dudx] = temperature;
        auto heat_flux = sin(u + param) * dudx;
        return serac::tuple{zero{}, heat_flux};
      },
      *mesh);

  auto r = residual(U, 3.0);
  
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
