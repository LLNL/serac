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

template < int p, int dim >
void elasticity_test(){

  // Define the test and trial spaces for an elasticity-like problem
  using test_space  = H1<p, dim>;
  using trial_space = H1<p, dim>;

  std::string meshfile;
  if (dim == 2) {
    meshfile = SERAC_REPO_DIR "/data/meshes/star.mesh";
  }
  if (dim == 3) {
    meshfile = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  }

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile));

  auto trial_fes = generateParFiniteElementSpace<trial_space>(mesh.get());
  auto test_fes = generateParFiniteElementSpace<test_space>(mesh.get());

  mfem::Vector U(trial_fes->TrueVSize());
  U.Randomize();

  Functional<test_space(trial_space)> residual(test_fes, {trial_fes});

  // note: this is not really an elasticity problem, it's testing source and flux
  // terms that have the appropriate shapes to ensure that all the differentiation
  // code works as intended
  residual.AddDomainIntegral(
      Dimension<dim>{},
      DependsOn<0>{},
      [=](auto x, auto displacement) {
        auto [u, du_dx] = displacement;
        auto source     = u * x[0];
        auto flux       = du_dx;
        return serac::tuple{source, flux};
      },
      *mesh);

  residual.AddBoundaryIntegral(Dimension<dim-1>{}, DependsOn<0>{}, [=](auto x, auto /*n*/, auto displacement) { 
        auto [u, du_dxi] = displacement;
        return u * x[0]; 
      }, 
      *mesh);

  check_gradient(residual, U);

  delete test_fes;
  delete trial_fes;
}

TEST(basic, elasticity_test_2D) { elasticity_test<2, 2>(); }
TEST(basic, elasticity_test_3D) { elasticity_test<2, 3>(); }

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
