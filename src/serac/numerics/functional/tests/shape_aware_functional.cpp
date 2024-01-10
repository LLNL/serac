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
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/shape_aware_functional.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

#include "serac/infrastructure/debug_print.hpp"

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int p, int dim>
void shape_aware_functional_test(mfem::ParMesh& mesh, double /* tolerance */)
{
  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec1 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace1(&mesh, &fec1);

  auto                        fec2 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace2(&mesh, &fec2, dim);

  ShapeAwareFunctional<H1<p, dim>, H1<p>(H1<p>)> residual(
      &fespace1, std::array<const mfem::ParFiniteElementSpace*, 2>{&fespace2, &fespace1});

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [](double /* time */, auto /* x */, auto u) {
        return serac::tuple{get<0>(u) * 0.0, get<1>(u) * 0.0};
      },
      mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{}, [](double /* time */, auto /* x */, auto u) { return get<0>(u) * 0.0; },
      mesh);
}

TEST(ShapeAware, 2DLinear) { shape_aware_functional_test<1, 2>(*mesh2D, 3.0e-14); }
TEST(ShapeAware, 2DQuadratic) { shape_aware_functional_test<2, 2>(*mesh2D, 3.0e-14); }

TEST(ShapeAware, 3DLinear) { shape_aware_functional_test<1, 3>(*mesh3D, 2.0e-13); }

// note: see description at top of file
TEST(ShapeAware, 3DQuadratic) { shape_aware_functional_test<2, 3>(*mesh3D, 1.5e-2); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
