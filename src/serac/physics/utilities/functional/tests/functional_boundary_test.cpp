// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace mfem;
using namespace serac;

int            num_procs, myid;
int            refinements = 0;
constexpr bool verbose     = true;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  auto                  fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);

  LinearForm          f(&fespace);
  FunctionCoefficient scalar_function([&](const Vector& coords) { return coords(0) * coords(1); });
  VectorFunctionCoefficient vector_function(dim, [&](const Vector& coords, Vector & output) { 
    output[0] = sin(coords[0]); 
    output[1] = coords[0] * coords[1];
  });

  f.AddBoundaryIntegrator(new BoundaryLFIntegrator(scalar_function));
  f.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(vector_function));
  f.Assemble();

  ParGridFunction u_global(&fespace);
  u_global.Randomize();

  Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  Functional<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddBoundaryIntegral(Dimension<dim-1>{}, [&](auto x, auto n, auto /* u */) { 
    tensor<double,dim> b{sin(x[0]), x[0] * x[1]};
    return x[0] * x[1] + dot(b, n);
  }, mesh);

  mfem::Vector r1 = f;
  mfem::Vector r2 = residual(u_global);

  if (verbose) {
    std::cout << "sum(r2):  " << r2.Sum() << std::endl;
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-4);

}

TEST(boundary, 3D_linear) { boundary_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(boundary, 3D_quadratic) { boundary_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }
TEST(boundary, 2D_linear) { boundary_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(boundary, 2D_quadratic) { boundary_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
