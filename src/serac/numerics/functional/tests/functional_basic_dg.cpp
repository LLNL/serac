// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

template <int p>
void L2_test_2D()
{
  constexpr int dim = 2;
  using test_space  = L2<p, dim>;
  using trial_space = L2<p, dim>;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch2D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  auto fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Construct the new functional object using the specified test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  constexpr int DERIVATIVE = 1;

  residual.AddInteriorFaceIntegral(
      Dimension<dim-1>{}, DependsOn<0>{},
      [=](double /*t*/, auto X, auto velocity) {

        // compute the surface normal
        auto dX_dxi = get<DERIVATIVE>(X);
        auto n = normalize(cross(dX_dxi));

        // extract the velocity values from each side of the interface
        // note: the orientation convention is such that the normal 
        //       computed as above will point from from side 1->2
        auto [u_1, u_2] = velocity; 

        auto a = dot(u_2 - u_1, n);

        auto s_1 = u_1 * a;
        auto s_2 = u_2 * a;

        return serac::tuple{s_1, s_2};

      },
      *mesh);

  double t = 0.0;
  check_gradient(residual, t, U);

}

TEST(basic, L2_test_2D_linear) { L2_test_2D<1>(); }

#if 0
template <int p>
void hcurl_test_3D()
{
  constexpr int dim = 3;

  std::string meshfile = SERAC_REPO_DIR "/data/meshes/patch3D.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile), 1);

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::ND_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  mfem::Vector U(fespace.TrueVSize());
  U.Randomize();

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = Hcurl<p>;
  using trial_space = Hcurl<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  residual.AddInteriorFaceIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /*x*/, auto vector_potential) {
        auto [A, curl_A] = vector_potential;
        auto source      = dot(d00, A) + dot(d01, curl_A);
        auto flux        = dot(d10, A) + dot(d11, curl_A);
        return serac::tuple{source, flux};
      },
      *mesh);

  check_gradient(residual, t, U);
}

TEST(basic, hcurl_test_3D_linear) { hcurl_test_3D<1>(); }

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
