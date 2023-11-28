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
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

template <int p, int dim>
void weird_mixed_test(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Define vector-valued test and trial spaces of different sizes
  using test_space  = H1<p, dim + 1>;
  using trial_space = H1<p, dim + 2>;

  auto [trial_fes, trial_col] = generateParFiniteElementSpace<trial_space>(mesh.get());
  auto [test_fes, test_col]   = generateParFiniteElementSpace<test_space>(mesh.get());

  mfem::Vector U(trial_fes->TrueVSize());
  U.Randomize();

  Functional<test_space(trial_space)> residual(test_fes.get(), {trial_fes.get()});

  auto d11 =
      1.0 * make_tensor<dim + 1, dim, dim + 2, dim>([](int i, int j, int k, int l) { return i - j + 2 * k - 3 * l; });

  auto s11 = 1.0 * make_tensor<dim + 1, dim + 2>([](int i, int j) { return i * i - j; });

  // note: this is not really an elasticity problem, it's testing source and flux
  // terms that have the appropriate shapes to ensure that all the differentiation
  // code works as intended
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /* x */, auto displacement) {
        auto [u, du_dx] = displacement;
        auto source     = zero{};
        auto flux       = double_dot(d11, du_dx);
        return serac::tuple{source, flux};
      },
      *mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        return dot(s11, u) * X[0];
      },
      *mesh);

  double t = 0.0;
  check_gradient(residual, t, U);
}

template <int p, int dim>
void elasticity_test(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Define the test and trial spaces for an elasticity-like problem
  using test_space  = H1<p, dim>;
  using trial_space = H1<p, dim>;

  auto [trial_fes, trial_col] = generateParFiniteElementSpace<trial_space>(mesh.get());
  auto [test_fes, test_col]   = generateParFiniteElementSpace<test_space>(mesh.get());

  mfem::Vector U(trial_fes->TrueVSize());
  U.Randomize();

  Functional<test_space(trial_space)> residual(test_fes.get(), {trial_fes.get()});

  [[maybe_unused]] auto d00 = make_tensor<dim, dim>([](int i, int j) { return i + 2 * j + 1; });
  [[maybe_unused]] auto d01 = make_tensor<dim, dim, dim>([](int i, int j, int k) { return i + 2 * j - k + 1; });
  [[maybe_unused]] auto d10 = make_tensor<dim, dim, dim>([](int i, int j, int k) { return i + 3 * j - 2 * k; });
  [[maybe_unused]] auto d11 =
      make_tensor<dim, dim, dim, dim>([](int i, int j, int k, int l) { return i - j + 2 * k - 3 * l + 1; });

  // note: this is not really an elasticity problem, it's testing source and flux
  // terms that have the appropriate shapes to ensure that all the differentiation
  // code works as intended
  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto /* x */, auto displacement) {
        auto [u, du_dx] = displacement;
        auto source     = dot(d00, u) + double_dot(d01, du_dx);
        auto flux       = dot(d10, u) + double_dot(d11, du_dx);
        return serac::tuple{source, flux};
      },
      *mesh);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        return u * X[0];
      },
      *mesh);

  double t = 0.0;
  check_gradient(residual, t, U);
}

void test_suite(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SERAC_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    constexpr int dim = 2;
    elasticity_test<1, dim>(mesh);
    elasticity_test<2, dim>(mesh);
    weird_mixed_test<1, dim>(mesh);
    weird_mixed_test<2, dim>(mesh);
  }

  if (mesh->Dimension() == 3) {
    constexpr int dim = 3;
    elasticity_test<1, dim>(mesh);
    elasticity_test<2, dim>(mesh);
    weird_mixed_test<1, dim>(mesh);
    weird_mixed_test<2, dim>(mesh);
  }
}

TEST(VectorValuedH1, test_suite_tris) { test_suite("/data/meshes/patch2D_tris.mesh"); }
TEST(VectorValuedH1, test_suite_quads) { test_suite("/data/meshes/patch2D_quads.mesh"); }
TEST(VectorValuedH1, test_suite_tris_and_quads) { test_suite("/data/meshes/patch2D_tris_and_quads.mesh"); }

TEST(VectorValuedH1, test_suite_tets) { test_suite("/data/meshes/patch3D_tets.mesh"); }
TEST(VectorValuedH1, test_suite_hexes) { test_suite("/data/meshes/patch3D_hexes.mesh"); }
TEST(VectorValuedH1, test_suite_tets_and_hexes) { test_suite("/data/meshes/patch3D_tets_and_hexes.mesh"); }

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
