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

template <int dim>
tensor<double, dim> average(std::vector<tensor<double, dim>>& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}

template <int ptest, int ptrial, int dim>
void whole_mesh_comparison_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Create standard MFEM bilinear and linear forms on H1
  auto test_fec = mfem::H1_FECollection(ptest, dim);
  mfem::ParFiniteElementSpace test_fespace(mesh.get(), &test_fec);

  auto trial_fec = mfem::H1_FECollection(ptrial, dim);
  mfem::ParFiniteElementSpace trial_fespace(mesh.get(), &trial_fec);

  mfem::Vector U(trial_fespace.TrueVSize());

  mfem::ParGridFunction U_gf(&trial_fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = H1<ptest>;
  using trial_space = H1<ptrial>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&test_fespace, {&trial_fespace});
  Functional<test_space(trial_space)> residual_comparison(&test_fespace, {&trial_fespace});

  auto everything = [](std::vector<tensor<double, dim>>, int /* attr */) { return true; };
  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 0.5; };
  auto on_right = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] >= 0.5; };
  auto on_bottom = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] < 0.5; };
  auto on_top = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] >= 0.5; };

  Domain whole_mesh = Domain::ofElements(*mesh, everything);
  Domain left = Domain::ofElements(*mesh, on_left);
  Domain right = Domain::ofElements(*mesh, on_right);

  Domain whole_boundary = Domain::ofBoundaryElements(*mesh, everything);
  Domain bottom_boundary = Domain::ofBoundaryElements(*mesh, on_bottom);
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  auto d00 = 1.0;
  auto d01 = 1.0 * make_tensor<dim>([](int i) { return i; });
  auto d10 = 1.0 * make_tensor<dim>([](int i) { return 2 * i * i; });
  auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j) { return i + j * (j + 1) + 1; });

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto source = d00 * u + dot(d01, du_dX) - 0.0 * (100 * X[0] * X[1]);
        auto flux = d10 * u + dot(d11, du_dX);
        return serac::tuple{source, flux};
      },
      left);

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto source = d00 * u + dot(d01, du_dX) - 0.0 * (100 * X[0] * X[1]);
        auto flux = d10 * u + dot(d11, du_dX);
        return serac::tuple{source, flux};
      },
      right);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = temperature;
        return X[0] + X[1] - cos(u);
      },
      bottom_boundary);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = temperature;
        return X[0] + X[1] - cos(u);
      },
      top_boundary);

  double t = 0.0;
  check_gradient(residual, t, U);

  auto r0 = residual(t, U);

  //////////////

  residual_comparison.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto source = d00 * u + dot(d01, du_dX) - 0.0 * (100 * X[0] * X[1]);
        auto flux = d10 * u + dot(d11, du_dX);
        return serac::tuple{source, flux};
      },
      whole_mesh);

  residual_comparison.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = temperature;
        return X[0] + X[1] - cos(u);
      },
      whole_boundary);

  auto r1 = residual_comparison(t, U);

  EXPECT_LT(r0.DistanceTo(r1.GetData()), 1.0e-14);
}

template <int ptest, int ptrial>
void whole_mesh_comparison_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SERAC_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    whole_mesh_comparison_test_impl<ptest, ptrial, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    whole_mesh_comparison_test_impl<ptest, ptrial, 3>(mesh);
  }
}

TEST(basic, whole_mesh_comparison_tris) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_tris.mesh"); }
TEST(basic, whole_mesh_comparison_quads) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_quads.mesh"); }
TEST(basic, whole_mesh_comparison_tris_and_quads)
{
  whole_mesh_comparison_test<1, 1>("/data/meshes/patch2D_tris_and_quads.mesh");
}

TEST(basic, whole_mesh_comparison_tets) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_tets.mesh"); }
TEST(basic, whole_mesh_comparison_hexes) { whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_hexes.mesh"); }
TEST(basic, whole_mesh_comparison_tets_and_hexes)
{
  whole_mesh_comparison_test<1, 1>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

TEST(mixed, whole_mesh_comparison_tris_and_quads)
{
  whole_mesh_comparison_test<2, 1>("/data/meshes/patch2D_tris_and_quads.mesh");
}
TEST(mixed, whole_mesh_comparison_tets_and_hexes)
{
  whole_mesh_comparison_test<2, 1>("/data/meshes/patch3D_tets_and_hexes.mesh");
}

template <int ptest, int ptrial, int dim>
void partial_mesh_comparison_test_impl(std::unique_ptr<mfem::ParMesh>& mesh)
{
  // Create standard MFEM bilinear and linear forms on H1
  auto test_fec = mfem::H1_FECollection(ptest, dim);
  mfem::ParFiniteElementSpace test_fespace(mesh.get(), &test_fec);

  auto trial_fec = mfem::H1_FECollection(ptrial, dim);
  mfem::ParFiniteElementSpace trial_fespace(mesh.get(), &trial_fec);

  mfem::Vector U(trial_fespace.TrueVSize());

  mfem::ParGridFunction U_gf(&trial_fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = H1<ptest>;
  using trial_space = H1<ptrial>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&test_fespace, {&trial_fespace});
  Functional<test_space(trial_space)> residual_comparison(&test_fespace, {&trial_fespace});

  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 4.0; };
  Domain left = Domain::ofElements(*mesh, on_left);

  auto on_top = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[1] >= 0.99; };
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  auto d00 = 1.0;
  auto d01 = 1.0 * make_tensor<dim>([](int i) { return i; });
  auto d10 = 1.0 * make_tensor<dim>([](int i) { return 2 * i * i; });
  auto d11 = 1.0 * make_tensor<dim, dim>([](int i, int j) { return i + j * (j + 1) + 1; });

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        auto source = d00 * u + dot(d01, du_dX) - 0.0 * (100 * X[0] * X[1]);
        auto flux = d10 * u + dot(d11, du_dX);
        return serac::tuple{source, flux};
      },
      left);

  residual.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = temperature;
        return X[0] + X[1] - cos(u);
      },
      top_boundary);

  double t = 0.0;
  check_gradient(residual, t, U);

  auto r0 = residual(t, U);

  //////////////

  residual_comparison.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dX] = temperature;
        double mask = (X[0] < 4.0);
        auto source = mask * (d00 * u + dot(d01, du_dX) - 0.0 * (100 * X[0] * X[1]));
        auto flux = mask * (d10 * u + dot(d11, du_dX));
        return serac::tuple{source, flux};
      },
      *mesh);

  residual_comparison.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0>{},
      [=](double /*t*/, auto position, auto temperature) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = temperature;
        double mask = (X[1] >= 0.99);
        return (X[0] + X[1] - cos(u)) * mask;
      },
      *mesh);

  auto r1 = residual_comparison(t, U);

  EXPECT_LT(r0.DistanceTo(r1.GetData()), 1.0e-14);
}

template <int ptest, int ptrial>
void partial_mesh_comparison_test(std::string meshfile)
{
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SERAC_REPO_DIR + meshfile), 1);

  if (mesh->Dimension() == 2) {
    partial_mesh_comparison_test_impl<ptest, ptrial, 2>(mesh);
  }

  if (mesh->Dimension() == 3) {
    partial_mesh_comparison_test_impl<ptest, ptrial, 3>(mesh);
  }
}

TEST(basic, partial_mesh_comparison_tris) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-tri.mesh"); }
TEST(basic, partial_mesh_comparison_quads) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-quad.mesh"); }

TEST(basic, partial_mesh_comparison_tets) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-tet.mesh"); }
TEST(basic, partial_mesh_comparison_hexes) { partial_mesh_comparison_test<1, 1>("/data/meshes/beam-hex.mesh"); }

TEST(qoi, partial_boundary)
{
  constexpr auto dim = 3;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SERAC_REPO_DIR "/data/meshes/beam-hex.mesh"), 1);

  auto trial_fec = mfem::H1_FECollection(1, dim);
  mfem::ParFiniteElementSpace trial_fespace(mesh.get(), &trial_fec);

  mfem::Vector U(trial_fespace.TrueVSize());

  mfem::ParGridFunction U_gf(&trial_fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = double;
  using trial_space = H1<1>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> qoi({&trial_fespace});

  auto on_top = [](std::vector<tensor<double, 3>> X, int /* attr */) { return average(X)[1] >= 0.99; };
  Domain top_boundary = Domain::ofBoundaryElements(*mesh, on_top);

  qoi.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn</*nothing*/>{}, [=](double /*t*/, auto /*position*/) { return 1.0; },
      top_boundary);

  double time = 0.0;

  auto area = qoi(time, U);

  EXPECT_NEAR(area, 8.0, 1.0e-14);
}

TEST(qoi, partial_domain)
{
  constexpr auto dim = 3;
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(SERAC_REPO_DIR "/data/meshes/beam-hex.mesh"), 1);

  auto trial_fec = mfem::H1_FECollection(1, dim);
  mfem::ParFiniteElementSpace trial_fespace(mesh.get(), &trial_fec);

  mfem::Vector U(trial_fespace.TrueVSize());

  mfem::ParGridFunction U_gf(&trial_fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using test_space = double;
  using trial_space = H1<1>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> qoi({&trial_fespace});

  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 4.0; };
  Domain left = Domain::ofElements(*mesh, on_left);

  qoi.AddDomainIntegral(
      Dimension<dim>{}, DependsOn</*nothing*/>{}, [=](double /*t*/, auto /*position*/) { return 1.0; }, left);

  double time = 0.0;
  auto volume = qoi(time, U);

  EXPECT_NEAR(volume, 4.0, 1.0e-14);
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
