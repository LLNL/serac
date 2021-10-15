// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
#include <gtest/gtest.h>

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

constexpr bool                 verbose = false;
std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

double measure_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm mass_lf(&fespace);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_lf.Assemble();

  mfem::ParGridFunction one_gf(&fespace);
  one_gf.ProjectCoefficient(one);

  return mass_lf(one_gf);
}

double x_moment_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm mass_lf(&fespace);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_lf.Assemble();

  mfem::FunctionCoefficient x_coordinate([](mfem::Vector x) { return x[0]; });
  mfem::ParGridFunction     x_gf(&fespace);
  x_gf.ProjectCoefficient(x_coordinate);

  return mass_lf(x_gf);
}

double sum_of_measures_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm lf(&fespace);
  lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  lf.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(one));
  lf.Assemble();

  mfem::ParGridFunction one_gf(&fespace);
  one_gf.ProjectCoefficient(one);

  return lf(one_gf);
}

template <typename T>
void check_gradient(Functional<T>& f, mfem::HypreParVector& U)
{
  int seed = 42;

  mfem::HypreParVector dU = U;
  dU                      = U;
  dU.Randomize(seed);

  double epsilon = 1.0e-8;

  // grad(f) evaluates the gradient of f at the last evaluation,
  // so we evaluate f(U) before calling grad(f)
  f(U);

  auto&                 dfdU     = grad(f);
  mfem::HypreParVector& dfdU_vec = dfdU;

  // TODO: fix this weird copy ctor behavior in mfem::HypreParVector
  auto U_plus = U;
  U_plus      = U;  // it hurts me to write this
  U_plus.Add(epsilon, dU);

  auto U_minus = U;
  U_minus      = U;
  U_minus.Add(-epsilon, dU);

  double df1 = (f(U_plus) - f(U_minus)) / (2 * epsilon);
  double df2 = InnerProduct(dfdU_vec, dU);
  double df3 = dfdU(dU);

  double relative_error1 = (df1 - df2) / df1;
  double relative_error2 = (df1 - df3) / df1;

  EXPECT_NEAR(0., relative_error1, 1.e-5);
  EXPECT_NEAR(0., relative_error2, 1.e-5);

  if (verbose) {
    std::cout << relative_error1 << " " << relative_error2 << std::endl;
  }
}

template <int p, int dim>
void functional_qoi_test(mfem::ParMesh& mesh, H1<p> trial, Dimension<dim>)
{
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParGridFunction     U_gf(&fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);

  mfem::HypreParVector* tmp = fespace.NewTrueDofVector();
  mfem::HypreParVector  U   = *tmp;
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = decltype(trial);

  // Construct the new functional object
  Functional<double(trial_space)> measure(&fespace);
  measure.AddDomainIntegral(
      Dimension<dim>{}, [&](auto /*x*/, auto /*u*/) { return 1.0; }, mesh);

  std::cout << "simplest possible domain qoi: " << measure(U) << " " << measure_mfem(mesh) << std::endl;

  Functional<double(trial_space)> x_moment(&fespace);
  x_moment.AddDomainIntegral(
      Dimension<dim>{}, [&](auto x, auto /*u*/) { return x[0]; }, mesh);

  std::cout << "spatially-dependent domain qoi: " << x_moment(U) << " " << x_moment_mfem(mesh) << std::endl;

  Functional<double(trial_space)> sum_of_measures(&fespace);
  sum_of_measures.AddDomainIntegral(
      Dimension<dim>{}, [&](auto /*x*/, auto /*u*/) { return 1.0; }, mesh);
  sum_of_measures.AddBoundaryIntegral(
      Dimension<dim - 1>{}, [&](auto /*x*/, auto /*n*/, auto /*u*/) { return 1.0; }, mesh);

  std::cout << "combined domain and boundary qoi: " << sum_of_measures(U) << " " << sum_of_measures_mfem(mesh)
            << std::endl;

  Functional<double(trial_space)> f(&fespace);
  f.AddDomainIntegral(
      Dimension<dim>{},
      [&](auto x, auto temperature) {
        auto [u, grad_u] = temperature;
        return x[0] * x[0] + sin(x[1]) + x[0] * u * u * u;
      },
      mesh);
  f.AddBoundaryIntegral(
      Dimension<dim - 1>{}, [&](auto x, auto /*n*/, auto u) { return x[0] - x[1] + cos(u * x[1]); }, mesh);

  // note: these answers are generated by a mathematica script that
  // integrates the qoi for these domains to machine precision
  //
  // see scripts/wolfram/qoi_examples.nb for more info
  constexpr double unused     = -1.0;
  constexpr double expected[] = {unused, unused, 9.71388562400895, 2.097457548402147e6};
  std::cout << "combined domain-and-boundary qoi with nonlinear spatial and temperature dependence: " << f(U) << " "
            << expected[dim] << std::endl;
  check_gradient(f, U);

  delete tmp;
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  functional_qoi_test(*mesh2D, H1<2>{}, Dimension<2>{});
  functional_qoi_test(*mesh3D, H1<1>{}, Dimension<3>{});

#if 0
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
#else
  MPI_Finalize();
  return 0;
#endif
}
