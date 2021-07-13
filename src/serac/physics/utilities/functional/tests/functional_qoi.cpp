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

double region_measure(mfem::ParMesh& mesh) {
  mfem::ConstantCoefficient one(1.0);

  auto fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm mass_lf(&fespace);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_lf.Assemble();

  mfem::ParGridFunction one_gf(&fespace);
  one_gf.ProjectCoefficient(one);

  return mass_lf(one_gf);
}

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_qoi_test(mfem::ParMesh& mesh, H1<p> trial, Dimension<dim>)
{

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParGridFunction u_global(&fespace);
  u_global = 0.0;

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  [[maybe_unused]] static constexpr double k = 2.0;

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = decltype(trial);

  // Construct the new functional object 
  Functional<QOI(trial_space)> measure(&fespace);
  measure.AddDomainIntegral(Dimension<dim>{}, [&](auto, auto) { return 1.0; }, mesh);

  // evaluate the quantity of interest (region measure)
  double m1 = measure(U);
  double m2 = region_measure(mesh);

  std::cout << m1 << " " << m2 << std::endl;

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

  functional_qoi_test(*mesh2D, H1<1>{}, Dimension<2>{});
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
