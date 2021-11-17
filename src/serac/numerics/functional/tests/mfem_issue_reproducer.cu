// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "serac/serac_config.hpp"

int serial_refinement   = 1;
int parallel_refinement = 0;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

std::map<int, std::string> meshfiles = {{2, SERAC_REPO_DIR "/data/meshes/star.mesh"},
                                        {3, SERAC_REPO_DIR "/data/meshes/beam-hex.mesh"}};

void some_parametrized_test(int p, int dim)
{
  std::ifstream stream(meshfiles[dim]);
  mfem::Mesh    serial_mesh(stream, 1, 1, true);
  for (int i = 0; i < serial_refinement; i++) {
    serial_mesh.UniformRefinement();
  }

  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  for (int i = 0; i < parallel_refinement; i++) {
    mesh.UniformRefinement();
  }

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParBilinearForm A(&fespace);

  // Add the diffusion term using the standard MFEM method
  mfem::ConstantCoefficient coef(1.0);
  A.AddDomainIntegrator(new mfem::DiffusionIntegrator(coef));
  A.Assemble(0);
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  // Create a linear form for the load term using the standard MFEM method
  mfem::ParLinearForm       f(&fespace);
  mfem::FunctionCoefficient load_func([&](const mfem::Vector& coords) { return 100 * coords(0) * coords(1); });

  // Create and assemble the linear load term into a vector
  f.AddDomainIntegrator(new mfem::DomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());
  F->UseDevice(true);

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  U.UseDevice(true);
  u_global.GetTrueDofs(U);

  // Compute the residual using standard MFEM methods
  mfem::Vector r1(U.Size());
  J->Mult(U, r1);
  r1 -= *F;

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
}

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int               num_procs = mpi.WorldSize();
  int               myid      = mpi.WorldRank();

  mfem::Device device("cuda");

  int dim = 2;
  for (int p = 1; p <= 3; ++p) {
    some_parametrized_test(p, dim);
  }

  return 0;
}
