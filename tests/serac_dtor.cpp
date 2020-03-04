// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "solvers/dynamic_solver.hpp"
#include <fstream>

template < typename T >
T do_nothing(T foo) {
  return foo;
}

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  MPI_Barrier(MPI_COMM_WORLD);

  std::ifstream imesh("../../data/beam-hex.mesh");
  mfem::Mesh* mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = NULL;
  mesh->UniformRefinement();

  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  int dim = pmesh->Dimension();

  // Define the finite element spaces for displacement field
  mfem::H1_FECollection fe_coll(1, dim);
  mfem::ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim, mfem::Ordering::byVDIM);

  int true_size = fe_space.TrueVSize();
  mfem::Array<int> true_offset(3);
  true_offset[0] = 0;
  true_offset[1] = true_size;
  true_offset[2] = 2*true_size;

  // define a boundary attribute array and initialize to 0
  mfem::Array<int> ess_bdr;
  ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
  ess_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  mfem::ConstantCoefficient visc(0.0);

  // construct the nonlinear mechanics operator
  DynamicSolver oper(fe_space, ess_bdr,
                     0.25, 5.0, visc,
                     1.0e-4, 1.0e-8,
                     500, true, false);

  do_nothing(oper); 

  do_nothing(oper); 

  do_nothing(oper); 

  do_nothing(oper); 

  MPI_Finalize();

}
