// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <sys/stat.h>

#include "mfem.hpp"
#include "solvers/thermal_solver.hpp"

template <typename T>
T do_nothing(T foo)
{
  return foo;
}

inline bool file_exists(const char *path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  MPI_Barrier(MPI_COMM_WORLD);

  const char *mesh_file = "NO_MESH_GIVEN";
  // Parse command line options
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.", true);
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  std::ifstream imesh("../../data/beam-hex.mesh");
  mfem::Mesh *  mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = nullptr;

  // Initialize the parallel mesh and delete the serial mesh
  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  delete mesh;

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  mfem::FunctionCoefficient u_0([](const mfem::Vector &x) { return x.Norml2(); });

  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(temp_bdr, &u_0);

  // Set the conductivity of the thermal operator
  mfem::ConstantCoefficient kappa(0.5);
  therm_solver.SetConductivity(kappa);

  // Define the linear solver params
  LinearSolverParameters params;
  params.rel_tol     = 1.0e-6;
  params.abs_tol     = 1.0e-12;
  params.print_level = 0;
  params.max_iter    = 100;
  therm_solver.SetLinearSolverParameters(params);

  // Complete the setup without allocating the mass matrices and dynamic
  // operator
  therm_solver.CompleteSetup();

  // just do something to make sure the dtor is being called
  // and that the member variables lifetime is managed properly
  do_nothing(therm_solver);
  do_nothing(therm_solver);
  do_nothing(therm_solver);
  do_nothing(therm_solver);

  MPI_Finalize();
}
