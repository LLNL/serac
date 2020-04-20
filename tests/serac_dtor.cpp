// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <fstream>

#include "mfem.hpp"
#include "solvers/thermal_solver.hpp"

template <typename T>
T do_nothing(T foo)
{
  return foo;
}

const char *mesh_file = "NO_MESH_GIVEN";

inline bool file_exists(const char *path)
{
  struct stat buffer;
  return (stat(path, &buffer) == 0);
}

TEST(serac_dtor, test1)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  ASSERT_TRUE(file_exists(mesh_file));
  std::ifstream imesh(mesh_file);
  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // Refine in serial
  mesh->UniformRefinement();

  // Initialize the parallel mesh and delete the serial mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  // Refine the parallel mesh
  pmesh->UniformRefinement();

  // Initialize the second order thermal solver on the parallel mesh
  ThermalSolver therm_solver(2, pmesh);

  // Set the time integration method
  therm_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Initialize the temperature boundary condition
  auto u_0 = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector &x) { return x.Norml2(); });

  std::vector<int> temp_bdr(pmesh->bdr_attributes.Max(), 1);

  // Set the temperature BC in the thermal solver
  therm_solver.SetTemperatureBCs(temp_bdr, u_0);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_shared<mfem::ConstantCoefficient>(0.5);
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
}

int main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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

  result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
