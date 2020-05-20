// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "solvers/nonlinear_solid_solver.hpp"
#include "serac_config.hpp"

void InitialDeformation(const mfem::Vector &x, mfem::Vector &y);

void InitialVelocity(const mfem::Vector &x, mfem::Vector &v);

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // mesh
  std::string base_mesh_file = std::string(SERAC_SRC_DIR) + "/data/beam-hex.mesh";
  const char *mesh_file      = base_mesh_file.c_str();

  // Open the mesh
  std::ifstream imesh(mesh_file);
  auto          mesh = std::make_shared<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  mesh->UniformRefinement();

  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);

  int dim = pmesh->Dimension();

  // define a boundary attribute array and initialize to 0
  std::vector<int> ess_bdr(pmesh->bdr_attributes.Max(), 0);

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  auto visc = std::make_shared<mfem::ConstantCoefficient>(0.0);

  // define the inital state coefficients
  std::vector<std::shared_ptr<mfem::VectorCoefficient> > initialstate(2);

  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, InitialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, InitialVelocity);

  // initialize the dynamic solver object
  NonlinearSolidSolver dyn_solver(1, pmesh);
  dyn_solver.SetDisplacementBCs(ess_bdr, deform);
  dyn_solver.SetHyperelasticMaterialParameters(0.25, 5.0);
  dyn_solver.SetViscosity(visc);
  dyn_solver.SetDisplacement(*deform);
  dyn_solver.SetVelocity(*velo);
  dyn_solver.SetTimestepper(TimestepMethod::SDIRK33);

  // Set the linear solver parameters
  LinearSolverParameters params;
  params.prec        = Preconditioner::BoomerAMG;
  params.abs_tol     = 1.0e-8;
  params.rel_tol     = 1.0e-4;
  params.max_iter    = 500;
  params.lin_solver  = LinearSolver::GMRES;
  params.print_level = 0;

  // Set the nonlinear solver parameters
  NonlinearSolverParameters nl_params;
  nl_params.rel_tol     = 1.0e-4;
  nl_params.abs_tol     = 1.0e-8;
  nl_params.print_level = 1;
  nl_params.max_iter    = 500;
  dyn_solver.SetSolverParameters(params, nl_params);

  // Initialize the VisIt output
  dyn_solver.InitializeOutput(OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.CompleteSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 3.0;

  // Ouput the initial state
  dyn_solver.OutputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.AdvanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.OutputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  auto state = dyn_solver.GetState();

  double v_norm = state[0].gf->ComputeLpError(2.0, zerovec);
  double x_norm = state[1].gf->ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(12.86733, x_norm, 0.0001);
  EXPECT_NEAR(0.22298, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

void InitialDeformation(const mfem::Vector &x, mfem::Vector &y)
{
  // set the initial configuration to be the same as the reference, stress
  // free, configuration
  y = x;
}

void InitialVelocity(const mfem::Vector &x, mfem::Vector &v)
{
  const int    dim = x.Size();
  const double s   = 0.1 / 64.;

  v          = 0.0;
  v(dim - 1) = s * x(0) * x(0) * (8.0 - x(0));
  v(0)       = -s * x(0) * x(0);
}
