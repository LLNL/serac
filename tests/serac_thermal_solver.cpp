// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction.hpp"

#include <sys/stat.h>

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

using test_utils::InputFileTest;

double One(const mfem::Vector& /*x*/) { return 1.0; }
double BoundaryTemperature(const mfem::Vector& x) { return x.Norml2(); }
double OtherBoundaryTemperature(const mfem::Vector& x) { return 2 * x.Norml2(); }

double InitialTemperature(const mfem::Vector& x)
{
  if (x.Norml2() < 0.5) {
    return 2.0;
  } else {
    return 1.0;
  }
}

TEST(thermal_solver, static_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/static_solve.lua";
  auto pmesh = buildBallMesh(10000);
  test_utils::runModuleTest<ThermalConduction>(input_file_path, "static_solve", pmesh);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_P(InputFileTest, thermal_conduction)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/" + GetParam() + ".lua";
  test_utils::runModuleTest<ThermalConduction>(input_file_path, GetParam());
  MPI_Barrier(MPI_COMM_WORLD);
}

const std::string input_files[] = {"static_solve_multiple_bcs", "static_solve_repeated_bcs", "dyn_exp_solve",
                                   "dyn_imp_solve"};

INSTANTIATE_TEST_SUITE_P(ThermalConductionInputFileTests, InputFileTest, ::testing::ValuesIn(input_files));

TEST(thermal_solver, dyn_imp_solve_time_varying)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 2, 1);

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(2, pmesh, ThermalConduction::defaultDynamicOptions());

  // by construction, f(x, y, t) satisfies df_dt == d2f_dx2 + d2f_dy2
  auto f = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x, double t) {
    return 1.0 + 6.0 * x[0] * t - 2.0 * x[1] * t + (x[0] - x[1]) * x[0] * x[0];
  });

  therm_solver.setTemperature(*f);

  std::set<int> temp_bdr = {1};
  therm_solver.setTemperatureBCs(temp_bdr, f);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(1.0);
  therm_solver.setConductivity(std::move(kappa));

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::VisIt, "thermal_implicit");

  // Complete the setup including the dynamic operators
  therm_solver.completeSetup();

  // Set timestep options
  double t         = 0.0;
  double t_final   = 5.0;
  double dt        = 0.5;
  bool   last_step = false;

  // Output the initial state
  therm_solver.outputState();

  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    // Advance the timestep
    therm_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  therm_solver.outputState();

  f->SetTime(t);
  double error = therm_solver.temperature().gridFunc().ComputeLpError(2.0, *f);
  EXPECT_NEAR(0.0, error, 0.00005);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(thermal_solver, nonlinear_reaction)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 1);

  DirectSolverOptions lin_options = {};

  NonlinearSolverOptions nonlin_options = {.rel_tol = 1.0e-6, .abs_tol = 1.0e-12, .max_iter = 500, .print_level = 1};

  // Initialize the second order thermal solver on the parallel mesh
  ThermalConduction therm_solver(3, pmesh, {lin_options, nonlin_options});

  // Set the nonlinear reaction term
  auto reaction   = [](double u) { return u * u; };
  auto d_reaction = [](double u) { return 2.0 * u; };
  therm_solver.setNonlinearSource(reaction, d_reaction);

  // Define the exact solution
  auto exact = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) { return x[0] * x[0] * x[1]; });
  therm_solver.setTemperatureBCs({1}, exact);

  // Set the conductivity of the thermal operator
  auto kappa = std::make_unique<mfem::ConstantCoefficient>(1.0);
  therm_solver.setConductivity(std::move(kappa));

  // Define the source term that corresponds to the exact solution
  auto source = std::make_unique<mfem::FunctionCoefficient>(
      [](const mfem::Vector& x) { return std::pow(x[0], 4.0) * std::pow(x[1], 2.0) - 2.0 * x[1]; });

  therm_solver.setSource(std::move(source));

  // Setup glvis output
  therm_solver.initializeOutput(serac::OutputType::VisIt, "thermal_nonlinear_reaction");

  // Complete the setup including the dynamic operators
  therm_solver.completeSetup();

  double dt = 1.0;
  therm_solver.advanceTimestep(dt);

  // Output the final state
  therm_solver.outputState();

  double error = therm_solver.temperature().gridFunc().ComputeLpError(2.0, *exact);
  EXPECT_NEAR(0.0, error, 0.00005);

  MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef MFEM_USE_AMGX
TEST(thermal_solver, static_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);
  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/thermal_conduction/static_amgx_solve.lua";
  auto pmesh = buildBallMesh(10000);
  test_utils::runModuleTest<ThermalConduction>(input_file_path, pmesh);
  MPI_Barrier(MPI_COMM_WORLD);
}
#endif

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
