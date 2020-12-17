// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "serac/infrastructure/input.hpp"
#include "mfem.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/solid.hpp"
#include "serac/serac_config.hpp"
#include "test_utilities.hpp"

namespace serac {

void initialDeformation(const mfem::Vector& x, mfem::Vector& y);

void initialVelocity(const mfem::Vector& x, mfem::Vector& v);

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet, /* dynamic = */ true);

  // Build the mesh
  auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file_path);
  auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);

  // Define the solid solver object
  auto           solid_solver_options = inlet["nonlinear_solid"].get<serac::Solid::InputOptions>();
  Solid dyn_solver(mesh, solid_solver_options);

  int dim = mesh->Dimension();

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_options.boundary_conditions) {
    if (bc.name == "displacement") {
      dyn_solver.setDisplacementBCs(bc.attrs, deform);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // initialize the dynamic solver object
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(dynamic_solver, dyn_direct_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_direct_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet, true);

  // Build the mesh
  auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file_path);
  auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);

  // Define the solid solver object
  auto solid_solver_options = inlet["nonlinear_solid"].get<serac::Solid::InputOptions>();
  // FIXME: These should be moved to part of the schema once the contains() logic is updated in Inlet
  solid_solver_options.solver_options.H_lin_options = DirectSolverOptions{0};
  Solid dyn_solver(mesh, solid_solver_options);

  int dim = mesh->Dimension();

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_options.boundary_conditions) {
    if (bc.name == "displacement") {
      dyn_solver.setDisplacementBCs(bc.attrs, deform);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // initialize the dynamic solver object
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef MFEM_USE_SUNDIALS
TEST(dynamic_solver, dyn_linesearch_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path = std::string(SERAC_REPO_DIR) +
                                "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_linesearch_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet, true);

  // Build the mesh
  auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file_path);
  auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);

  // Define the solid solver object
  auto           solid_solver_options = inlet["nonlinear_solid"].get<serac::Solid::InputOptions>();
  Solid dyn_solver(mesh, solid_solver_options);

  int dim = mesh->Dimension();

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_options.boundary_conditions) {
    if (bc.name == "displacement") {
      dyn_solver.setDisplacementBCs(bc.attrs, deform);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // initialize the dynamic solver object
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_SUNDIALS

#ifdef MFEM_USE_AMGX
TEST(dynamic_solver, dyn_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string input_file_path =
      std::string(SERAC_REPO_DIR) + "/data/input_files/tests/nonlinear_solid/serac_dynamic_solver/dyn_amgx_solve.lua";

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);

  testing::defineNonlinSolidInputFileSchema(inlet, /* dynamic = */ true);

  // Build the mesh
  auto mesh_options   = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_options.relative_mesh_file_name, input_file_path);
  auto mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);

  // Define the solid solver object
  auto           solid_solver_options = inlet["nonlinear_solid"].get<serac::Solid::InputOptions>();
  Solid dyn_solver(mesh, solid_solver_options);

  int dim = mesh->Dimension();

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // Pass the BC information to the solver object setting only the z direction
  for (const auto& bc : solid_solver_options.boundary_conditions) {
    if (bc.name == "displacement") {
      dyn_solver.setDisplacementBCs(bc.attrs, deform);
    } else {
      SLIC_WARNING("Ignoring unrecognized boundary condition: " << bc.name);
    }
  }

  // initialize the dynamic solver object
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = inlet["t_final"];
  double dt      = inlet["dt"];

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_AMGX

void initialDeformation(const mfem::Vector& /*x*/, mfem::Vector& u) { u = 0.0; }

void initialVelocity(const mfem::Vector& x, mfem::Vector& v)
{
  const int    dim = x.Size();
  const double s   = 0.1 / 64.;

  v          = 0.0;
  v(dim - 1) = s * x(0) * x(0) * (8.0 - x(0));
  v(0)       = -s * x(0) * x(0);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
