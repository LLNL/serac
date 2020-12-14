// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_utilities.hpp"

#include <gtest/gtest.h>

#include "serac/infrastructure/input.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/nonlinear_solid.hpp"

namespace serac {

namespace testing {

void defineNonlinSolidInputFileSchema(axom::inlet::Inlet& inlet)
{
  // Simulation time parameters
  inlet.addDouble("dt", "Time step.");
  inlet.addDouble("t_final", "Stopping point");

  // Integration test parameters
  inlet.addDouble("expected_x_l2norm", "Correct L2 norm of the displacement field");
  inlet.addDouble("expected_v_l2norm", "Correct L2 norm of the velocity field");
  inlet.addDouble("epsilon", "Threshold to be used in the comparison");

  auto& mesh_table = inlet.addTable("main_mesh", "The main mesh for the problem");
  serac::mesh::InputInfo::defineInputFileSchema(mesh_table);

  // Physics
  auto& solid_solver_table = inlet.addTable("nonlinear_solid", "Finite deformation solid mechanics module");
  // FIXME: Remove once Inlet's "contains" logic improvements are merged
  serac::NonlinearSolid::InputInfo::defineInputFileSchema(solid_solver_table);

  // Verify input file
  if (!inlet.verify()) {
    SLIC_ERROR("Input file failed to verify.");
  }
}

void runNonlinSolidDynamicTest(const std::string& input_file)
{
  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file);

  testing::defineNonlinSolidInputFileSchema(inlet);

  // Build the mesh
  auto mesh_info      = inlet["main_mesh"].get<serac::mesh::InputInfo>();
  auto full_mesh_path = serac::input::findMeshFilePath(mesh_info.relative_mesh_file_name, input_file);
  auto mesh           = serac::buildMeshFromFile(full_mesh_path, mesh_info.ser_ref_levels, mesh_info.par_ref_levels);

  // Define the solid solver object
  auto           solid_solver_info = inlet["nonlinear_solid"].get<serac::NonlinearSolid::InputInfo>();
  NonlinearSolid dyn_solver(mesh, solid_solver_info);

  // initialize the dynamic solver object
  auto visc = std::make_unique<mfem::ConstantCoefficient>(0.0);
  dyn_solver.setViscosity(std::move(visc));

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
  int          dim = mesh->Dimension();
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(inlet["expected_x_l2norm"], x_norm, inlet["epsilon"]);
  EXPECT_NEAR(inlet["expected_v_l2norm"], v_norm, inlet["epsilon"]);
}

}  // end namespace testing

}  // end namespace serac
