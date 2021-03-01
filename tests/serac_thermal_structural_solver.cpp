// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto      pmesh = buildMeshFromFile(mesh_file, 1, 0);
  const int dim   = pmesh->Dimension();
  serac::StateManager::setMesh(std::move(pmesh));

  // define a boundary attribute set
  std::set<int> ess_bdr = {1};

  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, [](const mfem::Vector& x, mfem::Vector& y) {
    y    = 0.0;
    y(1) = x(0) * 0.01;
  });

  auto velo =
      std::make_shared<mfem::VectorFunctionCoefficient>(dim, [](const mfem::Vector&, mfem::Vector& v) { v = 0.0; });

  auto temp = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) {
    double t = 2.0;
    if (x(0) < 1.0) {
      t = 5.0;
    }
    return t;
  });

  auto kappa = std::make_unique<mfem::ConstantCoefficient>(0.5);

  // set the traction boundary
  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(dim);
  traction           = 0.0;
  traction(1)        = 1.0e-3;
  auto traction_coef = std::make_shared<mfem::VectorConstantCoefficient>(traction);

  // Use the same configuration as the solid solver
  const IterativeSolverOptions default_dyn_linear_options = {.rel_tol     = 1.0e-4,
                                                             .abs_tol     = 1.0e-8,
                                                             .print_level = 0,
                                                             .max_iter    = 500,
                                                             .lin_solver  = LinearSolver::GMRES,
                                                             .prec        = HypreBoomerAMGPrec{}};

  auto therm_M_options = default_dyn_linear_options;
  auto therm_T_options = default_dyn_linear_options;
  therm_M_options.prec = HypreSmootherPrec{};
  therm_T_options.prec = HypreSmootherPrec{};

  auto therm_options = ThermalConduction::defaultDynamicOptions();

  const NonlinearSolverOptions default_dyn_nonlinear_options = {
      .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

  const Solid::SolverOptions default_dynamic = {
      default_dyn_linear_options, default_dyn_nonlinear_options,
      Solid::TimesteppingOptions{TimestepMethod::AverageAcceleration, DirichletEnforcementMethod::RateControl}};

  // initialize the dynamic solver object
  ThermalSolid ts_solver(1, therm_options, default_dynamic);
  ts_solver.setDisplacementBCs(ess_bdr, deform);
  ts_solver.setTractionBCs(trac_bdr, traction_coef, false);
  ts_solver.setSolidMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                       std::make_unique<mfem::ConstantCoefficient>(5.0));
  ts_solver.setConductivity(std::move(kappa));
  ts_solver.setDisplacement(*deform);
  ts_solver.setVelocity(*velo);
  ts_solver.setTemperature(*temp);
  ts_solver.setCouplingScheme(serac::CouplingScheme::OperatorSplit);

  // Make a temperature-dependent viscosity
  double offset = 0.1;
  double scale  = 1.0;

  auto temp_gf_coef = std::make_shared<mfem::GridFunctionCoefficient>(&ts_solver.temperature().gridFunc());
  auto visc_coef    = std::make_unique<mfem_ext::TransformedScalarCoefficient>(
      temp_gf_coef, [offset, scale](const double x) { return scale * x + offset; });
  ts_solver.setViscosity(std::move(visc_coef));

  // Initialize the VisIt output
  ts_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_thermal_solid");

  // Construct the internal dynamic solver data structures
  ts_solver.completeSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 1.0;

  // Ouput the initial state
  ts_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    ts_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  ts_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm    = ts_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm    = ts_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);
  double temp_norm = ts_solver.temperature().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(0.122796, x_norm, 0.001);
  EXPECT_NEAR(0.001791, v_norm, 0.001);
  EXPECT_NEAR(6.494477, temp_norm, 0.001);

  MPI_Barrier(MPI_COMM_WORLD);
}

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
