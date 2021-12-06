// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/physics/coefficients/coefficient_extensions.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

namespace serac {

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "dynamic_solve");

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto      pmesh = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);
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
  ThermalSolid ts_solver(1, therm_options, default_dynamic, "coupled");
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

  auto temp_gf_coef = ts_solver.temperature().gridFuncCoef();
  auto visc_coef    = std::make_unique<mfem_ext::TransformedScalarCoefficient<mfem::Coefficient>>(
      [offset, scale](double& x) -> double { return scale * x + offset; }, temp_gf_coef);
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

  EXPECT_NEAR(0.122796, norm(ts_solver.displacement()), 0.001);
  EXPECT_NEAR(0.001791, norm(ts_solver.velocity()), 0.001);
  EXPECT_NEAR(6.494477, norm(ts_solver.temperature()), 0.001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(dynamic_solver, dyn_solve_restart)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Hardcoded for simplicity
  constexpr int dim = 3;

  // define a boundary attribute set
  std::set<int> ess_bdr = {1};

  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, [](const mfem::Vector& x, mfem::Vector& y) {
    y    = 0.0;
    y(1) = x(0) * 0.01;
  });

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

  // Used for temperature-dependent viscosity
  const double offset = 0.1;
  const double scale  = 1.0;

  const std::string primary_datacoll_name   = "primary";
  const std::string secondary_datacoll_name = "secondary";

  double       t_primary   = 0.0;
  double       t_secondary = 0.0;
  const double t_final     = 6.0;
  const double dt          = 1.0;

  int primary_reload_cycle   = 0;
  int secondary_reload_cycle = 0;

  // First create two solvers on two different meshes, running one for 3 steps and the other for 2
  {
    // Create DataStore
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "dynamic_solve_restart");

    // Open the mesh
    std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

    auto velo =
        std::make_shared<mfem::VectorFunctionCoefficient>(dim, [](const mfem::Vector&, mfem::Vector& v) { v = 0.0; });

    auto temp = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector& x) {
      double t = 2.0;
      if (x(0) < 1.0) {
        t = 5.0;
      }
      return t;
    });

    // The meshes are identical but they're treated as primary and secondary
    auto pmesh_primary = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);
    SLIC_ERROR_ROOT_IF(pmesh_primary->Dimension() != dim, "Dimensions do not match");
    auto pmesh_primary_ptr = serac::StateManager::setMesh(std::move(pmesh_primary), primary_datacoll_name);

    // initialize the dynamic solver object
    ThermalSolid ts_solver_primary(1, therm_options, default_dynamic, "coupled_primary", pmesh_primary_ptr);
    ts_solver_primary.setDisplacementBCs(ess_bdr, deform);
    ts_solver_primary.setTractionBCs(trac_bdr, traction_coef, false);
    ts_solver_primary.setSolidMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                                 std::make_unique<mfem::ConstantCoefficient>(5.0));
    ts_solver_primary.setConductivity(std::make_unique<mfem::ConstantCoefficient>(0.5));
    ts_solver_primary.setDisplacement(*deform);
    ts_solver_primary.setVelocity(*velo);
    ts_solver_primary.setTemperature(*temp);
    ts_solver_primary.setCouplingScheme(serac::CouplingScheme::OperatorSplit);

    auto temp_gf_coef_primary = ts_solver_primary.temperature().gridFuncCoef();
    auto visc_coef_primary    = std::make_unique<mfem_ext::TransformedScalarCoefficient<mfem::Coefficient>>(
        [offset, scale](double& x) -> double { return scale * x + offset; }, temp_gf_coef_primary);
    ts_solver_primary.setViscosity(std::move(visc_coef_primary));

    ts_solver_primary.initializeOutput(serac::OutputType::SidreVisIt, "");

    // Construct the internal dynamic solver data structures
    ts_solver_primary.completeSetup();

    // Ouput the initial state
    ts_solver_primary.outputState();

    auto pmesh_secondary     = mesh::refineAndDistribute(buildMeshFromFile(mesh_file), 1, 0);
    auto pmesh_secondary_ptr = serac::StateManager::setMesh(std::move(pmesh_secondary), secondary_datacoll_name);

    // initialize the dynamic solver object
    ThermalSolid ts_solver_secondary(1, therm_options, default_dynamic, "coupled_secondary", pmesh_secondary_ptr);
    ts_solver_secondary.setDisplacementBCs(ess_bdr, deform);
    ts_solver_secondary.setTractionBCs(trac_bdr, traction_coef, false);
    ts_solver_secondary.setSolidMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                                   std::make_unique<mfem::ConstantCoefficient>(5.0));
    ts_solver_secondary.setConductivity(std::make_unique<mfem::ConstantCoefficient>(0.5));
    ts_solver_secondary.setDisplacement(*deform);
    ts_solver_secondary.setVelocity(*velo);
    ts_solver_secondary.setTemperature(*temp);
    ts_solver_secondary.setCouplingScheme(serac::CouplingScheme::OperatorSplit);

    auto temp_gf_coef_secondary = ts_solver_secondary.temperature().gridFuncCoef();
    auto visc_coef_secondary    = std::make_unique<mfem_ext::TransformedScalarCoefficient<mfem::Coefficient>>(
        [offset, scale](double& x) -> double { return scale * x + offset; }, temp_gf_coef_secondary);
    ts_solver_secondary.setViscosity(std::move(visc_coef_secondary));

    ts_solver_secondary.initializeOutput(serac::OutputType::SidreVisIt, "");

    // Construct the internal dynamic solver data structures
    ts_solver_secondary.completeSetup();

    // Ouput the initial state
    ts_solver_secondary.outputState();

    double t_intermediate_primary = 3.0;
    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_intermediate_primary - t_primary);
      t_primary += dt_real;
      last_step = (t_primary >= t_intermediate_primary - 1e-8 * dt);

      ts_solver_primary.advanceTimestep(dt_real);
      primary_reload_cycle = ti;
    }

    // Output the final state
    ts_solver_primary.outputState();

    double t_intermediate_secondary = 2.0;
    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_intermediate_secondary - t_secondary);
      t_secondary += dt_real;
      last_step = (t_secondary >= t_intermediate_secondary - 1e-8 * dt);

      ts_solver_secondary.advanceTimestep(dt_real);
      secondary_reload_cycle = ti;
    }

    // Output the final state
    ts_solver_secondary.outputState();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Then restart the simulation, running each to a total of six cycles
  {
    // Create DataStore
    axom::sidre::DataStore datastore;
    serac::StateManager::initialize(datastore, "dynamic_solve_restart");
    serac::StateManager::load(primary_reload_cycle, primary_datacoll_name);
    serac::StateManager::load(secondary_reload_cycle, secondary_datacoll_name);

    // What's awkward about using a mesh as the param is that we have to then extract the mesh
    // instead of using the string ID we just used in the call to load()
    // Note that just the 'auto' will not compile but only because ParMesh move ctor is explicitly deleted
    auto& primary_mesh   = serac::StateManager::mesh(primary_datacoll_name);
    auto& secondary_mesh = serac::StateManager::mesh(secondary_datacoll_name);

    // initialize the dynamic solver object
    ThermalSolid ts_solver_primary(1, therm_options, default_dynamic, "coupled_primary", &primary_mesh);
    ts_solver_primary.setDisplacementBCs(ess_bdr, deform);
    ts_solver_primary.setTractionBCs(trac_bdr, traction_coef, false);
    ts_solver_primary.setSolidMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                                 std::make_unique<mfem::ConstantCoefficient>(5.0));
    ts_solver_primary.setConductivity(std::make_unique<mfem::ConstantCoefficient>(0.5));
    ts_solver_primary.setCouplingScheme(serac::CouplingScheme::OperatorSplit);

    auto temp_gf_coef_primary = ts_solver_primary.temperature().gridFuncCoef();
    auto visc_coef_primary    = std::make_unique<mfem_ext::TransformedScalarCoefficient<mfem::Coefficient>>(
        [offset, scale](double& x) -> double { return scale * x + offset; }, temp_gf_coef_primary);
    ts_solver_primary.setViscosity(std::move(visc_coef_primary));

    ts_solver_primary.initializeOutput(serac::OutputType::SidreVisIt, "");

    // Construct the internal dynamic solver data structures
    ts_solver_primary.completeSetup();

    // Ouput the initial state
    ts_solver_primary.outputState();

    // initialize the dynamic solver object
    ThermalSolid ts_solver_secondary(1, therm_options, default_dynamic, "coupled_secondary", &secondary_mesh);
    ts_solver_secondary.setDisplacementBCs(ess_bdr, deform);
    ts_solver_secondary.setTractionBCs(trac_bdr, traction_coef, false);
    ts_solver_secondary.setSolidMaterialParameters(std::make_unique<mfem::ConstantCoefficient>(0.25),
                                                   std::make_unique<mfem::ConstantCoefficient>(5.0));
    ts_solver_secondary.setConductivity(std::make_unique<mfem::ConstantCoefficient>(0.5));
    ts_solver_secondary.setCouplingScheme(serac::CouplingScheme::OperatorSplit);

    auto temp_gf_coef_secondary = ts_solver_secondary.temperature().gridFuncCoef();
    auto visc_coef_secondary    = std::make_unique<mfem_ext::TransformedScalarCoefficient<mfem::Coefficient>>(
        [offset, scale](double& x) -> double { return scale * x + offset; }, temp_gf_coef_secondary);
    ts_solver_secondary.setViscosity(std::move(visc_coef_secondary));

    ts_solver_secondary.initializeOutput(serac::OutputType::SidreVisIt, "");

    // Construct the internal dynamic solver data structures
    ts_solver_secondary.completeSetup();

    // Ouput the initial state
    ts_solver_secondary.outputState();

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    bool last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t_primary);
      t_primary += dt_real;
      last_step = (t_primary >= t_final - 1e-8 * dt);

      ts_solver_primary.advanceTimestep(dt_real);
    }

    // Output the final state
    ts_solver_primary.outputState();

    // Perform time-integration
    // (looping over the time iterations, ti, with a time-step dt).
    last_step = false;
    for (int ti = 1; !last_step; ti++) {
      double dt_real = std::min(dt, t_final - t_secondary);
      t_secondary += dt_real;
      last_step = (t_secondary >= t_final - 1e-8 * dt);

      ts_solver_secondary.advanceTimestep(dt_real);
    }

    // Output the final state
    ts_solver_secondary.outputState();

    EXPECT_NEAR(0.122796, norm(ts_solver_primary.displacement()), 0.001);
    EXPECT_NEAR(0.001791, norm(ts_solver_primary.velocity()), 0.001);
    EXPECT_NEAR(6.494477, norm(ts_solver_primary.temperature()), 0.001);

    EXPECT_NEAR(0.122796, norm(ts_solver_secondary.displacement()), 0.001);
    EXPECT_NEAR(0.001791, norm(ts_solver_secondary.velocity()), 0.001);
    EXPECT_NEAR(6.494477, norm(ts_solver_secondary.temperature()), 0.001);
  }

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
