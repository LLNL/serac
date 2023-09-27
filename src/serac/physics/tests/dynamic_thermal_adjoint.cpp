// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include "serac/physics/heat_transfer.hpp"

#include <functional>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/parameterized_thermal_material.hpp"
#include "serac/serac_config.hpp"

namespace serac {

constexpr int dim = 2;
constexpr int p   = 1;

const std::string thermal_prefix              = "thermal";
const std::string parametrized_thermal_prefix = "thermal_with_param";

struct TimeSteppingInfo {
  double totalTime     = 0.6;
  int    num_timesteps = 4;
};

double computeStepQoi(const FiniteElementState& temperature, double dt)
{
  return 0.5 * dt * innerProduct(temperature, temperature);
}

void computeStepAdjointLoad(const FiniteElementState& temperature, FiniteElementDual& d_qoi_d_temperature, double dt)
{
  d_qoi_d_temperature = temperature;
  d_qoi_d_temperature *= dt;
}

std::unique_ptr<HeatTransfer<p, dim>> create_nonlinear_heat_transfer(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat)
{
  // eventually figure out how to clear out sidre state
  // auto saveMesh = std::make_unique<mfem::ParMesh>(StateManager::mesh());
  // StateManager::reset();
  // static int iter = 0;
  // StateManager::initialize(data_store, "thermal_dynamic_solve"+std::to_string(iter++));
  // std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
  // mfem::ParMesh* mesh = StateManager::setMesh(std::move(saveMesh));
  static int iter = 0;
  auto thermal = std::make_unique<HeatTransfer<p, dim>>(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts,
                                                        thermal_prefix + std::to_string(iter++));
  thermal->setMaterial(mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();
  return thermal;
}

using ParametrizedHeatTransferT = HeatTransfer<p, dim, Parameters<H1<p>>, std::integer_sequence<int, 0>>;

std::unique_ptr<ParametrizedHeatTransferT> create_parameterized_heat_transfer(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions& dyn_opts, const heat_transfer::ParameterizedLinearIsotropicConductor& mat)
{
  // eventually figure out how to clear out sidre state
  // auto saveMesh = std::make_unique<mfem::ParMesh>(StateManager::mesh());
  // StateManager::reset();
  // static int iter = 0;
  // StateManager::initialize(data_store, "thermal_dynamic_solve"+std::to_string(iter++));
  // std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
  // mfem::ParMesh* mesh = StateManager::setMesh(std::move(saveMesh));
  static int iter = 0;
  auto       thermal =
      std::make_unique<ParametrizedHeatTransferT>(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts,
                                                  parametrized_thermal_prefix + std::to_string(iter++));
  auto user_defined_conductivity_ptr = thermal->generateParameter("pcond", 0);
  *user_defined_conductivity_ptr     = 1.1;
  thermal->setMaterial(DependsOn<0>{}, mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();
  return thermal;  // std::make_pair(thermal, user_defined_conductivity);
}

double computeThermalQoiAdjustingInitalTemperature(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat, const TimeSteppingInfo& ts_info,
    const FiniteElementState& init_temp_derivative_direction, double pertubation)
{
  auto thermal = create_nonlinear_heat_transfer(data_store, nonlinear_opts, dyn_opts, mat);

  auto& temperature = thermal->temperature();
  SLIC_ASSERT_MSG(temperature.Size() == init_temp_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  temperature.Add(pertubation, init_temp_derivative_direction);

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }
  return qoi;
}

double computeThermalQoiAdjustingShape(axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& shape_derivative_direction, double pertubation)
{
  auto thermal = create_nonlinear_heat_transfer(data_store, nonlinear_opts, dyn_opts, mat);

  auto& shapeDisp = thermal->shapeDisplacement();
  SLIC_ASSERT_MSG(shapeDisp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shapeDisp.Add(pertubation, shape_derivative_direction);

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }
  return qoi;
}

std::tuple<double, FiniteElementDual, FiniteElementDual> computeThermalQoiAndInitialTemperatureAndShapeSensitivity(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat, const TimeSteppingInfo& ts_info)
{
  auto thermal = create_nonlinear_heat_transfer(data_store, nonlinear_opts, dyn_opts, mat);

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }

  FiniteElementDual initial_temperature_sensitivity(thermal->temperature().space(), "init_temp_sensitivity");
  initial_temperature_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(thermal->shapeDisplacement().space(), "shape_sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(thermal->temperature().space(), "adjoint_load");

  for (int i = ts_info.num_timesteps; i > 0; --i) {
    double             dt                      = ts_info.totalTime / ts_info.num_timesteps;
    FiniteElementState temperature_end_of_step = thermal->loadCheckpointedTemperature(thermal->cycle());
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    thermal->reverseAdjointTimestep({{"temperature", adjoint_load}});
    shape_sensitivity += thermal->computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(0, thermal->cycle());  // we are back to the start
  auto initialConditionSensitivities     = thermal->computeInitialConditionSensitivity();
  auto initialTemperatureSensitivityIter = initialConditionSensitivities.find("temperature");
  SLIC_ASSERT_MSG(initialTemperatureSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find temperature in the computed initial condition sensitivities.");
  initial_temperature_sensitivity += initialTemperatureSensitivityIter->second;

  return std::make_tuple(qoi, initial_temperature_sensitivity, shape_sensitivity);
}

std::tuple<double, FiniteElementDual, FiniteElementDual> computeThermalConductivitySensitivity(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions& dyn_opts, const heat_transfer::ParameterizedLinearIsotropicConductor& mat,
    const TimeSteppingInfo& ts_info)
{
  auto thermal = create_parameterized_heat_transfer(data_store, nonlinear_opts, dyn_opts, mat);

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }

  FiniteElementDual initial_temperature_sensitivity(thermal->temperature().space(), "init_temp_sensitivity");
  initial_temperature_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(thermal->shapeDisplacement().space(), "shape_sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(thermal->temperature().space(), "adjoint_load");

  for (int i = ts_info.num_timesteps; i > 0; --i) {
    double             dt                      = ts_info.totalTime / ts_info.num_timesteps;
    FiniteElementState temperature_end_of_step = thermal->loadCheckpointedTemperature(thermal->cycle());
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    thermal->reverseAdjointTimestep({{"temperature", adjoint_load}});
    shape_sensitivity += thermal->computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(0, thermal->cycle());  // we are back to the start
  auto initialConditionSensitivities     = thermal->computeInitialConditionSensitivity();
  auto initialTemperatureSensitivityIter = initialConditionSensitivities.find("temperature");
  SLIC_ASSERT_MSG(initialTemperatureSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find temperature in the computed initial condition sensitivities.");
  initial_temperature_sensitivity += initialTemperatureSensitivityIter->second;

  return std::make_tuple(qoi, initial_temperature_sensitivity, shape_sensitivity);
}

struct HeatTransferSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "thermal_dynamic_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
    mesh                 = StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0));
  }

  void fillDirection(FiniteElementState& direction) const { direction = 1.1; }

  // Create DataStore
  axom::sidre::DataStore dataStore;
  mfem::ParMesh*         mesh;

  // Solver options
  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};

  heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature mat{1.0, 1.0, 1.0, 2.0};
  heat_transfer::ParameterizedLinearIsotropicConductor                 parameterizedMat;

  TimeSteppingInfo tsInfo{.totalTime = 0.5, .num_timesteps = 4};
};

TEST_F(HeatTransferSensitivityFixture, InitialTemperatureSensitivities)
{
  auto [qoi_base, temperature_sensitivity, _] =
      computeThermalQoiAndInitialTemperatureAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(temperature_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps      = 1e-7;
  double       qoi_plus = computeThermalQoiAdjustingInitalTemperature(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                derivative_direction, eps);
  double       directional_deriv = innerProduct(derivative_direction, temperature_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, ShapeSensitivities)
{
  auto [qoi_base, _, shape_sensitivity] =
      computeThermalQoiAndInitialTemperatureAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus =
      computeThermalQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, DISABLED_ConductivityParameterSensitivities)
{
  auto [qoi_base, _, shape_sensitivity] =
      computeThermalConductivitySensitivity(dataStore, nonlinear_opts, dyn_opts, parameterizedMat, tsInfo);

  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus =
      computeThermalQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
