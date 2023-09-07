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
#include "serac/serac_config.hpp"

namespace serac {

constexpr int     dim            = 2;
constexpr int     p              = 1;
const std::string thermal_prefix = "thermal";

struct TimeSteppingInfo {
  double totalTime     = 0.6;
  int    num_timesteps = 4;
};

double computeStepQoi(const FiniteElementState& temperature, double dt)
{
  // Compute qoi: \int_t \int_omega 0.5 * (T - T_target(x,t)^2), T_target = 0 here
  return 0.5 * dt * innerProduct(temperature, temperature);
}

void computeStepAdjointLoad(const FiniteElementState& temperature, FiniteElementDual& d_qoi_d_temperature, double dt)
{
  d_qoi_d_temperature = temperature;
  d_qoi_d_temperature *= dt;
}

std::unique_ptr<HeatTransfer<p, dim>> create_heat_transfer(
    const NonlinearSolverOptions& nonlinear_opts, const TimesteppingOptions& dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat)
{
  // eventually figure out how to clear out cider state
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

double computeThermalQoiAdjustingInitalTemperature(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat, const TimeSteppingInfo& ts_info,
    const FiniteElementState& init_temp_derivative_direction, double pertubation)
{
  auto thermal = create_heat_transfer(nonlinear_opts, dyn_opts, mat);

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

double computeThermalQoiAdjustingShape(axom::sidre::DataStore& /*data_store*/,
                                       const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions&    dyn_opts,
                                       const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& shape_derivative_direction, double pertubation)
{
  auto thermal = create_heat_transfer(nonlinear_opts, dyn_opts, mat);

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

std::tuple<double, FiniteElementDual, FiniteElementDual> computeThermalQoiAndInitialTemperatureAndShapeGradient(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat, const TimeSteppingInfo& ts_info)
{
  auto thermal = create_heat_transfer(nonlinear_opts, dyn_opts, mat);

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  FiniteElementDual initial_temperature_gradient(thermal->temperature().space(), "init_temp_gradient");
  initial_temperature_gradient = 0.0;
  FiniteElementDual shape_gradient(thermal->shapeDisplacement().space(), "shape_gradient");
  shape_gradient = 0.0;
  FiniteElementDual adjoint_load(thermal->temperature().space(), "adjoint_load");

  for (int i = ts_info.num_timesteps; i > 0; --i) {
    double             dt                      = ts_info.totalTime / ts_info.num_timesteps;
    FiniteElementState temperature_end_of_step = thermal->previousTemperature(thermal->cycle());
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    thermal->reverseAdjointTimestep({{"temperature", adjoint_load}});
    shape_gradient += thermal->computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(0, thermal->cycle());  // we are back to the start
  auto initialConditionSensitivities     = thermal->computeInitialConditionSensitivity();
  auto initialTemperatureSensitivityIter = initialConditionSensitivities.find("temperature");
  SLIC_ASSERT_MSG(initialTemperatureSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find temperature in the computed initial condition sensitivities.");
  initial_temperature_gradient += initialTemperatureSensitivityIter->second;

  return std::make_tuple(qoi, initial_temperature_gradient, shape_gradient);
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
  TimesteppingOptions    dyn_opts{.timestepper        = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};
  heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature mat{1.0, 1.0, 1.0, 2.0};
  TimeSteppingInfo                                                     tsInfo{.totalTime = 0.5, .num_timesteps = 4};
};

TEST_F(HeatTransferSensitivityFixture, InitialTemperatureSensitivities)
{
  auto [qoi_base, temperature_gradient, _] =
      computeThermalQoiAndInitialTemperatureAndShapeGradient(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(temperature_gradient.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps      = 1e-7;
  double       qoi_plus = computeThermalQoiAdjustingInitalTemperature(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                derivative_direction, eps);
  double       directional_deriv = innerProduct(derivative_direction, temperature_gradient);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, ShapeSensitivities)
{
  auto [qoi_base, _, shape_gradient] =
      computeThermalQoiAndInitialTemperatureAndShapeGradient(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(shape_gradient.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;
  double       qoi_plus =
      computeThermalQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_gradient);
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
