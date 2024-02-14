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

const std::string mesh_tag                    = "mesh";
const std::string thermal_prefix              = "thermal";
const std::string parametrized_thermal_prefix = "thermal_with_param";

struct TimeSteppingInfo {
  TimeSteppingInfo() : dts({0.2, 0.4, 0.24, 0.12, 0.31}) {}

  int numTimesteps() const { return dts.Size(); }

  mfem::Vector dts;
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

static int iter = 0;

std::unique_ptr<HeatTransfer<p, dim>> createNonlinearHeatTransfer(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat)
{
  auto thermal = std::make_unique<HeatTransfer<p, dim>>(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts,
                                                        thermal_prefix + std::to_string(iter++), mesh_tag);
  thermal->setMaterial(mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();
  return thermal;
}

using ParametrizedHeatTransferT = HeatTransfer<p, dim, Parameters<H1<p>>, std::integer_sequence<int, 0>>;

std::unique_ptr<ParametrizedHeatTransferT> createParameterizedHeatTransfer(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions& dyn_opts, const heat_transfer::ParameterizedLinearIsotropicConductor& mat)
{
  std::vector<std::string> names{"conductivity"};

  auto thermal = std::make_unique<ParametrizedHeatTransferT>(
      nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts,
      parametrized_thermal_prefix + std::to_string(iter++), mesh_tag, names);

  FiniteElementState user_defined_conductivity(StateManager::mesh(mesh_tag), H1<p>{}, "user_defined_conductivity");
  user_defined_conductivity = 1.1;
  thermal->setParameter(0, user_defined_conductivity);
  thermal->setMaterial(DependsOn<0>{}, mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; },
                     EntireDomain(StateManager::mesh(mesh_tag)));
  thermal->completeSetup();
  return thermal;
}

std::unique_ptr<ParametrizedHeatTransferT> createParameterizedNonlinearHeatTransfer(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                               dyn_opts,
    const heat_transfer::ParameterizedIsotropicConductorWithLinearConductivityVsTemperature& mat)
{
  std::vector<std::string> names{"conductivity"};

  auto thermal = std::make_unique<ParametrizedHeatTransferT>(
      nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts,
      parametrized_thermal_prefix + std::to_string(iter++), mesh_tag, names);

  FiniteElementState user_defined_conductivity(StateManager::mesh(mesh_tag), H1<p>{}, "user_defined_conductivity");
  user_defined_conductivity = 1.1;
  thermal->setParameter(0, user_defined_conductivity);
  thermal->setMaterial(DependsOn<0>{}, mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();
  return thermal;
}

double computeThermalQoi(BasePhysics& physics_solver, const TimeSteppingInfo& ts_info)
{
  double qoi = 0.0;
  physics_solver.outputStateToDisk();
  for (int i = 0; i < ts_info.numTimesteps(); ++i) {
    double dt = ts_info.dts[i];
    physics_solver.advanceTimestep(dt);
    physics_solver.outputStateToDisk();
    qoi += computeStepQoi(physics_solver.state("temperature"), dt);
  }
  return qoi;
}

double computeThermalQoiAdjustingInitalTemperature(BasePhysics& solver, const TimeSteppingInfo& ts_info,
                                                   const FiniteElementState& init_temp_derivative_direction,
                                                   double                    pertubation)
{
  FiniteElementState initial_temp(solver.state("temperature"));
  SLIC_ASSERT_MSG(initial_temp.Size() == init_temp_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  initial_temp.Add(pertubation, init_temp_derivative_direction);
  solver.setState("temperature", initial_temp);

  return computeThermalQoi(solver, ts_info);
}

double computeThermalQoiAdjustingShape(BasePhysics& solver, const TimeSteppingInfo& ts_info,
                                       const FiniteElementState& shape_derivative_direction, double pertubation)
{
  FiniteElementState shape_disp(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");

  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solver.setShapeDisplacement(shape_disp);

  return computeThermalQoi(solver, ts_info);
}

double computeThermalQoiAdjustingConductivity(BasePhysics& solver, const TimeSteppingInfo& ts_info,
                                              const FiniteElementState& conductivity_derivative_direction,
                                              double                    pertubation)
{
  FiniteElementState cond(StateManager::mesh(mesh_tag), H1<p>{}, "input_conductivity");
  cond = 1.1;

  SLIC_ASSERT_MSG(cond.Size() == conductivity_derivative_direction.Size(),
                  "Conductivity and intended derivative direction FiniteElementState sizes do not agree.");

  cond.Add(pertubation, conductivity_derivative_direction);
  solver.setParameter(0, cond);

  return computeThermalQoi(solver, ts_info);
}

std::tuple<double, FiniteElementDual, FiniteElementDual> computeThermalQoiAndInitialTemperatureAndShapeSensitivity(
    BasePhysics& solver, const TimeSteppingInfo& ts_info)
{
  double qoi = computeThermalQoi(solver, ts_info);

  FiniteElementDual initial_temperature_sensitivity(solver.state("temperature").space(), "init_temp_sensitivity");
  initial_temperature_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "shape_sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(solver.state("temperature").space(), "adjoint_load");

  for (int i = solver.cycle(); i > 0; --i) {
    double             dt                      = solver.loadCheckpointedTimestep(i - 1);
    FiniteElementState temperature_end_of_step = solver.loadCheckpointedState("temperature", solver.cycle());
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    solver.setAdjointLoad({{"temperature", adjoint_load}});
    solver.reverseAdjointTimestep();
    shape_sensitivity += solver.computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(0, solver.cycle());  // we are back to the start
  auto initialConditionSensitivities     = solver.computeInitialConditionSensitivity();
  auto initialTemperatureSensitivityIter = initialConditionSensitivities.find("temperature");
  SLIC_ASSERT_MSG(initialTemperatureSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find temperature in the computed initial condition sensitivities.");
  initial_temperature_sensitivity += initialTemperatureSensitivityIter->second;

  return std::make_tuple(qoi, initial_temperature_sensitivity, shape_sensitivity);
}

std::tuple<double, FiniteElementDual> computeThermalConductivitySensitivity(BasePhysics&            solver,
                                                                            const TimeSteppingInfo& ts_info)
{
  double qoi = computeThermalQoi(solver, ts_info);

  FiniteElementDual conductivity_sensitivity(StateManager::mesh(mesh_tag), H1<p>{}, "conductivity_sensitivity");
  conductivity_sensitivity = 0.0;

  FiniteElementDual adjoint_load(solver.state("temperature").space(), "adjoint_load");

  for (int i = solver.cycle(); i > 0; --i) {
    double             dt                      = solver.loadCheckpointedTimestep(i - 1);
    FiniteElementState temperature_end_of_step = solver.loadCheckpointedState("temperature", solver.cycle());
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    solver.setAdjointLoad({{"temperature", adjoint_load}});
    solver.reverseAdjointTimestep();
    conductivity_sensitivity += solver.computeTimestepSensitivity(0);
  }

  EXPECT_EQ(0, solver.cycle());  // we are back to the start

  return std::make_tuple(qoi, conductivity_sensitivity);
}

struct HeatTransferSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(data_store, "thermal_dynamic_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
    mesh                 = &StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0), mesh_tag);
  }

  void fillDirection(FiniteElementState& direction) const
  {
    auto sz = direction.Size();
    for (int i = 0; i < sz; ++i) {
      direction(i) = -1.2 + 2.02 * (double(i) / sz);
    }
  }

  // Create DataStore
  axom::sidre::DataStore data_store;
  mfem::ParMesh*         mesh;

  // Solver options
  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};

  heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature              nonlinearMat{1.1, 1.2, 1.3, 1.9};
  heat_transfer::ParameterizedLinearIsotropicConductor                              parameterizedMat{1.0, 1.2, 0.01};
  heat_transfer::ParameterizedIsotropicConductorWithLinearConductivityVsTemperature parameterizedNonlinearMat{1.1, 1.2,
                                                                                                              1.3, 1.9};

  TimeSteppingInfo tsInfo;

  static constexpr double eps = 2e-7;
};

TEST_F(HeatTransferSensitivityFixture, InitialTemperatureSensitivities)
{
  auto thermal_solver = createNonlinearHeatTransfer(data_store, nonlinear_opts, dyn_opts, nonlinearMat);

  auto [qoi_base, temperature_sensitivity, _] =
      computeThermalQoiAndInitialTemperatureAndShapeSensitivity(*thermal_solver, tsInfo);

  thermal_solver->resetStates();
  FiniteElementState derivative_direction(temperature_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeThermalQoiAdjustingInitalTemperature(*thermal_solver, tsInfo, derivative_direction, eps);

  double directional_deriv = innerProduct(derivative_direction, temperature_sensitivity);
  ASSERT_TRUE(std::abs(directional_deriv) > 1e-13);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, ShapeSensitivities)
{
  auto thermal_solver = createNonlinearHeatTransfer(data_store, nonlinear_opts, dyn_opts, nonlinearMat);

  auto [qoi_base, _, shape_sensitivity] =
      computeThermalQoiAndInitialTemperatureAndShapeSensitivity(*thermal_solver, tsInfo);

  thermal_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus          = computeThermalQoiAdjustingShape(*thermal_solver, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  ASSERT_TRUE(std::abs(directional_deriv) > 1e-13);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, ConductivityParameterSensitivities)
{
  auto thermal_solver = createParameterizedHeatTransfer(data_store, nonlinear_opts, dyn_opts, parameterizedMat);
  auto [qoi_base, conductivity_sensitivity] = computeThermalConductivitySensitivity(*thermal_solver, tsInfo);

  thermal_solver->resetStates();
  FiniteElementState derivative_direction(conductivity_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus          = computeThermalQoiAdjustingConductivity(*thermal_solver, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, conductivity_sensitivity);
  ASSERT_TRUE(std::abs(directional_deriv) > 1e-13);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

TEST_F(HeatTransferSensitivityFixture, NonlinearConductivityParameterSensitivities)
{
  auto thermal_solver =
      createParameterizedNonlinearHeatTransfer(data_store, nonlinear_opts, dyn_opts, parameterizedNonlinearMat);
  auto [qoi_base, conductivity_sensitivity] = computeThermalConductivitySensitivity(*thermal_solver, tsInfo);

  thermal_solver->resetStates();
  FiniteElementState derivative_direction(conductivity_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus          = computeThermalQoiAdjustingConductivity(*thermal_solver, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, conductivity_sensitivity);
  ASSERT_TRUE(std::abs(directional_deriv) > 1e-13);
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
