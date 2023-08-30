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

constexpr int dim = 2;
constexpr int p   = 1;
const std::string thermal_prefix = "thermal";


struct TimeSteppingInfo
{
  double totalTime = 0.5;
  int num_timesteps = 4;
};


std::unique_ptr<HeatTransfer<p,dim>> create_heat_transfer(const NonlinearSolverOptions& nonlinear_opts,
                                                          const TimesteppingOptions& dyn_opts,
                                                          const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat)
{
  //eventually figure out how to clear out cider state 
  //auto saveMesh = std::make_unique<mfem::ParMesh>(StateManager::mesh());
  //StateManager::reset();
  //static int iter = 0;
  //StateManager::initialize(data_store, "thermal_dynamic_solve"+std::to_string(iter++));
  //std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
  //mfem::ParMesh* mesh = StateManager::setMesh(std::move(saveMesh));
  static int iter = 0;
  auto thermal = std::make_unique<HeatTransfer<p, dim>>(nonlinear_opts, heat_transfer::direct_linear_options, dyn_opts, thermal_prefix + std::to_string(iter++));
  thermal->setMaterial(mat);
  thermal->setTemperature([](const mfem::Vector&, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](const mfem::Vector&, double) { return 0.0; });
  thermal->setSource([](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();
  return thermal;
}


/* QOI */

double computeStepQoi(const FiniteElementState& temperature, double dt)
{
  // Compute qoi: \int_t \int_omega 0.5 * (T - T_target(x,t)^2)
  double nodalTemperatureNormSquared = 0.0;
  for (int n=0; n < temperature.Size(); ++n) {
    nodalTemperatureNormSquared += temperature(n) * temperature(n);
  }
  return nodalTemperatureNormSquared * dt;
}

void computeStepAdjointLoad(const FiniteElementState& temperature, FiniteElementDual& d_qoi_d_temperature, double dt)
{
  for (int n=0; n < temperature.Size(); ++n) {
    d_qoi_d_temperature(n) = dt * temperature(n);
  }
}


double computeThermalQoiAdjustingInitalTemperature(axom::sidre::DataStore& /*data_store*/,
                                                   const NonlinearSolverOptions& nonlinear_opts,
                                                   const TimesteppingOptions& dyn_opts,
                                                   const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat,
                                                   const TimeSteppingInfo& ts_info,
                                                   int dofIndex,
                                                   double pertubation)
{
  auto thermal = create_heat_transfer(nonlinear_opts, dyn_opts, mat);

  // finite difference change to initial temperature
  auto& temperatureSol = thermal->temperature();
  temperatureSol(dofIndex) += pertubation;

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }
  return 0.5 * qoi;
}


double computeThermalQoiAdjustingShape(axom::sidre::DataStore& /*data_store*/,
                                       const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat,
                                       const TimeSteppingInfo& ts_info,
                                       int dofIndex,
                                       double pertubation)
{
  auto thermal = create_heat_transfer(nonlinear_opts, dyn_opts, mat);

  // finite difference change to shape displacement
  auto& shapeDisp = thermal->shapeDisplacement();
  shapeDisp(dofIndex) += pertubation;

  double qoi = 0.0;
  thermal->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    thermal->advanceTimestep(dt);
    thermal->outputState();
    qoi += computeStepQoi(thermal->temperature(), dt);
  }
  return 0.5 * qoi;
}


std::pair<double, std::vector<double>> computeThermalQoiAndInitialTemperatureGradient(axom::sidre::DataStore& /*data_store*/,
                                                                                      const NonlinearSolverOptions& nonlinear_opts,
                                                                                      const TimesteppingOptions& dyn_opts,
                                                                                      const heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature& mat,
                                                                                      const TimeSteppingInfo& ts_info)
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
  qoi *= 0.5;

  size_t Nsize = static_cast<size_t>(thermal->temperature().Size());
  std::vector<double> gradient(Nsize, 0.0);

  FiniteElementDual adjoint_load(thermal->temperature().space(), "adjoint_load");
  FiniteElementState temperature_end_of_step(thermal->temperature()); // cannot get with thermal.previousTemperature(adjointCycle) yet, as adjointCycle is not defined until reverseAdjointTimestep is called.

  for (int i = ts_info.num_timesteps; i > 0; --i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    computeStepAdjointLoad(temperature_end_of_step, adjoint_load, dt);
    std::unordered_map<std::string, const FiniteElementState&> adjoint_sol = thermal->reverseAdjointTimestep({{"temperature", adjoint_load}});
    temperature_end_of_step = thermal->previousTemperature(thermal->cycle());

    if (i==1) {
      // initial temperature sensitivity math
      auto mu = adjoint_sol.find("adjoint_d_temperature_dt")->second;
      for (size_t n=0; n < Nsize; ++n) {
        gradient[n] += mu(int(n));
      }
    }
  }

  return std::make_pair(qoi, gradient);
}


class HeatTransferSensitivityFixture : public ::testing::Test
{
  protected:

  void SetUp() override {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "thermal_dynamic_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
    mesh = StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0));

  }

  int NumDofs() const {
    auto initialTemperature = StateManager::newState(FiniteElementState::Options{.order = p, .vector_dim = 1, .name = detail::addPrefix(thermal_prefix, "initial_temperature")}, StateManager::collectionID(mesh)); // really just constructing a field to get a size
    return initialTemperature.Size();
  }

  // Create DataStore
  axom::sidre::DataStore dataStore;
  mfem::ParMesh* mesh;

  // Solver options
  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};
  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::BackwardEuler,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};
  heat_transfer::IsotropicConductorWithLinearConductivityVsTemperature mat{1.0, 1.0, 1.0, 2.0};
  TimeSteppingInfo tsInfo{.totalTime=0.5, .num_timesteps=4};
};

TEST_F(HeatTransferSensitivityFixture, InitialTemperatureSensitivities)
{
  std::pair<double, std::vector<double>> trueGrad = computeThermalQoiAndInitialTemperatureGradient(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);
  double qoiBase = trueGrad.first;
  const auto& adjGradient = trueGrad.second;
  int N = NumDofs();
  //size_t Nsize = static_cast<size_t>(N);

  double eps = 1e-7;
  std::vector<double> numericalGradients(static_cast<size_t>(N));

  for (int i=0; i < N; ++i) {
    auto qoiPlus = computeThermalQoiAdjustingInitalTemperature(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, i, eps);
    numericalGradients[size_t(i)] = (qoiPlus-qoiBase)/eps;
  }

  for (size_t i=0; i < size_t(N); ++i) {
    EXPECT_NEAR(numericalGradients[i], adjGradient[i], 1e-6);
  }
}

TEST_F(HeatTransferSensitivityFixture, ShapeSensitivities)
{
  std::pair<double, std::vector<double>> trueGrad = computeThermalQoiAndInitialTemperatureGradient(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);
  double qoiBase = trueGrad.first;
  const auto& adjGradient = trueGrad.second;
  int N = NumDofs();

  double eps = 1e-7;
  std::vector<double> numericalGradients(static_cast<size_t>(N));

  for (int i=0; i < N; ++i) {
    auto qoiPlus = computeThermalQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, i, eps);
    numericalGradients[size_t(i)] = (qoiPlus-qoiBase)/eps;
  }

  for (size_t i=0; i < size_t(N); ++i) {
    EXPECT_NEAR(numericalGradients[i], adjGradient[i], 1e-6);
  }
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
