// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <functional>
#include <set>
#include <string>

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/serac_config.hpp"

namespace serac {

constexpr int dim = 2;
constexpr int p   = 1;

const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::NeoHookean;

struct TimeSteppingInfo {
  double totalTime     = 0.0;
  int    num_timesteps = 0;
};

// MRT: add explicit velocity dependence
double computeStepQoi(const FiniteElementState& displacement, double dt)
{
  return 0.5 * dt * innerProduct(displacement, displacement);
}

void computeStepAdjointLoad(const FiniteElementState& displacement, FiniteElementDual& d_qoi_d_displacement, double dt)
{
  d_qoi_d_displacement = displacement;
  d_qoi_d_displacement *= dt;
}

std::unique_ptr<SolidMechanics<p, dim>> createNonlinearSolidMechanicsSolver(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const SolidMaterial& mat)
{
  static int iter = 0;
  auto solid = std::make_unique<SolidMechanics<p, dim>>(nonlinear_opts, solid_mechanics::direct_linear_options, dyn_opts,
                                                        GeometricNonlinearities::On, physics_prefix + std::to_string(iter++));
  solid->setMaterial(mat);
  solid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& disp) { disp = 0.0; });
  solid->addBodyForce([](auto X, auto /* t */) {
    auto Y = X;
    Y[0] = 1.0;
    Y[1] = -0.5;
    return 0.0*X + Y;
  });
  solid->completeSetup();
  return solid;
}

double computeSolidMechanicsQoiAdjustingShape(axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const SolidMaterial& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& shape_derivative_direction, double pertubation)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  auto& shape_disp = solid_solver->shapeDisplacement();
  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);

  double qoi = 0.0;
  solid_solver->outputState();
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    solid_solver->advanceTimestep(dt);
    solid_solver->outputState();
    qoi += computeStepQoi(solid_solver->displacement(), dt);
  }
  return qoi;
}

std::tuple<double, FiniteElementDual, FiniteElementDual> computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const SolidMaterial& mat, const TimeSteppingInfo& ts_info)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  double qoi = 0.0;
  solid_solver->outputState(); // because acceleration is solved interior to mfem time integrator, we do not have a0 checkpointed!
  for (int i = 0; i < ts_info.num_timesteps; ++i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;
    solid_solver->advanceTimestep(dt);
    solid_solver->outputState(); // checkpoint disp, velo, accel
    qoi += computeStepQoi(solid_solver->displacement(), dt);
  }

  FiniteElementDual initial_displacement_sensitivity(solid_solver->displacement().space(), "init_displacement_sensitivity");
  initial_displacement_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(solid_solver->shapeDisplacement().space(), "shape_sensitivity");
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(solid_solver->displacement().space(), "adjoint_displacement_load");

  // for solids, we go back to time = 0, because there is an extra hidden implicit solve at the start
  // consider unifying the interface between solids and thermal
  for (int i = ts_info.num_timesteps; i >= 0; --i) {
    double dt = ts_info.totalTime / ts_info.num_timesteps;

    FiniteElementState displacement_end_of_step = solid_solver->loadCheckpointedDisplacement(solid_solver->cycle());
    computeStepAdjointLoad(displacement_end_of_step, adjoint_load, dt);
    solid_solver->reverseAdjointTimestep({{"displacement", adjoint_load}});
    shape_sensitivity += solid_solver->computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(-1, solid_solver->cycle());  // we are back to the start
  auto initialConditionSensitivities     = solid_solver->computeInitialConditionSensitivity();
  auto initialDisplacementSensitivityIter = initialConditionSensitivities.find("displacement");
  SLIC_ASSERT_MSG(initialDisplacementSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find displacement in the computed initial condition sensitivities.");
  initial_displacement_sensitivity = initialDisplacementSensitivityIter->second;

  return std::make_tuple(qoi, initial_displacement_sensitivity, shape_sensitivity);
}


struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
    mesh                 = StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0));
    mat.density = 1.0;
    mat.K = 1.0;
    mat.G = 0.1;
  }

  void fillDirection(FiniteElementState& direction) const 
  {
    direction = 1.1;
    auto sz = direction.Size();
    for (int i=0; i < sz; ++i) {
      direction(i) = -1.2 + 2.5 * (double(i)/sz);
    }
  }

  axom::sidre::DataStore dataStore;
  mfem::ParMesh*         mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::Newmark,
                               .enforcement_method = DirichletEnforcementMethod::DirectControl};

  SolidMaterial mat;
  TimeSteppingInfo tsInfo{.totalTime = 1.0, .num_timesteps = 4};
};

TEST_F(SolidMechanicsSensitivityFixture, InitialDisplacementSensitivities)
{
  auto [qoi_base, init_disp_sensitivity, _] =
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(init_disp_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps      = 1e-7;
  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_disp_sensitivity);
  std::cout << "qoi, other = " << directional_deriv << " " << (qoi_plus - qoi_base) / eps << std::endl;
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

/* TEST_F(HeatTransferSensitivityFixture, ShapeSensitivities)
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
} */

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
