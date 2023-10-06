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

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::NeoHookean;
//using SolidMaterial = solid_mechanics::LinearIsotropic;
auto geoNonlinear = GeometricNonlinearities::On;
//auto geoNonlinear = GeometricNonlinearities::Off;

struct TimeSteppingInfo {
  //TimeSteppingInfo() : dts({0.0, 0.2, 0.4, 0.24, 0.12, 0.0}) {}
  TimeSteppingInfo() : dts({0.0, 0.2, 0.2, 0.2, 0.2, 0.0}) {}

  int numTimesteps() const { return dts.Size()-2; }

  mfem::Vector dts;
};

// MRT: add explicit velocity dependence
double computeStepQoi(const FiniteElementState& displacement, double dt)
{
  FiniteElementState cheat(displacement);
  cheat = -0.0;
  cheat.Add(1.0, displacement);
  return 0.5 * dt * innerProduct(cheat, cheat);
}

void computeStepAdjointLoad(const FiniteElementState& displacement, FiniteElementDual& d_qoi_d_displacement, double dt)
{
  d_qoi_d_displacement = -0.0;
  d_qoi_d_displacement.Add(1.0, displacement);
  d_qoi_d_displacement *= dt;
}

std::unique_ptr<SolidMechanics<p, dim>> createNonlinearSolidMechanicsSolver(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                                  dyn_opts,
    const SolidMaterial& mat)
{
  static int iter = 0;
  auto solid = std::make_unique<SolidMechanics<p, dim>>(nonlinear_opts, solid_mechanics::direct_linear_options, dyn_opts,
                                                        geoNonlinear, physics_prefix + std::to_string(iter++), mesh_tag);
  solid->setMaterial(mat);
  solid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& disp) { disp = 0.0; });
  solid->addBodyForce([](auto X, auto /* t */) {
    auto Y = X;
    Y[0] = 0.1 + 0.1 * X[0] + 0.3 * X[1];
    Y[1] = -0.05 - 0.08 * X[0] + 0.15 * X[1];
    return 0.1*X + Y;
  });
  solid->completeSetup();
  return solid;
}

double computeSolidMechanicsQoiAdjustingShape(axom::sidre::DataStore& data_store,
                                              const NonlinearSolverOptions& nonlinear_opts,
                                              const TimesteppingOptions& dyn_opts,
                                              const SolidMaterial& mat,
                                              const TimeSteppingInfo&   ts_info,
                                              const FiniteElementState& shape_derivative_direction, double pertubation)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  FiniteElementState shape_disp(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");

  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputStateToDisk();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputStateToDisk();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }
  return qoi;
}

double computeSolidMechanicsQoiAdjustingInitialDisplacement(axom::sidre::DataStore& data_store,
                                       const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const SolidMaterial& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& derivative_direction, double pertubation)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  FiniteElementState disp = solid_solver->displacement();
  SLIC_ASSERT_MSG(disp.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  disp.Add(pertubation, derivative_direction);
  solid_solver->setDisplacement(disp);

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputStateToDisk();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputStateToDisk();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }
  return qoi;
}

double computeSolidMechanicsQoiAdjustingInitialVelocity(axom::sidre::DataStore& data_store,
                                       const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const SolidMaterial& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& derivative_direction, double pertubation)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  FiniteElementState velo = solid_solver->velocity();
  SLIC_ASSERT_MSG(velo.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  velo.Add(pertubation, derivative_direction);
  solid_solver->setVelocity(velo);

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputStateToDisk();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputStateToDisk();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }
  return qoi;
}

std::tuple<double, FiniteElementDual, FiniteElementDual, FiniteElementDual> computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                        dyn_opts,
    const SolidMaterial& mat, const TimeSteppingInfo& ts_info)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);
  EXPECT_EQ(0, solid_solver->cycle());

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputStateToDisk();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputStateToDisk();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }

  FiniteElementDual initial_displacement_sensitivity(solid_solver->displacement().space(), "init_displacement_sensitivity");
  initial_displacement_sensitivity = 0.0;
  FiniteElementDual initial_velocity_sensitivity(solid_solver->velocity().space(), "init_velocity_sensitivity");
  initial_velocity_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "shape_sensitivity"); // MRT, try to get this size from the physics
  shape_sensitivity = 0.0;

  FiniteElementDual adjoint_load(solid_solver->displacement().space(), "adjoint_displacement_load");

  // for solids, we go back to time = 0, because there is an extra hidden implicit solve at the start
  // consider unifying the interface between solids and thermal
  for (int i = solid_solver->cycle(); i > 0; --i) {
    FiniteElementState displacement = solid_solver->loadCheckpointedDisplacement(solid_solver->cycle());
    computeStepAdjointLoad(displacement, adjoint_load, 
                      0.5*(solid_solver->loadCheckpointedTimestep(i-1) + solid_solver->loadCheckpointedTimestep(i)));
    solid_solver->reverseAdjointTimestep({{"displacement", adjoint_load}});
    shape_sensitivity += solid_solver->computeTimestepShapeSensitivity();
  }

  EXPECT_EQ(0, solid_solver->cycle());  // we are back to the start
  auto initialConditionSensitivities     = solid_solver->computeInitialConditionSensitivity();
  auto initialDisplacementSensitivityIter = initialConditionSensitivities.find("displacement");
  auto initialVelocitySensitivityIter = initialConditionSensitivities.find("velocity");
  SLIC_ASSERT_MSG(initialDisplacementSensitivityIter != initialConditionSensitivities.end(),
                  "Could not find displacement in the computed initial condition sensitivities.");
  SLIC_ASSERT_MSG(initialVelocitySensitivityIter != initialConditionSensitivities.end(),
                  "Could not find displacement in the computed initial condition sensitivities.");
  initial_displacement_sensitivity = initialDisplacementSensitivityIter->second;
  initial_velocity_sensitivity = initialVelocitySensitivityIter->second;

  return std::make_tuple(qoi, initial_displacement_sensitivity, initial_velocity_sensitivity, shape_sensitivity);
}


struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/star.mesh";
    mesh                 = &StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0), mesh_tag);
    mat.density = 1.0;
    mat.K = 1.0;
    mat.G = 0.1;
  }

  void fillDirection(FiniteElementState& direction) const 
  {
    auto sz = direction.Size();
    for (int i=0; i < sz; ++i) {
      direction(i) = -1.2 + 2.1 * (double(i)/sz);
    }
  }

  axom::sidre::DataStore dataStore;
  mfem::ParMesh*         mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::Newmark,
                               .enforcement_method = DirichletEnforcementMethod::RateControl};

  SolidMaterial mat;
  TimeSteppingInfo tsInfo;
};

TEST_F(SolidMechanicsSensitivityFixture, InitialDisplacementSensitivities)
{
  auto [qoi_base, init_disp_sensitivity, _, __] = 
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(init_disp_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus = computeSolidMechanicsQoiAdjustingInitialDisplacement(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_disp_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 16*eps);
}

TEST_F(SolidMechanicsSensitivityFixture, InitialVelocitySensitivities)
{
  auto [qoi_base, _, init_velo_sensitivity, __] = 
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(init_velo_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus = computeSolidMechanicsQoiAdjustingInitialVelocity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                     derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_velo_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 16*eps);
}

TEST_F(SolidMechanicsSensitivityFixture, ShapeSensitivities)
{
  auto [qoi_base, _, __, shape_sensitivity] =
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus =
      computeSolidMechanicsQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, derivative_direction, eps);
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
