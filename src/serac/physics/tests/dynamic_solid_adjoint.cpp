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
auto geoNonlinear = GeometricNonlinearities::On;

struct TimeSteppingInfo {
  TimeSteppingInfo() : dts({0.0, 0.2, 0.4, 0.24, 0.12, 0.0}) {}

  int numTimesteps() const { return dts.Size()-2; }

  mfem::Vector dts;
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
                                                        geoNonlinear, physics_prefix + std::to_string(iter++));
  solid->setMaterial(mat);
  // solid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& disp) { disp = 0.0; });
  solid->addBodyForce([](auto X, auto /* t */) {
    auto Y = X;
    Y[0] = 0.1;
    Y[1] = -0.05;
    return 0.1*X + Y;
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

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputState();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputState();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }
  return qoi;
}

double computeSolidMechanicsQoiAdjustingInitialDisplacement(axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
                                       const TimesteppingOptions& dyn_opts,
                                       const SolidMaterial& mat,
                                       const TimeSteppingInfo&   ts_info,
                                       const FiniteElementState& derivative_direction, double pertubation)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);

  auto& disp = solid_solver->displacement();
  SLIC_ASSERT_MSG(disp.Size() == derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  disp.Add(pertubation, derivative_direction);

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputState();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputState();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }
  return qoi;
}

std::tuple<double, FiniteElementDual, FiniteElementDual> computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(
    axom::sidre::DataStore& data_store, const NonlinearSolverOptions& nonlinear_opts,
    const TimesteppingOptions&                                        dyn_opts,
    const SolidMaterial& mat, const TimeSteppingInfo& ts_info)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(data_store, nonlinear_opts, dyn_opts, mat);
  EXPECT_EQ(0, solid_solver->cycle());

  auto dts = ts_info.dts;
  solid_solver->advanceTimestep(dts(0)); // advance by 0.0 seconds to get initial acceleration
  solid_solver->outputState();
  double qoi = computeStepQoi(solid_solver->displacement(), 0.5 * (dts(0) + dts(1)));
  for (int i = 1; i <= ts_info.numTimesteps(); ++i) {
    EXPECT_EQ(i, solid_solver->cycle());
    solid_solver->advanceTimestep(dts(i));
    solid_solver->outputState();
    qoi += computeStepQoi(solid_solver->displacement(), 0.5 * (dts(i) + dts(i+1)));
  }

  FiniteElementDual initial_displacement_sensitivity(solid_solver->displacement().space(), "init_displacement_sensitivity");
  initial_displacement_sensitivity = 0.0;
  FiniteElementDual shape_sensitivity(solid_solver->shapeDisplacement().space(), "shape_sensitivity");
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
    auto sz = direction.Size();
    for (int i=0; i < sz; ++i) {
      direction(i) = -1.2 + 2.1 * (double(i)/sz);
    }
  }

  axom::sidre::DataStore dataStore;
  mfem::ParMesh*         mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 5.0e-13, .absolute_tol = 5.0e-13};

  TimesteppingOptions dyn_opts{.timestepper        = TimestepMethod::Newmark,
                               .enforcement_method = DirichletEnforcementMethod::FullControl};

  SolidMaterial mat;
  TimeSteppingInfo tsInfo;
};

TEST_F(SolidMechanicsSensitivityFixture, InitialDisplacementSensitivities)
{
  auto [qoi_base, init_disp_sensitivity, _] = 
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(init_disp_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus = computeSolidMechanicsQoiAdjustingInitialDisplacement(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo,
                                                                derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, init_disp_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 16*eps);
}

TEST_F(SolidMechanicsSensitivityFixture, ShapeSensitivities)
{
  auto [qoi_base, _, shape_sensitivity] =
      computeSolidMechanicsQoiAndInitialDisplacementAndShapeSensitivity(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo);

  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  const double eps = 1e-7;

  double qoi_plus =
      computeSolidMechanicsQoiAdjustingShape(dataStore, nonlinear_opts, dyn_opts, mat, tsInfo, derivative_direction, eps);
  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, 8*eps);
}

struct Thermal;
struct SolidMechnaics;

struct ThermalNeohookeanMaterialSimp {
  double E,G,rho,stiffness_temperature_slope;
};

struct IsotropicConductivitySimp {
  double density = 1.0;
  double specific_heat_capacity = 1.0;
  double conductivity = 1.0;
};

struct SmoothedCoulombFriction {
  double mu = 0.0;
  double length_scale = 0.0;
}

struct Field {}
struct DualField {}
struct QuadratureField {}
struct DualQuadratureField {}

template <typename dim>
struct Vec { double data[dim]; }

TEST(SimpleInterface, A)
{
  static constexpr p = 1;
  static constexpr dim = 2;
  using VecD = Vec<dim>;

  int mesh, sub_mesh;

  ThermalNeohookeanMaterialSimp solid_mat;
  solid_mat.E = 1.0;
  solid_mat.rho = 1.0;
  solid_mat.G = 0.3;
  solid_mat.stiffness_temperature_slope = 0.01;

  IsotropicConductivitySimp thermal_mat;
  thermal_mat.conductivity = 1.1;
  thermal_mat.density = 0.5;
  thermal_mat.specific_heat_capacity = 0.3;

  SmoothedCoulombFriction friction_model;
  friction_model.mu = 0.4;
  friction_model.length_scale = 1e-4;

  Field shape_displacement(mesh, H1<p,2>);
  Field displacement(mesh, H1<p,2>);
  Field velocity(mesh, H1<p,2>);
  Field acceleration(mesh, H1<p,2>);
  Field temperature(mesh, H1<p,1>);
  Field simp_density(mesh, L2<0,1>, subMesh);

  int numStateVars = 4;
  QuadratureField state_variables(mesh, L2<0,1>, numStateVars);

  DualField solid_residual(displacement);
  DualField thermal_residual(temperature);

  DualField contact_area_gaps(surfaceOf(mesh, {2}), p, 1);
  Field contact_pressures(contact_area_gaps);

  SolidMechanics solidMech(mesh);

   // set different materials, bcs, forces, etc on different blocks
  solidMech->setMaterial(DependsOn<0,1>{}, {0}, solid_mat);
  solidMech->setDisplacementBCs({1}, [](VecD /*x*/, VecD disp) { disp = 0.0; });
  solidMech->addBodyForce({0}, [](VecD X, double /*t*/) {
    VecD Y = X;
    Y[0] = 0.1;
    Y[1] = -0.05;
    return 0.1*X + Y;
  });
  solidMech->completeSetup();

  // parameters are displacement, elem-wise simp-density
  Thermal thermal(mesh); // parameters are current displacement, elem-wise simp-density
  thermal->setMaterial(DependsOn<1>{}, {0}, thermal_mat);
  thermal->setTemperature({0}, [](VecD, double) { return 0.0; });
  thermal->setTemperatureBCs({1}, [](Vec3, double) { return 0.0; });
  thermal->setSource({0}, [](auto /* X */, auto /* time */, auto /* u */, auto /* du_dx */) { return 1.0; });
  thermal->completeSetup();

  SolidContact contact(surfaceMesh);
  contact->setFrictionModel({1}, friction_model);

  double dt = 0.1;

  // the true and false template arguments say which sensitivity linear operators to compute and return
  auto [solidRes, op_d_solidRes_d_disp, op_d_solidRes_d_temp] = solidMech<true, false, false, true, false>forceOperators(dt, displacement, velocity, shape_displacement, temperature, simp_density);
  auto [inertialSolidRes, op_d_solidRes_d_velo] = solidMech<false, true, false, false, false>inertialOperators(dt, displacement, velocity, shape_displacement, temperature, simp_density);
  auto [thermalRes, op_d_thermalRes_d_temp] = thermal<true, false, true, false>forceOperators(dt, temperature, shape_displacement, displacement, simp_density);
  auto [inertialThermalRes, op_d_thermalRes_d_tempDot, op_d_thermalRes_d_disp] = thermal<true, false, true, false>inertialOperators(dt, temperature, shape_displacement, displacement, simp_density);
  auto [evalContactForceConstraintPressure, op_contactStiffUU, op_contactStiffUL, op_contactStiffLU, op_contactStiffLL, _, _] = contact.contactOperators(dt, displacement, velocity, contact_pressures, shape_displacement);
  
  // alternative syntax
  // auto [solidRes, op_d_solidRes_d_disp, _, op_d_solidRes_d_temp, _] = forceOperators(dt, displacement, velocity, shapeDisplacement, temperature, simpDensity, {true, false, false, false, true, false});

  // sum solid internal force contributions
  solidRes(solid_residual);
  inertialSolidRes(solid_residual);
  // careful of the state of these contact presure.  They are inputs for the linearization (perhaps), but also outputs of the operator here
  evalContactForceConstraintPressure(solid_residual, contact_area_gaps, contact_pressures);

  // sum thermal residual contributions
  thermalRes(thermal_residual);

  // we are trying to solve for
  // solid_residual = 0
  // thermal_residual = 0
  // NCP(contact_area_gaps, contact_pressures) = 0
  // i.e., g <= 0, p >= 0, g * p = 0
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
