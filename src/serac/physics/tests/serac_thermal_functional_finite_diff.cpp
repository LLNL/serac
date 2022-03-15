// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_conduction_functional.hpp"
#include "serac/physics/materials/thermal_functional_material.hpp"
#include "serac/physics/materials/parameterized_thermal_functional_material.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

TEST(thermal_functional_finite_diff, finite_difference)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/square.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  serac::StateManager::setMesh(std::move(mesh));

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1, 2};

  // Construct and initialized the user-defined conductivity to be used as a differentiable parameter in
  // the thermal conduction physics module.
  FiniteElementState user_defined_conductivity(
      StateManager::newState(FiniteElementState::Options{.order = 1, .name = "parameterized_conductivity"}));

  double conductivity_value = 1.2;

  user_defined_conductivity = conductivity_value;

  // We must know the index of the parameter finite element state in our parameter pack to take sensitivities.
  // As we only have one parameter in this example, the index is zero.
  constexpr int conductivity_parameter_index = 0;

  // Construct a functional-based thermal conduction solver
  //
  // Note that we now include an extra template parameter indicating the finite element space for the parameterized
  // field, in this case the thermal conductivity. We also pass an array of finite element states for each of the
  // requested parameterized fields.
  ThermalConductionFunctional<p, dim, H1<1>> thermal_solver(Thermal::defaultQuasistaticOptions(), "thermal_functional",
                                                            {user_defined_conductivity});

  // Construct a potentially user-defined parameterized material and send it to the thermal module
  Thermal::ParameterizedLinearIsotropicConductor mat;
  thermal_solver.setMaterial(mat);

  // Define the function for the initial temperature and boundary condition
  auto bdr_temp = [](const mfem::Vector& x, double) -> double {
    if (x[0] < 0.5 || x[1] < 0.5) {
      return 1.0;
    }
    return 0.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, bdr_temp);
  thermal_solver.setTemperature(bdr_temp);

  // Define a constant source term
  Thermal::ConstantSource source{1.0};
  thermal_solver.setSource(source);

  // Set the flux term to zero for testing code paths
  Thermal::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // Output the sidre-based plot files
  thermal_solver.outputState();

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjoint_load_form(&thermal_solver.temperature().space());
  adjoint_load_form = 1.0;

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the temperature.
  serac::FiniteElementDual              adjoint_load(*mesh, thermal_solver.temperature().space(), "adjoint_load");
  std::unique_ptr<mfem::HypreParVector> assembled_vector(adjoint_load_form.ParallelAssemble());
  adjoint_load.trueVec() = *assembled_vector;
  adjoint_load.distributeSharedDofs();

  // Solve the adjoint problem
  thermal_solver.solveAdjoint(adjoint_load);

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  [[maybe_unused]] auto& sensitivity = thermal_solver.computeSensitivity<conductivity_parameter_index>();

  // Perform finite difference on each conduction value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the temperature
  double eps = 1.0e-4;
  for (int i = 0; i < user_defined_conductivity.gridFunc().Size(); ++i) {
    // Perturb the conductivity
    user_defined_conductivity.trueVec()(i) = conductivity_value + eps;
    user_defined_conductivity.distributeSharedDofs();

    thermal_solver.advanceTimestep(dt);
    mfem::ParGridFunction temperature_plus = thermal_solver.temperature().gridFunc();

    user_defined_conductivity.trueVec()(i) = conductivity_value - eps;
    user_defined_conductivity.distributeSharedDofs();

    thermal_solver.advanceTimestep(dt);
    mfem::ParGridFunction temperature_minus = thermal_solver.temperature().gridFunc();

    // Reset to the original conductivity value
    user_defined_conductivity.trueVec()(i) = conductivity_value;
    user_defined_conductivity.distributeSharedDofs();

    // Finite difference to compute sensitivity of temperature with respect to conductivity
    mfem::ParGridFunction dtemp_dconductivity(&thermal_solver.temperature().space());
    for (int i2 = 0; i2 < temperature_plus.Size(); ++i2) {
      dtemp_dconductivity(i2) = (temperature_plus(i2) - temperature_minus(i2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to conductivity
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dconductivity = adjoint_load_form(dtemp_dconductivity);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dconductivity: {}", dqoi_dconductivity));
    SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity.trueVec()(i)));
    EXPECT_NEAR((sensitivity.trueVec()(i) - dqoi_dconductivity) / dqoi_dconductivity, 0.0, 1.0e-3);
  }
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
