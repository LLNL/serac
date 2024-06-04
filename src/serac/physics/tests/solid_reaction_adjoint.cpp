// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

using SolidMechanicsType = SolidMechanics<p, dim>;

const std::string mesh_tag       = "mesh";
const std::string physics_prefix = "solid";

using SolidMaterial = solid_mechanics::NeoHookean;
auto geoNonlinear   = GeometricNonlinearities::On;

// constexpr double dispTarget   = -0.34;
constexpr double boundaryDisp = 0.013;

std::unique_ptr<SolidMechanicsType> createNonlinearSolidMechanicsSolver(
    axom::sidre::DataStore& /*data_store*/, const NonlinearSolverOptions& nonlinear_opts,
    const SolidMaterial& mat)
{
  static int iter = 0;
  auto       solid =
      std::make_unique<SolidMechanicsType>(nonlinear_opts, solid_mechanics::direct_linear_options, TimesteppingOptions(),
                                           geoNonlinear, physics_prefix + std::to_string(iter++), mesh_tag);
  solid->setMaterial(mat);
  solid->setDisplacementBCs({1}, [](const mfem::Vector& x, mfem::Vector& disp) {
    disp = boundaryDisp;
    std::cout << x.Size() << ", x = " << x[0] << " " << x[1] << std::endl;

  });
  solid->addBodyForce([](auto X, auto /* t */) {
    auto Y = X;
    Y[0]   = 0.1 + 0.1 * X[0] + 0.3 * X[1];
    Y[1]   = -0.05 - 0.08 * X[0] + 0.15 * X[1];
    return 0.1 * X + Y;
  });
  solid->completeSetup();

  return solid;
}

FiniteElementState createReactionDirection(const SolidMechanicsType& solid_solver,  int direction) 
{
  const FiniteElementDual& reactions = solid_solver.reactions();
  FiniteElementState reactionDirections(reactions.space(), "reaction_directions");
  reactionDirections = 0.0;

  auto reactionTrueDofs = solid_solver.calculateConstrainedDofs([&](const mfem::Vector& x) {
    return x[0] < 1e-14;
  }, direction);

  reactionDirections.SetSubVector(reactionTrueDofs, 1.0);
  return reactionDirections;
}

double computeSolidMechanicsQoi(BasePhysics& solid_solver)
{
  solid_solver.advanceTimestep(0.0);
  auto& solidS = dynamic_cast<SolidMechanicsType&>(solid_solver);
  const FiniteElementDual& reactions = solidS.reactions();
  auto reactionDirections = createReactionDirection(solidS, 0);
  return innerProduct(reactions, reactionDirections);
}

std::tuple<double, FiniteElementDual> computeSolidMechanicsQoiAndShapeSensitivity(BasePhysics& solid_solver)
{
  double qoi = computeSolidMechanicsQoi(solid_solver);
  FiniteElementDual shape_sensitivity(solid_solver.shapeDisplacement().space(), "shape sensitivity");
  shape_sensitivity = 0.0;

  auto& solidS = dynamic_cast<SolidMechanicsType&>(solid_solver);

  auto reactionDirections = createReactionDirection(solidS, 0);

  auto reactionAdjointLoad = solidS.computeReactionsAdjointLoad(reactionDirections);
  shape_sensitivity += solidS.computeReactionsShapeSensitivity(reactionDirections);

  solid_solver.setAdjointLoad({{"displacement", reactionAdjointLoad}});
  solid_solver.reverseAdjointTimestep();
  shape_sensitivity += solid_solver.computeTimestepShapeSensitivity();
  
  return std::make_pair(qoi, shape_sensitivity);
}

double computeSolidMechanicsQoiAdjustingShape(BasePhysics& solid_solver, const FiniteElementState& shape_derivative_direction, double pertubation)
{
  FiniteElementState shape_disp(StateManager::mesh(mesh_tag), H1<SHAPE_ORDER, dim>{}, "input_shape_displacement");
  SLIC_ASSERT_MSG(shape_disp.Size() == shape_derivative_direction.Size(),
                  "Shape displacement and intended derivative direction FiniteElementState sizes do not agree.");

  shape_disp.Add(pertubation, shape_derivative_direction);
  solid_solver.setShapeDisplacement(shape_disp);

  return computeSolidMechanicsQoi(solid_solver);
}


struct SolidMechanicsSensitivityFixture : public ::testing::Test {
  void SetUp() override
  {
    MPI_Barrier(MPI_COMM_WORLD);
    StateManager::initialize(dataStore, "solid_mechanics_solve");
    std::string filename = std::string(SERAC_REPO_DIR) + "/data/meshes/patch2D_quads.mesh";  //"/data/meshes/star.mesh";
    mesh                 = &StateManager::setMesh(mesh::refineAndDistribute(buildMeshFromFile(filename), 0), mesh_tag);
    mat.density          = 1.0;
    mat.K                = 1.0;
    mat.G                = 0.1;
  }

  void fillDirection(FiniteElementState& direction) const
  {
    auto sz = direction.Size();
    for (int i = 0; i < sz; ++i) {
      direction(i) = -1.2 + 2.02 * (double(i) / sz);
    }
  }

  axom::sidre::DataStore dataStore;
  mfem::ParMesh* mesh;

  NonlinearSolverOptions nonlinear_opts{.relative_tol = 1.0e-15, .absolute_tol = 1.0e-15};

  bool dispBc = true;

  SolidMaterial mat;

  static constexpr double eps = 2e-7;
};

TEST_F(SolidMechanicsSensitivityFixture, ReactionShapeSensitivities)
{
  auto solid_solver = createNonlinearSolidMechanicsSolver(dataStore, nonlinear_opts, mat);
  auto [qoi_base, shape_sensitivity] = computeSolidMechanicsQoiAndShapeSensitivity(*solid_solver);

  std::cout << "qoi = " << qoi_base << std::endl;

  solid_solver->resetStates();
  FiniteElementState derivative_direction(shape_sensitivity.space(), "derivative_direction");
  fillDirection(derivative_direction);

  double qoi_plus = computeSolidMechanicsQoiAdjustingShape(*solid_solver, derivative_direction, eps);

  double directional_deriv = innerProduct(derivative_direction, shape_sensitivity);
  EXPECT_NEAR(directional_deriv, (qoi_plus - qoi_base) / eps, eps);
}

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;
  std::cout << std::setprecision(16);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
