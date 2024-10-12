// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

template <int dim>
tensor<double, dim> average(std::vector<tensor<double, dim> >& positions)
{
  tensor<double, dim> total{};
  for (auto x : positions) {
    total += x;
  }
  return total / double(positions.size());
}


TEST(Solid, MultiMaterial)
{
  /*
   * Checks multi material case with the following uniaxial problem:
   *              MATERIAL 1            MATERIAL 2
   *               E = 1                 E = 2 
   * u = 0   --------------------|-------------------- stress = 1
   * 
   * strain =       1                    0.5
   * 
   */
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 2;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_multimaterial");

  constexpr double L = 8.0;
  constexpr double W = 1.0;
  constexpr double H = 1.0;
  constexpr double VOLUME = L*W*H;

  auto mesh = mesh::refineAndDistribute(buildCuboidMesh(8, 1, 1, L, W, H), serial_refinement, parallel_refinement);

  const std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // _solver_params_start
  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               GeometricNonlinearities::Off, "solid_mechanics", mesh_tag);
  // _solver_params_end

  using Material = solid_mechanics::LinearIsotropic;

  constexpr double E_left = 1.0;
  constexpr double nu_left = 0.125;
  Material mat_left{.K = E_left/3.0/(1 - 2*nu_left), .G = 0.5*E_left/(1 + nu_left)};

  constexpr double E_right = 2.0*E_left;
  constexpr double nu_right = 2*nu_left;
  Material mat_right{.K = E_right/3.0/(1 - 2*nu_right), .G = 0.5*E_right/(1 + nu_right)};
//   using Hardening = solid_mechanics::LinearHardening;
//   using MaterialB  = solid_mechanics::J2SmallStrain<Hardening>;

//   Hardening hardening{.sigma_y = 10.0, .Hi = 0.1, .density = 1.0;};
//   Material  mat{
//        .E         = 2.0,  // Young's modulus
//        .nu        = 0.25,   // Poisson's ratio
//        .hardening = hardening,
//        .Hk        = 0.0,  // kinematic hardening constant
//        .density   = 1.0   // mass density
//   };

//   MaterialB::State initial_state{};

  //auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  auto is_in_left = [](std::vector<tensor<double, dim>> coords, int /* attribute */) {
    return average(coords)[0] < 0.5*L;
  };
  Domain left = Domain::ofElements(pmesh, is_in_left);

  auto is_in_right = [=](std::vector<tensor<double, dim>> coords, int attr){ return !is_in_left(coords, attr); };
  Domain right = Domain::ofElements(pmesh, is_in_right);

  solid.setMaterial(mat_left, left);
  solid.setMaterial(mat_right, right);

  constexpr double stress = 1.0;
  Domain end_face = Domain::ofBoundaryElements(pmesh, by_attr<dim>(3));
  solid.setTraction(DependsOn<>{}, [stress](auto, auto n, auto){ return stress*n; }, end_face);

  solid.setDisplacementBCs({2}, [](auto){ return 0.0; }, 1);
  solid.setDisplacementBCs({5}, [](auto){ return 0.0; }, 0);
  solid.setDisplacementBCs({1}, [](auto){ return 0.0; }, 2);

  solid.completeSetup();

  // Perform the quasi-static solve
  solid.advanceTimestep(1.0);
  solid.outputStateToDisk("paraview");

  // Define output functionals for verification

  constexpr double subdomain_volume = 0.5*VOLUME;

  auto average_strain_integrand = [](auto, auto, auto displacement) {
    auto strain = get<1>(displacement);
    return strain[0][0]/subdomain_volume;
  };

  Functional<double(H1<p, dim>)> average_strain_left({&solid.displacement().space()});
  average_strain_left.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand,
    left);

  Functional<double(H1<p, dim>)> average_strain_right({&solid.displacement().space()});
  average_strain_right.AddDomainIntegral(Dimension<dim>{}, DependsOn<0>{}, average_strain_integrand,
    right);

  EXPECT_NEAR(average_strain_left(solid.time(), solid.displacement()), stress/E_left, 1e-10);
  EXPECT_NEAR(average_strain_right(solid.time(), solid.displacement()), stress/E_right, 1e-10);
}

} // namespace serac

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
