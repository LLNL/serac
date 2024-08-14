// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/fit.hpp"
#include "serac/numerics/functional/functional.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"

#include <gtest/gtest.h>

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

int    n = 0;
double t = 0.0;

template <typename output_space>
void stress_extrapolation_test()
{
  int serial_refinement   = 2;
  int parallel_refinement = 0;

  std::string filename = SERAC_REPO_DIR "/data/meshes/notched_plate.mesh";

  auto mesh_ = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  constexpr int p   = 2;
  constexpr int dim = 2;

  using input_space = H1<2, dim>;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store

  std::string mesh_tag{"mesh"};

  auto& pmesh = StateManager::setMesh(std::move(mesh_), mesh_tag);

  LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                           .relative_tol   = 1.0e-12,
                                           .absolute_tol   = 1.0e-12,
                                           .max_iterations = 5000,
                                           .print_level    = 1};

  FiniteElementState sigma_J2(pmesh, output_space{}, "sigma_J2");

  SolidMechanics<p, dim, serac::Parameters<output_space> > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::Off,
      "solid_mechanics", mesh_tag, {"sigma_J2"});

  solid_mechanics::NeoHookean mat{
      1.0,    // density
      100.0,  // bulk modulus
      50.0    // shear modulus
  };

  solid_solver.setMaterial(mat);

  // prescribe small displacement at each hole, pulling the plate apart
  std::set<int> top_hole = {2};
  auto          up       = [](const mfem::Vector&, mfem::Vector& u) -> void { u[1] = 0.01; };
  solid_solver.setDisplacementBCs(top_hole, up);

  std::set<int> bottom_hole = {3};
  auto          down        = [](const mfem::Vector&, mfem::Vector& u) -> void { u[1] = -0.01; };
  solid_solver.setDisplacementBCs(bottom_hole, down);

  auto zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("paraview" + std::to_string(n));

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  auto u = solid_solver.displacement();

  Empty internal_variables{};

  sigma_J2 = fit<dim, output_space(input_space)>(
      [&](double /*t*/, [[maybe_unused]] auto position, [[maybe_unused]] auto displacement_) {
        mat3 du_dx  = to_3x3(get_value(get<1>(displacement_)));
        auto stress = mat(internal_variables, du_dx);
        return tuple{I2(dev(stress)), zero{}};
      },
      pmesh, u);

  solid_solver.setParameter(0, sigma_J2);

  solid_solver.outputStateToDisk("paraview" + std::to_string(n));
  n++;
}

TEST(StressExtrapolation, PiecewiseConstant2D) { stress_extrapolation_test<L2<0> >(); }
TEST(StressExtrapolation, PiecewiseLinear2D) { stress_extrapolation_test<H1<1> >(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
