// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/fit.hpp"
#include "serac/numerics/functional/functional.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

#include <gtest/gtest.h>

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

double t = 0.0;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int p, int dim>
void stress_extrapolation_test(mfem::ParMesh& mesh) {

  double lambda = 100.0; 
  double mu = 100.0; 

  using output_space = L2<p>; 
  using input_space = H1<2,dim>;

  FiniteElementState u(mesh, input_space{}, "displacement");

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      GeometricNonlinearities::Off, "solid_mechanics", mesh_tag);

  solid_mechanics::Neohookean mat{
    1.0,   // density
    100.0, // bulk modulus
    50.0   // shear modulus
  };

  solid_solver.setMaterial(mat);

  // prescribe zero displacement at the supported end of the beam,
  std::set<int> support           = {1};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  // apply a displacement along z to the the tip of the beam
  auto translated_in_z = [](const mfem::Vector&, double t, mfem::Vector& u) -> void {
    u    = 0.0;
    u[2] = t * (t - 1);
  };
  std::set<int> tip = {2};
  solid_solver.setDisplacementBCs(tip, translated_in_z);

  solid_solver.setDisplacement(zero_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("paraview");

  double dt = 1.0;
  solid_solver.advanceTimestep(dt);

  FiniteElementState sigma_01 = fit< dim, output_space(input_space) >([&](double /*t*/, auto /*x*/, auto displacement){ 
    auto du_dx = get<1>(displacement);
    auto stress = lambda * tr(du_dx) * Identity<dim>() + mu * (du_dx + transpose(du_dx));
    return tuple{stress[0][1], zero{}};
  }, mesh, u);

  // TODO:
  // verify sigma_01 := f(x, y)

}

TEST(StressExtrapolation, PiecewiseConstant2D) { stress_extrapolation_test< 0, 2 >(*mesh2D); }
TEST(StressExtrapolation, PiecewiseConstant3D) { stress_extrapolation_test< 0, 3 >(*mesh3D); }

TEST(StressExtrapolation, PiecewiseLinear2D) { stress_extrapolation_test< 1, 2 >(*mesh2D); }
TEST(StressExtrapolation, PiecewiseLinear3D) { stress_extrapolation_test< 1, 3 >(*mesh3D); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
