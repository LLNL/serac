// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>
#include <algorithm>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include <serac/physics/materials/liquid_crystal_elastomer.hpp>

#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/profiling.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "petscmat.h"

using namespace serac;

std::string mesh_path = ".";

auto get_opts(int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                  //.nonlin_solver = NonlinearSolver::NewtonLineSearch, //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewton, //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint, // breaks for snap_cell
                                                  .relative_tol               = abs_tol,
                                                  .absolute_tol               = abs_tol,
                                                  .min_iterations             = 1,
                                                  .max_iterations             = 10000,
                                                  .max_line_search_iterations = 20,
                                                  .print_level                = 1};

  serac::LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::CG,
                                               //.linear_solver  = LinearSolver::PetscGMRES,
                                               //.linear_solver  = LinearSolver::PetscCG,
                                               .preconditioner = Preconditioner::HypreJacobi,
                                               //.preconditioner = Preconditioner::Petsc,
                                               //.petsc_preconditioner = PetscPCType::JACOBI,
                                               //.petsc_preconditioner = PetscPCType::JACOBI_ROWMAX,
                                               //.petsc_preconditioner = PetscPCType::GAMG, 
                                               //.petsc_preconditioner = PetscPCType::HMG,
                                               //.petsc_preconditioner = PetscPCType::LU,
                                               //.petsc_preconditioner = PetscPCType::CHOLESKY,
                                               .relative_tol   = 0.7 * abs_tol,
                                               .absolute_tol   = 0.7 * abs_tol,
                                               .max_iterations = max_iters,
                                               .print_level = 0};

  return std::make_pair(nonlinear_options, linear_options);
 }


void functional_solid_test_nonlinear_buckle()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "logicStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 100;
  int Ny = 50;
  int Nz = 3;

  double Lx = 20.0;
  double Ly = 10.0;
  double Lz = 0.3;

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 2e-2; //0.2e-5;  // 2e-2;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(3*Nx*Ny*Nz, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3, 4, 5}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));
  seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  //seracSolid->setPressure([&](auto, auto) { return -loadMagnitude; }, topSurface);
  seracSolid->completeSetup();
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_buckle");
}


void functional_solid_test_friction_box()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "boxStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 5;
  int Ny = 5;
  int Nz = 5;

  double Lx = 5.0;
  double Ly = 5.0;
  double Lz = 8.3;

  double density       = 1.0;
  double E             = 1000.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 1e-5; //0.2e-5;  // 2e-2;
  double eta = 0.45;
  double mu = 0.3; //3;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(Nx*Ny*Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  double initial_penalty = 1000.0;
  std::array<double,DIM> rigid_velo{0.0, 0.0, 0.0};
  std::array<double,DIM> corner{-0.1, -0.1, -0.1};
  std::array<double,DIM> plane_normal;
  for (size_t i=0; i < DIM; ++i) {
    plane_normal = {};
    plane_normal[i] = 1.0;
    auto lset = std::make_unique<LevelSetPlane<DIM>>(corner, plane_normal);
    auto friction = std::make_unique<NodalFriction<DIM>>(mu, 1e-2, rigid_velo);
    auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset), std::move(friction),
                                                                         "serac_solid_"+std::to_string(i), meshTag,
                                                                         initial_penalty);
    seracSolid->addInequalityConstraint(std::move(constraint));
  }

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int num_time_steps = 1;
  double total_time = 1.0; //num_time_steps; //.0;
  double dt = total_time / num_time_steps;

  const tensor<double, DIM> nx{1.0, 0.0, 0.0};
  const tensor<double, DIM> ny{0.0, 1.0, 0.0};
  const tensor<double, DIM> nz{0.0, 0.0, 1.0};
  //seracSolid->setTraction([&](auto, auto, double t) {
  //  //printf("t = %g\n", t);
  //  auto sideLoad = eta * (t / total_time) * loadMagnitude;
  //  return -loadMagnitude * ny - sideLoad / std::sqrt(2.0) * nx - sideLoad / std::sqrt(2.0) * nz;
  //}, topSurface);

  // fix displacement on top surface
  seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
    u[2] = -0.15;
    u[1] = -0.05;
  });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  for (int i=0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_friction_box");
  }
}

void functional_solid_test_buckle_ball()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "buckleBallStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  int Nx = 5;
  int Ny = 5;
  int Nz = 15;

  double Lx = 1.0;
  double Ly = 1.0;
  double Lz = 30.0;

  double density       = 1.0;
  double E             = 1000.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 1e-5; //0.2e-5;  // 2e-2;
  double eta = 0.45;
  double mu = 0.0; //3;

  std::string    meshTag = "mesh";
  mfem::Mesh     mesh    = mfem::Mesh::MakeCartesian3D(Nx, Ny, Nz, mfem::Element::HEXAHEDRON, Lx, Ly, Lz);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(Nx*Ny*Nz, 1e-9);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  double initial_penalty = 1000.0;
  std::array<double,DIM> rigid_velo{0.0, 0.0, 0.0};
  std::array<double,DIM> corner{-0.1, -0.1, -0.1};
  std::array<double,DIM> plane_normal;
  for (size_t i=0; i < DIM; ++i) {
    plane_normal = {};
    plane_normal[i] = 1.0;
    auto lset = std::make_unique<LevelSetPlane<DIM>>(corner, plane_normal);
    auto friction = std::make_unique<NodalFriction<DIM>>(mu, 1e-3, rigid_velo);
    auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset), std::move(friction),
                                                                         "serac_solid_"+std::to_string(i), meshTag,
                                                                         initial_penalty);
    seracSolid->addInequalityConstraint(std::move(constraint));
  }

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(6));

  int num_time_steps = 1;
  double total_time = 1.0; //num_time_steps; //.0;
  double dt = total_time / num_time_steps;

  const tensor<double, DIM> nx{1.0, 0.0, 0.0};
  const tensor<double, DIM> ny{0.0, 1.0, 0.0};
  const tensor<double, DIM> nz{0.0, 0.0, 1.0};
  //seracSolid->setTraction([&](auto, auto, double t) {
  //  //printf("t = %g\n", t);
  //  auto sideLoad = eta * (t / total_time) * loadMagnitude;
  //  return -loadMagnitude * ny - sideLoad / std::sqrt(2.0) * nx - sideLoad / std::sqrt(2.0) * nz;
  //}, topSurface);

  // fix displacement on top surface
  seracSolid->setDisplacementBCs({6}, [](const mfem::Vector&, mfem::Vector& u) {
    u = 0.0;
    u[2] = -0.15;
    u[1] = -0.05;
  });

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  //seracSolid->advanceTimestep(0.0);
  //seracSolid->outputStateToDisk("paraview_friction_box");
  for (int i=0; i < num_time_steps; ++i) {
    seracSolid->advanceTimestep(dt);
    seracSolid->outputStateToDisk("paraview_friction_box");
  }
}

void functional_solid_test_nonlinear_arch()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "archStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.33;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  //double loadMagnitude = 1.2e-2; //0.2e-5;  // 2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "arch.g";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  //auto lset = std::make_unique<LevelSetPlane<DIM>>(std::array<double,DIM>{0.0, -4.0, 0.0}, 
  //                                                 std::array<double,DIM>{0.0, 1.0, 0.0});
  double initial_penalty = 0.1;
  std::array<double,DIM> ball_velo{10.0, 0.0, 0.0};
  auto lset = std::make_unique<LevelSetSphere<DIM>>(std::array<double,DIM>{-5.0, -5.0, 1.5}, 2.2, ball_velo);
  auto friction = std::make_unique<NodalFriction<DIM>>(0.5, 1e-3, ball_velo);
  auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::move(lset), std::move(friction),
                                                                       "serac_solid", meshTag, initial_penalty);
  seracSolid->addInequalityConstraint(std::move(constraint));

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({2, 3}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(1));
  //seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);
  //seracSolid->setPressure([&](auto, auto) { return loadMagnitude; }, topSurface);

  seracSolid->completeSetup();

  int num_steps = 16;

  seracSolid->outputStateToDisk("paraview_arch");
  for (int step=0; step < num_steps; ++step) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    std::cout << "outputting at step " << step+1 << std::endl;
    seracSolid->outputStateToDisk("paraview_arch");
  }
  
}


void functional_solid_test_nonlinear_snap_cell()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "snapCellStore");

  static constexpr int ORDER{2};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 4.e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "snap_cell.exo";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  seracSolid->setDisplacementBCs({2,3,5,6}, [](const mfem::Vector&) { return 0.0; }, 0);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(7));
  seracSolid->setPressure([&](auto, auto time) { return time * loadMagnitude; }, topSurface);
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_snap_cell");
  int num_steps = 10;
  for (int i=0; i < num_steps; ++i) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    std::cout << "outputing at step " << i << "/" << num_steps << std::endl;
    seracSolid->outputStateToDisk("paraview_snap_cell");
  }
}


void functional_solid_test_nonlinear_snap_chain()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "snapChainStore");

  static constexpr int ORDER{1};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 1.2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = mesh_path + "snap_chain.exo";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  // set linear elastic material
  serac::solid_mechanics::NeoHookean material{density, bulkMod, shearMod};
  // serac::solid_mechanics::StVenantKirchhoff material{density, bulkMod, shearMod};
  seracSolid->setMaterial(serac::DependsOn<>{}, material);

  // fix displacement on side surface
  seracSolid->setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });
  seracSolid->setDisplacementBCs({2,4}, [](const mfem::Vector&) { return 0.0; }, 0);

  serac::Domain topSurface = serac::Domain::ofBoundaryElements(*meshPtr, serac::by_attr<DIM>(5));
  seracSolid->setPressure([&](auto, auto time) { return time * loadMagnitude; }, topSurface);
  // seracSolid->setTraction([&](auto, auto n, auto) { return -loadMagnitude * n; }, topSurface);

  seracSolid->completeSetup();

  seracSolid->outputStateToDisk("paraview_snap_chain");
  int num_steps = 10;
  for (int i=0; i < num_steps; ++i) {
    seracSolid->advanceTimestep(1.0 / num_steps);
    seracSolid->outputStateToDisk("paraview_snap_chain");
  }
}


TEST(SolidMechanics, nonlinear_solve_buckle) { functional_solid_test_nonlinear_buckle(); }
TEST(SolidMechanics, nonlinear_solve_arch) { functional_solid_test_nonlinear_arch(); }
TEST(SolidMechanics, nonlinear_solve_snap_chain) { functional_solid_test_nonlinear_snap_chain(); }
TEST(SolidMechanics, nonlinear_solve_snap_cell) { functional_solid_test_nonlinear_snap_cell(); }
TEST(SolidMechanics, nonlinear_solve_friction_box) { functional_solid_test_friction_box(); }

class InputParser
{
public:
  InputParser(int& argc, char** argv){
    for (int i=1; i < argc; ++i) {
      this->tokens.push_back(std::string(argv[i]));
    }
  }
  std::string getCmdOption(const std::string& option) const {
    std::vector<std::string>::const_iterator itr;
    itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }
  bool cmdOptionExists(const std::string& option) const {
    return std::find(this->tokens.begin(), this->tokens.end(), option)
            != this->tokens.end();
  }
private:
  std::vector <std::string> tokens;
};

int main(int argc, char* argv[])
{
  InputParser parser(argc, argv);
  auto filename = parser.getCmdOption("-p");
  if (!filename.empty()) {
    mesh_path = filename;
  }

  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
