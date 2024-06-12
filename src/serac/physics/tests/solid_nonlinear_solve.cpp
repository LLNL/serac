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

std::string path = ".";

auto get_opts(int max_iters, double abs_tol = 1e-9)
{
  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                  //.nonlin_solver = NonlinearSolver::NewtonLineSearch, //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewton, //LineSearch,
                                                  //.nonlin_solver = NonlinearSolver::PetscNewtonCriticalPoint, // breaks for snap_cell
                                                  .relative_tol               = 1000 * abs_tol,
                                                  .absolute_tol               = abs_tol,
                                                  .min_iterations             = 0,
                                                  .max_iterations             = 10000,
                                                  .max_line_search_iterations = 20,
                                                  .print_level                = 1};

  serac::LinearSolverOptions linear_options = {//.linear_solver  = LinearSolver::CG,
                                               //.linear_solver  = LinearSolver::PetscGMRES,
                                               .linear_solver  = LinearSolver::PetscCG,
                                               //.preconditioner = Preconditioner::HypreJacobi,
                                               .preconditioner = Preconditioner::Petsc,
                                               .petsc_preconditioner = PetscPCType::JACOBI,
                                               //.petsc_preconditioner = PetscPCType::JACOBI_ROWMAX,
                                               //.petsc_preconditioner = PetscPCType::GAMG, 
                                               //.petsc_preconditioner = PetscPCType::HMG,
                                               //.petsc_preconditioner = PetscPCType::LU,
                                               //.petsc_preconditioner = PetscPCType::CHOLESKY,
                                               .relative_tol   = 500 * abs_tol,
                                               .absolute_tol   = 0.5 * abs_tol,
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

  double Lx = 20.0;  // in
  double Ly = 10.0;  // in
  double Lz = 0.3;   // in

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.34;
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
  double loadMagnitude = 1.2e-2; //0.2e-5;  // 2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = path + "arch.g";

  auto initial_mesh = serac::buildMeshFromFile(input_file_name);
  auto           pmesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, initial_mesh);
  mfem::ParMesh* meshPtr = &serac::StateManager::setMesh(std::move(pmesh), meshTag);

  // solid mechanics
  using seracSolidType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<>>;

  auto [nonlinear_options, linear_options] = get_opts(50000, 1e-8);

  auto seracSolid = std::make_unique<seracSolidType>(
      nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options,
      serac::GeometricNonlinearities::On, "serac_solid", meshTag, std::vector<std::string>{});

  auto constraint = std::make_unique<InequalityConstraint<ORDER, DIM>>(std::make_unique<LevelSetPlane<DIM>>(
                                                                         std::array<double,DIM>{0.0, -4.0, 0.0}, 
                                                                         std::array<double,DIM>{0.0, 1.0, 0.0}), 
                                                                       "serac_solid", meshTag);
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
  seracSolid->advanceTimestep(1.0);

  seracSolid->outputStateToDisk("paraview_arch");
}


void functional_solid_test_nonlinear_snap_cell()
{
  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "snap_cell");

  static constexpr int ORDER{2};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 4.e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = path + "snap_cell.exo";

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
  serac::StateManager::initialize(datastore, "snap_chain");

  static constexpr int ORDER{1};
  static constexpr int DIM{2};

  double density       = 1.0;
  double E             = 1.0;
  double v             = 0.49;
  double bulkMod       = E / (3. * (1. - 2. * v));
  double shearMod      = E / (2. * (1. + v));
  double loadMagnitude = 1.2e-2;

  std::string meshTag = "mesh";
  std::string input_file_name = path + "snap_chain.exo";

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

// LCE problems

static constexpr int problemID = 4;
static constexpr int maxTimeSteps = 4;
static constexpr double deltaTime = 1.0/maxTimeSteps;
static constexpr double timeDecrRat = 0.9;
static constexpr double timeIncrRat = 1.0 + 0.0*1.0/timeDecrRat;

auto create_lce_solid_mechanics(std::unique_ptr<serac::EquationSolver> eq_solver, std::string lido_mesh_tag)
{
  static constexpr int ORDER{1};
  static constexpr int DIM{3};

  // Material properties and parameters
  double density         = 1.0;
  double shear_mod       = 4476171.852; // 3.113e4; //  young_modulus_ / 2.0 / (1.0 + poisson_ratio_);
  double ini_order_param = 0.28544; // 0.40;
  double min_order_param = ini_order_param; // 0.001;
  double omega_param     = 0.1151; // 1.0e2;
  double bulk_mod        = 1.0e3*shear_mod;
  double lx              = 1e-3;

  using L2_FES = serac::L2<0>;
  using H1_FES = serac::H1<ORDER>;
  //using shapeFES_H1 = serac::H1<ORDER, DIM>;

  using basePhysicsType = serac::SolidMechanics<ORDER, DIM, serac::Parameters<H1_FES, L2_FES, L2_FES>>;
  std::vector<std::string> parameter_names{"order", "gamma", "eta"};

  std::unique_ptr<basePhysicsType> seracLCE = std::make_unique<basePhysicsType>(std::move(eq_solver),
      serac::solid_mechanics::default_quasistatic_options, serac::GeometricNonlinearities::On, "serac_LCE", lido_mesh_tag,
      parameter_names);

  // -----------------------------------
  // Geometry-based parameter assignment
  // -----------------------------------
  double latticeHeight = 0.0;
  switch (problemID) {
    case 0: // 19.4x12x9.63
    case 1: // 16.94x12x9.63
    case 2: // 15.7x12x9.63
    { latticeHeight = 12.0e-3; break; }
    case 3: // 13.5x12x9.63
    { latticeHeight = 12.2e-3; break; }
    case 4: // 13.5x9.63x12
    { latticeHeight = 12.0e-3; break; }
    default:
    {
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
    }
  }

  auto is_on_top = [=](const mfem::Vector& x) {
    bool tag = false;
    double compressionEps = 5.0e-4; // 2.5e-5; // 0.01*latticeHeight; 
    if (x(1)>(latticeHeight-compressionEps)) { tag = true; }
    return tag;
  };

  auto oddLogPileLevel = [](const mfem::Vector& x, const double levelEps = 0.0) -> bool {
    if(x[2]<2.0e-4 + levelEps ||
       (x[2]>4.0e-4 - levelEps && x[2]<6.0e-4 + levelEps) || 
       (x[2]>8.0e-4 - levelEps && x[2]<10.0e-4 + levelEps) || 
       (x[2]>12.0e-4 - levelEps && x[2]<14.0e-4 + levelEps) || 
       (x[2]>16.0e-4 - levelEps && x[2]<18.0e-4 + levelEps) || 
       (x[2]>20.0e-4 - levelEps && x[2]<22.0e-4 + levelEps) || 
       (x[2]>24.0e-4 - levelEps && x[2]<26.0e-4 + levelEps) || 
       (x[2]>28.0e-4 - levelEps && x[2]<30.0e-4 + levelEps) || 
       (x[2]>32.0e-4 - levelEps && x[2]<34.0e-4 + levelEps) || 
       (x[2]>36.0e-4 - levelEps && x[2]<38.0e-4 + levelEps) || 
       (x[2]>40.0e-4 - levelEps && x[2]<42.0e-4 + levelEps) || 
       (x[2]>44.0e-4 - levelEps && x[2]<46.0e-4 + levelEps) || 
       (x[2]>48.0e-4 - levelEps && x[2]<50.0e-4 + levelEps) || 
       (x[2]>52.0e-4 - levelEps && x[2]<54.0e-4 + levelEps) || 
       (x[2]>56.0e-4 - levelEps && x[2]<58.0e-4 + levelEps) || 
       (x[2]>60.0e-4 - levelEps && x[2]<62.0e-4 + levelEps) || 
       (x[2]>64.0e-4 - levelEps && x[2]<66.0e-4 + levelEps) || 
       (x[2]>68.0e-4 - levelEps && x[2]<70.0e-4 + levelEps) || 
       (x[2]>72.0e-4 - levelEps && x[2]<74.0e-4 + levelEps) || 
       (x[2]>76.0e-4 - levelEps && x[2]<78.0e-4 + levelEps) || 
       (x[2]>80.0e-4 - levelEps && x[2]<82.0e-4 + levelEps) || 
       (x[2]>84.0e-4 - levelEps && x[2]<86.0e-4 + levelEps) || 
       (x[2]>88.0e-4 - levelEps && x[2]<90.0e-4 + levelEps) || 
       (x[2]>92.0e-4 - levelEps && x[2]<94.0e-4 + levelEps)
      )
    {
      return true;
    }
    return false;
  };

  // Gamma angle function
  // --------------------
  
  double levelEps = 0.1e-4;
  auto gammaFunc = [=](const mfem::Vector& x, double) -> double {
    double alignmentAngle = 0.0;
    double strutEps = 0.1e-4;
    switch (problemID) {
      case 0:
      {
        if(oddLogPileLevel(x, levelEps))
        {
          // vertical walls
          if (x[0]<=0.6e-3 + strutEps || x[0]>=18.8e-3 - strutEps
          || (x[0]>=4.0e-3 - strutEps && x[0]<=4.3e-3 + strutEps) 
          || (x[0]>=7.7e-3 - strutEps && x[0]<=8.0e-3 + strutEps) 
          || (x[0]>=11.4e-3 - strutEps && x[0]<=11.7e-3 + strutEps) 
          || (x[0]>=15.1e-3 - strutEps && x[0]<=15.4e-3 + strutEps) ) {
            alignmentAngle = M_PI_2;
          }
          else
          {
              alignmentAngle = 0.0 * M_PI_2;
          }
        } // horizontal incline (excluding vertical walls)
        
        if(!oddLogPileLevel(x, -levelEps))
        { 
          // horizontal walls
          if (x[1]<=0.6e-3 + strutEps || x[1]>=11.4e-3 - strutEps
          || (x[1]>=4.0e-3 - strutEps && x[1]<=4.3e-3 + strutEps) 
          || (x[1]>=7.7e-3 - strutEps && x[1]<=8.0e-3 + strutEps) ) {
            alignmentAngle = 0.0 * M_PI_2;
          }
          else
          {
              alignmentAngle = M_PI_2;
          }
        }
        break;
      }

      case 1:
      { // vertical walls
        if((x[0]<0.3e-3 || x[0]>=16.65e-3)
        || (x[0]>=0.85e-3 && x[0]<=1.20e-3) // || (x[0]>=0.85e-3 && x[0]<=1.15e-3) 
        || (x[0]>=3.35e-3 && x[0]<=3.65e-3) 
        || (x[0]>=5.85e-3 && x[0]<=6.15e-3) 
        || (x[0]>=8.30e-3 && x[0]<=8.65e-3) // || (x[0]>=8.35e-3 && x[0]<=8.65e-3) 
        || (x[0]>=10.75e-3 && x[0]<=11.15e-3) 
        || (x[0]>=13.25e-3 && x[0]<=13.55e-3) 
        || (x[0]>=15.70e-3 && x[0]<=16.05e-3)  ) { // || (x[0]>=15.75e-3 && x[0]<=16.05e-3)  ) { 
          if(oddLogPileLevel(x, levelEps))
          {
            alignmentAngle = M_PI_2;
          }
          else
          {
            if( (x[0]<=0.5e-3 && x[1]>1.5e-3) ||
                (x[0]>=16.35e-3 && x[1]<10.0e-3)  )
            {
              alignmentAngle = M_PI_2;
            }
          }
        } // horizontal incline (excluding vertical walls)
        else { alignmentAngle = 0.0; }
        break;
      }

      case 2:
      { // vertical walls
        if((x[0]<0.6e-3 || x[0]>=15.05e-3)
        || (x[0]>=2.15e-3 && x[0]<=2.45e-3) 
        || (x[0]>=4.00e-3 && x[0]<=4.3e-3) 
        || (x[0]>=5.85e-3 && x[0]<=6.15e-3) 
        || (x[0]>=7.70e-3 && x[0]<=8.00e-3) 
        || (x[0]>=9.55e-3 && x[0]<=9.85e-3) 
        || (x[0]>=11.375e-3 && x[0]<=11.675e-3) 
        || (x[0]>=13.25e-3 && x[0]<=13.55e-3) ) { 
          if(oddLogPileLevel(x, levelEps))
          {
            alignmentAngle = M_PI_2;
          }
          else
          {
            if( (x[0]<=0.5e-3 && x[1]>1.0e-3) ||
                (x[0]>=15.05e-3 && x[1]<11.0e-3)  )
            {
              alignmentAngle = M_PI_2;
            }
          }
        }
        else { alignmentAngle = 0.0; }
        break;
      }

      case 3:
      { // vertical walls
        if((x[0]<0.35e-3 || x[0]>=13.25e-3)
        || (x[0]>=0.7e-3 && x[0]<=1.0e-3) 
        || (x[0]>=1.4e-3 && x[0]<=1.7e-3) 
        || (x[0]>=2.1e-3 && x[0]<=2.4e-3) 
        || (x[0]>=2.8e-3 && x[0]<=3.1e-3) 
        || (x[0]>=3.5e-3 && x[0]<=3.8e-3) 
        || (x[0]>=4.2e-3 && x[0]<=4.5e-3) 
        || (x[0]>=4.9e-3 && x[0]<=5.2e-3) 
        || (x[0]>=5.6e-3 && x[0]<=5.9e-3) 
        || (x[0]>=6.3e-3 && x[0]<=6.6e-3) 
        || (x[0]>=7.0e-3 && x[0]<=7.3e-3) 
        || (x[0]>=7.7e-3 && x[0]<=8.0e-3) 
        || (x[0]>=8.4e-3 && x[0]<=8.7e-3) 
        || (x[0]>=9.1e-3 && x[0]<=9.4e-3) 
        || (x[0]>=9.8e-3 && x[0]<=10.1e-3) 
        || (x[0]>=10.5e-3 && x[0]<=10.8e-3) 
        || (x[0]>=11.2e-3 && x[0]<=11.5e-3) 
        || (x[0]>=11.9e-3 && x[0]<=12.2e-3) 
        || (x[0]>=12.6e-3 && x[0]<=12.9e-3)) { 
          if(oddLogPileLevel(x, 0.75*levelEps))
          {
            alignmentAngle = M_PI_2;
          }
          else
          {
            if( (x[0]<=0.35e-3 && x[1]>0.5e-3 && x[1]<11.5e-3) ||
                (x[0]>=13.25e-3)  )
            {
              alignmentAngle = M_PI_2;
            }
          }
          // if(!oddLogPileLevel(x, levelEps))
          // {
          //   alignmentAngle = 0.0*M_PI_2;
          // }
          // else
          // {
          //   if( (x[0]<=0.35e-3 && x[1]>0.5e-3 && x[1]<11.5e-3) ||
          //       (x[0]>=13.25e-3)  )
          //   {
          //     alignmentAngle = 0.0*M_PI_2;
          //   }
          // }
        }
        else { alignmentAngle = 0.0; }
        break;
      }

      case 4:
      { // horizontal walls
        if(x[1]<0.3e-3 || x[1]>=11.7e-3) { alignmentAngle = 0.0;  }
        else { alignmentAngle = M_PI_2; }
        break;
      }
      
      default:
      { std::cout << "...... Wrong problem ID ......" << std::endl; exit(0); }
      break;
    }

    return alignmentAngle;
  };

 // Boundray and initial conditions
 // -------------------------------

  double eps = 2.5e-5;

  // Function to identify left face
  auto is_on_left = [=](const mfem::Vector& x) {
    if (x(0) < eps) { return true; }
    return false;
  };

  // Function to identify bottom face
  auto is_on_bottom = [=](const mfem::Vector& x) {
    if (x(1) < eps) { return true; }
    return false;
  };

  // Function to identify back face
  auto is_on_back = [=](const mfem::Vector& x) {
    if (x(2) < eps) { return true; }
    return false;
  };

  // Function to identify small region in the middle of the domain (different for all samples)
  auto is_on_Xcoord_fixed_region = [=](const mfem::Vector& x) {
    bool tag = false;
    if (x(1) < eps) {
      switch (problemID) {
        case 0: 
        { if ((x(0)>9.7e-3-eps) && (x(0)<9.7e-3+eps)) { tag = true; } break; }
        case 1: 
        { if ((x(0)>8.475e-3-eps) && (x(0)<8.475e-3+eps)) { tag = true; } break; }
        case 2:
        { if ((x(0)>7.825e-3-eps) && (x(0)<7.825e-3+eps)) { tag = true; } break; }
        case 3:
        { if ((x(0)>6.75e-3-eps) && (x(0)<6.75e-3+eps)) { tag = true; } break; }
        case 4:
        { if ((x(0)>6.0e-3-eps) && (x(0)<6.0e-3+eps)) { tag = true; } break; }
        default:
        {
          std::cout << "...... Wrong problem ID ......" << std::endl;
          exit(0);
        }
      }
    }
    return tag;
  };

  // ------------
  // LCE Material
  // ------------

  constexpr int ORDER_INDEX = 0;
  serac::FiniteElementState orderParam(seracLCE->parameter(ORDER_INDEX));
  auto orderFunc = [=](const mfem::Vector&, double t) -> double {
      return min_order_param + (ini_order_param - min_order_param) * (timeIncrRat - t) / timeIncrRat;
  };
  mfem::FunctionCoefficient orderCoef(orderFunc);
  orderParam.project(orderCoef);

  // Varying gamma angle
  constexpr int GAMMA_INDEX = 1;
  serac::FiniteElementState gammaParam(seracLCE->parameter(GAMMA_INDEX));
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Constant eta param
  constexpr int ETA_INDEX = 2;
  serac::FiniteElementState etaParam(seracLCE->parameter(ETA_INDEX));
  etaParam = 0.0;

  // Generate order parameter
  seracLCE->setParameter(ORDER_INDEX, orderParam);
  // Generate gamma Parameter
  seracLCE->setParameter(GAMMA_INDEX, gammaParam);
  // Generate eta parameter
  seracLCE->setParameter(ETA_INDEX, etaParam);

  // Generate material
  serac::LiquidCrystalElastomerZhang LCE_material(density, shear_mod, ini_order_param, omega_param, bulk_mod);
  seracLCE->setMaterial(serac::DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX> {}, LCE_material);

  // Set Dirichlet conditions
  auto zero_scalar   = [](const mfem::Vector&) { return 0.0; };
  seracLCE->setDisplacementBCs(is_on_back, zero_scalar, 2);
  seracLCE->setDisplacementBCs(is_on_bottom, zero_scalar, 1);
  if(problemID==4){ 
    seracLCE->setDisplacementBCs(is_on_left, zero_scalar, 0);
  }
  else
  {
    seracLCE->setDisplacementBCs(is_on_Xcoord_fixed_region, zero_scalar, 0);
  }

  // Displacement-based compression
  double targetDisp = -0.005; // -0.010 * latticeHeight;
  auto scalar_offset = [=](const mfem::Vector&, double t) { return targetDisp*t; };
  seracLCE->setDisplacementBCs(is_on_top, scalar_offset, 1);

  // Set initial displacement
  auto ini_displacement = [](const mfem::Vector& x, mfem::Vector& u) -> void {
    //  u = 0.0000000001;
    u[0] = 0.0001 * x[0];
    u[1] = 0.0001 * x[1];
    u[2] = 0.0001 * x[2];
  };
  seracLCE->setDisplacement(ini_displacement);

  // Set load (if any)
  double loadVal = 0.0*0.0e4;
  seracLCE->setTraction([&loadVal, lx](auto x, auto /*n*/, auto /*t*/) {
    return serac::tensor<double, 3>{loadVal * (x[0] > 0.975 * lx), 0.0, 0.0};
  });

  // Let serac now you are done configuring
  seracLCE->completeSetup();

  return seracLCE;
}


void functional_solid_test_nonlinear_lce()
{
  // (non-dimensional/scaled) time variables
  int plotIter   = 1;

  // initialize serac
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "lceStore");

  std::string inputFilename;
  switch (problemID) {
    case 4:
      inputFilename = path + "logpile_solid_medium.g";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }
  auto initial_mesh = serac::buildMeshFromFile(inputFilename);

  int serial_refinement   = 0; 
  int parallel_refinement = 0;
  if(problemID==4){ parallel_refinement=0; }
  auto mesh = serac::mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  std::cout << "... Successfully loaded mesh." << std::endl;

  // register mesh with serac and lido
  std::string lido_mesh_tag = "pmesh";
  mfem::ParMesh *pmesh = &serac::StateManager::setMesh(std::move(mesh), lido_mesh_tag);

  auto [nonlinear_options, linear_options] = get_opts(3*480000, 1e-10);
  auto eq_solver = std::make_unique<serac::EquationSolver>(nonlinear_options, linear_options, pmesh->GetComm());

  std::cout << "... Made equation solver\n" << std::endl;

  // ----------------------
  // LCE solver definitions
  // ----------------------

  auto seracLCE = create_lce_solid_mechanics(std::move(eq_solver), lido_mesh_tag);

  for (int n=0; n < maxTimeSteps; ++n) {
    bool didPlot = false;
    if (n%plotIter == 0) {
      std::cout << "writing output at step " << n << std::endl;
      seracLCE->outputStateToDisk("paraview_lce");
      didPlot = true;
    }
    seracLCE->advanceTimestep(deltaTime);
    if (!didPlot && n==maxTimeSteps-1) {
      std::cout << "writing output at final step " << n << std::endl;
      seracLCE->outputStateToDisk("paraview_lce");
    }
  }

}

TEST(SolidMechanics, nonlinear_solve_buckle) { functional_solid_test_nonlinear_buckle(); }
TEST(SolidMechanics, nonlinear_solve_lce) { functional_solid_test_nonlinear_lce(); }
TEST(SolidMechanics, nonlinear_solve_arch) { functional_solid_test_nonlinear_arch(); }
TEST(SolidMechanics, nonlinear_solve_snap_chain) { functional_solid_test_nonlinear_snap_chain(); }
TEST(SolidMechanics, nonlinear_solve_snap_cell) { functional_solid_test_nonlinear_snap_cell(); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);

  return result;
}
