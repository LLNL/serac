// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/liquid_crystal_elastomer.hpp"

using namespace serac;

#define PERIODIC_MESH
// #undef PERIODIC_MESH

const static int problemID = 1;

using serac::solid_mechanics::default_static_options;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 2;
  double lx = 3.0e-3, ly = 3.0e-3, lz = 0.25e-3;
  auto initial_mesh = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4*nElem, 4*nElem, nElem, mfem::Element::HEXAHEDRON, lx, ly, lz));

#ifdef PERIODIC_MESH

  // Create translation vectors defining the periodicity
  mfem::Vector x_translation({lx, 0.0, 0.0});
  // mfem::Vector y_translation({0.0, ly, 0.0});
  // std::vector<mfem::Vector> translations = {x_translation, y_translation};
  std::vector<mfem::Vector> translations = {x_translation};
  double tol = 1e-6;

  std::vector<int> periodicMap = initial_mesh.CreatePeriodicVertexMapping(translations, tol);

  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  auto periodic_mesh = mfem::Mesh::MakePeriodic(initial_mesh, periodicMap);
  auto mesh = mesh::refineAndDistribute(std::move(periodic_mesh), serial_refinement, parallel_refinement);

#else

  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

#endif

  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  IterativeSolverOptions default_linear_options = {.rel_tol     = 1.0e-6,
                                                       .abs_tol     = 1.0e-16,
                                                       .print_level = 0,
                                                       .max_iter    = 600,
                                                       .lin_solver  = LinearSolver::GMRES,
                                                       .prec        = HypreBoomerAMGPrec{}};
  NonlinearSolverOptions default_nonlinear_options = {
    .rel_tol = 1.0e-6, .abs_tol = 1.0e-10, .max_iter = 6, .print_level = 1};
  SolidMechanics<p, dim, Parameters< H1<p>, L2<p>, L2<p> > > solid_solver({default_linear_options, default_nonlinear_options}, GeometricNonlinearities::Off,
                                       "lce_solid_functional");

  // Material properties
  double density = 1.0;
  double young_modulus = 0.4;
  double possion_ratio = 0.49;
  double beta_param = 0.041;
  double max_order_param = 0.1;
  double gamma_angle = 0.0;
  double eta_angle = 0.0;

  switch (problemID)
  {
    case 0:
      gamma_angle = 0.0;
      eta_angle = 0.0;
      break;
    case 1:
      gamma_angle = M_PI_2;
      eta_angle = 0.0;
      break;
    case 2:
      gamma_angle = 0.0;
      eta_angle = M_PI_2;
      break;
    case 3:
      gamma_angle = 0.5 * M_PI_2;
      eta_angle = 0.0;
      break;
    case 4:
      gamma_angle = 0.5 * M_PI_2;
      eta_angle = M_PI_2;
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  // Parameter 1
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  orderParam = max_order_param;

  // Parameter 2
  auto fec = std::unique_ptr< mfem::FiniteElementCollection >(new mfem::L2_FECollection(p, dim));
  FiniteElementState gammaParam(StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(fec), .name = "gammaParam"}));
  auto gammaFunc = [gamma_angle](const mfem::Vector& /*x*/, double) -> double { return gamma_angle; };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  auto fec2 = std::unique_ptr< mfem::FiniteElementCollection >(new mfem::L2_FECollection(p, dim));
  FiniteElementState etaParam(StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(fec2), .name = "etaParam"}));
  auto etaFunc = [eta_angle](const mfem::Vector& /*x*/, double) -> double { return eta_angle; };
  mfem::FunctionCoefficient etaCoef(etaFunc);
  etaParam.project(etaCoef);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;

  solid_solver.setParameter(orderParam, ORDER_INDEX);
  solid_solver.setParameter(gammaParam, GAMMA_INDEX);
  solid_solver.setParameter(etaParam, ETA_INDEX);

  // Set material
  LiqCrystElast_Bertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);
  LiqCrystElast_Bertoldi::State initial_state{};

  auto param_data = solid_solver.createQuadratureDataBuffer(initial_state);
  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat, param_data);

  // Boundary conditions:
  // Prescribe zero displacement at the supported end of the beam
  std::set<int> support           = {2};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  double iniDispVal =  5.0e-6;
  if (problemID==4)
  {
    iniDispVal =  5.0e-8;
  }
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  int num_steps = 4;

  std::string outputFilename;
  switch (problemID)
  {
    case 0:
      outputFilename = "sol_lce_bertoldi_rect_lattice_gamma_00_eta_00";
      break;
    case 1:
      outputFilename = "sol_lce_bertoldi_rect_lattice_gamma_90_eta_00";
      break;
    case 2:
      outputFilename = "sol_lce_bertoldi_rect_lattice_gamma_00_eta_90";
      break;
    case 3:
      outputFilename = "sol_lce_bertoldi_rect_lattice_gamma_45_eta_00";
      break;
    case 4:
      outputFilename = "sol_lce_bertoldi_rect_lattice_gamma_45_eta_90";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }
  solid_solver.outputState(outputFilename);
 
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
double gblDispYmin;

  for (int i = 0; i < num_steps; i++) 
  {
    if(rank==0)
    {
      std::cout 
      << "\n\n............................"
      << "\n... Entering time step: "<< i + 1 << " (/" << num_steps << ")"
      << "\n............................\n"
      << "\n... Using order parameter: "<< max_order_param * (tmax - t) / tmax
      << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle
      << std::endl;
    }

    // FiniteElementState &displacement = solid_solver.displacement();
    auto &fes = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    mfem::Vector dispVecY(fes.GetNDofs()); dispVecY = 0.0;

    for (int k = 0; k < fes.GetNDofs(); k++) 
    {
      dispVecY(k) = displacement_gf(3*k+1);
    }

    double lclDispYmin = dispVecY.Min();
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if(rank==0)
    {
      std::cout 
      <<"... Min Y displacement: " << gblDispYmin
      << std::endl;
    }

    t += dt;
    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

    orderParam = max_order_param * (tmax - t) / tmax;
  }

EXPECT_NEAR(gblDispYmin, -1.51674e-05, 1.0e-8);

  MPI_Finalize();
}