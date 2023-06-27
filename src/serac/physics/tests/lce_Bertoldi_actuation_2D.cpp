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
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

#define CUSTOM_SOLVER
// #undef CUSTOM_SOLVER

using namespace serac;

// void pinNode(mfem::ParMesh parMesh, mfem::Array<int> tdof_list)
// {
//   int dim       = 3;

//   SLIC_ERROR_ROOT_IF(
//       pin_coords.size() != dim,
//       axom::fmt::format("Input coordinate is of dimension {} while the solid mechanics module is of dimension {}.",
//                         pin_coords.size(), dim));

//   mfem::ParGridFunction& nodes     = static_cast<mfem::ParGridFunction&>(*parMesh.GetNodes());
//   const int              num_nodes = nodes.Size() / dim;

//   double min_distance = 1.0e6;
//   int    min_node     = -1;

//   for (int i = 0; i < num_nodes; i++) 
//   {
//     if (nodes.ParFESpace()->GetLocalTDofNumber(i) >= 0) 
//     {
//       std::vector<double> node_coords(dim);
//       for (int d = 0; d < dim; d++) 
//       {
//         auto ldof = mfem::Ordering::Map<mfem::Ordering::byNODES>(nodes.FESpace()->GetNDofs(), nodes.FESpace()->GetVDim(), i, d);
//         node_coords[d] = nodes(ldof)
//       }

//       if ( (node_coords[0] <= 0.001) && (node_coords[1] >= 0.45) && (node_coords[1] <= 0.55 ))
//       {
//         min_distance = 1;
//         min_node     = i;
//       }
//     }
//   }

//   // See if this MPI rank contains the closest node
//   auto [num_procs, rank] = getMPIInfo(parMesh.GetComm());

//   // Define an unnamed struct containing the rank and minmum for each MPI process
//   struct {
//     double val;
//     int    rank;
//   } my_min, global_min;

//   // Determine the minimum distance and rank of the closest node
//   my_min.val  = min_distance;
//   my_min.rank = rank;

//   MPI_Reduce(&my_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, parMesh.GetComm());

//   // If this rank contains the closest node, add it's tdofs to the constrained list
//   if (global_min.rank == rank) {
//     for (int d = 0; d < dim; ++d) {
//       auto ldof = mfem::Ordering::Map<mfem::Ordering::byNODES>(nodes.FESpace()->GetNDofs(), nodes.FESpace()->GetVDim(),
//                                                           min_node, d);
//       auto tdof = nodes.ParFESpace()->GetLocalTDofNumber(ldof);
//       tdof_list.Append(tdof);
//     }
//   }

//   return tdof_list;
//   // auto zero_disp = [](const mfem::Vector&, mfem::Vector& out) { out = 0.0; };
//   // disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, zero_disp);
//   // bcs_.addEssential(tdof_list, disp_bdr_coef_, displacement_.space());
// }

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p = 1;
  constexpr int dim       = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store
  // int nElem = 4;
  int xyratio = 10;
  double l = 5.0e0;
  double lx = xyratio*l;
  double ly = l + 0.0*lx;
  // double lz = ly/nElem;

  // auto initial_mesh = mfem::Mesh(mfem::Mesh::MakeCartesian3D( xyratio*nElem, nElem, 1, mfem::Element::HEXAHEDRON, lx, ly, lz));
  
  std::string filename = SERAC_REPO_DIR "/data/meshes/pseudo2DBrick.g";
  auto initial_mesh = buildMeshFromFile(filename);

  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);


#ifdef CUSTOM_SOLVER
  mfem::ParMesh *pmesh = serac::StateManager::setMesh(std::move(mesh));

  // ---------------------
  // Custom solver options
  // ---------------------
  // _custom_solver_start
  auto nonlinear_solver = std::make_unique<mfem::NewtonSolver>(pmesh->GetComm());
  // auto nonlinear_solver = std::make_unique<mfem::KINSolver>(pmesh->GetComm(), KIN_LINESEARCH, true);
  nonlinear_solver->SetPrintLevel(1);
  nonlinear_solver->SetMaxIter(50);
  nonlinear_solver->SetAbsTol(1.0e-10);
  nonlinear_solver->SetRelTol(1.0e-6);

  auto linear_solver = std::make_unique<mfem::HypreGMRES>(pmesh->GetComm());
  linear_solver->SetPrintLevel(0);
  linear_solver->SetMaxIter(1000);
  linear_solver->SetKDim(500);
  linear_solver->SetTol(1.0e-10);

  // preconditioners: HypreILU, HypreBoomerAMG, HypreEuclid (paralle incomplete LU factorization)
  auto preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
  preconditioner->SetPrintLevel(0);
  linear_solver->SetPreconditioner(*preconditioner);

  auto equation_solver = std::make_unique<serac::EquationSolver>(
      std::move(nonlinear_solver), std::move(linear_solver), std::move(preconditioner));

  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      std::move(equation_solver), solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional");
  // _custom_solver_end

#else
  serac::StateManager::setMesh(std::move(mesh));

  // Construct a functional-based solid mechanics solver
  LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 15,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<H1<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional");
#endif

  // Material properties
  double density         = 1.0;
  double young_modulus   = 0.25e6; // 0.25e6; (multiply by 10e-3 to go from SI to [Kg/s/mm])
  double possion_ratio   = 0.48;
  double beta_param      = 5.2e4;  // 5.2e4; (multiply by 10e-3 to go from SI to [Kg/s/mm])
  double max_order_param = 0.4;
  double gamma_angle     = 0.0;
  double eta_angle       = 0.0;

  // Parameter 1
  FiniteElementState orderParam(StateManager::newState(FiniteElementState::Options{.order = p, .name = "orderParam"}));
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "gammaParam"}));
  auto               gammaFunc = [gamma_angle](const mfem::Vector&, double) -> double {
    return gamma_angle;
  };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState        etaParam(StateManager::newState(
      FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "etaParam"}));
  auto                      etaFunc = [eta_angle](const mfem::Vector& /*x*/, double) -> double { return eta_angle; };
  mfem::FunctionCoefficient etaCoef(etaFunc);
  etaParam.project(etaCoef);

  // Set parameters
  constexpr int ORDER_INDEX = 0;
  constexpr int GAMMA_INDEX = 1;
  constexpr int ETA_INDEX   = 2;

  solid_solver.setParameter(ORDER_INDEX, orderParam);
  solid_solver.setParameter(GAMMA_INDEX, gammaParam);
  solid_solver.setParameter(ETA_INDEX, etaParam);

  // Set material
  LiquidCrystalElastomerBertoldi        lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  // solid_solver.setDisplacementBCs({1}, zeroFunc, 2);  // back face z-dir disp = 0
  // solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
  // solid_solver.setDisplacementBCs({5}, zeroFunc, 0);  // left face x-dir disp = 0
  // solid_solver.setDisplacementBCs({5}, [](const mfem::Vector &, mfem::Vector&u){u=0.0;});

  std::set<int> support           = {1};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);
  solid_solver.setDisplacementBCs({2}, zeroFunc, 0);  // left face x-dir disp = 0

  // pin selected dofs
  // mfem::Array<int> dofs;
  // mfem::Array<int> pinned_bc_true_dofs;
  // mfem::GridFunction* nodeCoords = pmesh->GetNodes();
  // mfem::ParFiniteElementSpace *fespace = new mfem::ParFiniteElementSpace(pmesh, pmesh->GetNodes()->OwnFEC());
  // const int num_nodes = nodeCoords->Size() / 3;

  // auto in_range = [&](double x, double y, double /*z*/) {
  //     // if (x <= 0.001*lx && y <= 0.001*ly)
  //   if ( (x <= 0.001*lx) && (y >= 0.5*(ly-lz)) && (y <= 0.5*(ly+lz)) )
  //     {
  //         return 1;
  //     }
  //     return -1;
  // };

  // for (int i = 0; i < num_nodes; i++)
  // {
  //     int local_x_vdof = mfem::Ordering::Map<mfem::Ordering::byNODES>(nodeCoords->FESpace()->GetNDofs(), nodeCoords->FESpace()->GetVDim(), i, 0);
  //     int local_y_vdof = mfem::Ordering::Map<mfem::Ordering::byNODES>(nodeCoords->FESpace()->GetNDofs(), nodeCoords->FESpace()->GetVDim(), i, 1);
  //     int local_z_vdof = mfem::Ordering::Map<mfem::Ordering::byNODES>(nodeCoords->FESpace()->GetNDofs(), nodeCoords->FESpace()->GetVDim(), i, 2);

  //     int true_x_dof_bc = fespace->GetLocalTDofNumber(local_x_vdof); // returns negative if dof is not owned
  //     if(true_x_dof_bc > 0)
  //     {
  //       int true_y_dof_bc = fespace->GetLocalTDofNumber(local_y_vdof); // returns negative if dof is not owned
  //       int true_z_dof_bc = fespace->GetLocalTDofNumber(local_z_vdof); // returns negative if dof is not owned
  //       if ( in_range((*nodeCoords)(local_x_vdof), (*nodeCoords)(local_y_vdof), (*nodeCoords)(local_z_vdof)) == 1 ) 
  //       {
  //           pinned_bc_true_dofs.Append(true_x_dof_bc);
  //           pinned_bc_true_dofs.Append(true_y_dof_bc);
  //           pinned_bc_true_dofs.Append(true_z_dof_bc);
  //       }
  //     }
  // }

  // auto pinned_bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };
  // solid_solver.setDisplacementBCs(pinned_bc_true_dofs, pinned_bc);

  double iniDispVal = 1.0e-6*ly;
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename = "sol_lce_bertoldi_actuation_2D_order_pinned_nodes";
  solid_solver.outputState(outputFilename);

  int num_steps = 10;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;

  for (int i = 0; i < num_steps; i++) {

    t += dt;
    // orderParam = max_order_param * (tmax - t) / tmax;
    orderParam = max_order_param * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputState(outputFilename);

    auto&                 fes             = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    int                   numDofs         = fes.GetNDofs();
    mfem::Vector dispVecX(numDofs);
    dispVecX = 0.0;
    mfem::Vector dispVecY(numDofs);
    dispVecY = 0.0;
    mfem::Vector dispVecZ(numDofs);
    dispVecZ = 0.0;

    for (int k = 0; k < numDofs; k++) {
      dispVecX(k) = displacement_gf(0 * numDofs + k);
      dispVecY(k) = displacement_gf(1 * numDofs + k);
      dispVecZ(k) = displacement_gf(2 * numDofs + k);
    }
    double gblDispXmin, lclDispXmin = dispVecX.Min();
    double gblDispXmax, lclDispXmax = dispVecX.Max();
    double gblDispYmin, lclDispYmin = dispVecY.Min();
    double gblDispYmax, lclDispYmax = dispVecY.Max();
    double gblDispZmin, lclDispZmin = dispVecZ.Min();
    double gblDispZmax, lclDispZmax = dispVecZ.Max();
    MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
                << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
                << std::endl;

      if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        exit(1);
      }
    }
  }

  // Compute sensitivities
  // ---------------------

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjoint_load_form(&solid_solver.displacement().space());
  adjoint_load_form = 1.0;

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the displacement.
  serac::FiniteElementDual              adjoint_load(solid_solver.displacement().space(), "adjoint_load");
  std::unique_ptr<mfem::HypreParVector> assembled_vector(adjoint_load_form.ParallelAssemble());
  adjoint_load = *assembled_vector;

  // Solve the adjoint problem
  solid_solver.solveAdjoint({{"displacement", adjoint_load}});
  solid_solver.outputState(outputFilename);

  if (rank == 0) {
    std::cout << "\n... Solved adjoint problem... " << std::endl;
  }

  // Second forward solve
  // --------------------
  solid_solver.advanceTimestep(dt);

  serac::exitGracefully();
}