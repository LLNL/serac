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

using namespace serac;

const static int problemID = 2;

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p = 1;
  constexpr int dim      = 3;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // ---------------------------
  // ---------------------------
  // Initial run (original mesh)
  // ---------------------------
  // ---------------------------

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Construct the appropriate dimension mesh and give it to the data store

  std::string   filename = SERAC_REPO_DIR "/data/meshes/reEntrantHoneycomb_3D_2x1_no_border.g";
  auto initial_mesh = buildMeshFromFile(filename);
  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh}"};
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a functional-based solid mechanics solver
  LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};

  // LinearSolverOptions linear_options = {.linear_solver  = LinearSolver::GMRES,
  //                                                     .preconditioner = Preconditioner::HypreAMG,
  //                                                     .relative_tol   = 1.0e-6,
  //                                                     .absolute_tol   = 1.0e-10,
  //                                                     .max_iterations = 500,
  //                                                     .print_level    = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-14,
                                              .max_iterations = 12,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<L2<p>, L2<p>, L2<p> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional", mesh_tag);

  // Material properties
  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 4.0e5;  // 4.0e5 [Kg /s2 / mm]
  double possion_ratio   = 0.45;   // 0.49;   // 0.48 // 
  double beta_param      = 2.0e5; // 5.20e5; // 2.31e5; // [Kg /s2 / mm] 
  double max_order_param = 0.45;   // 0.20;   // 0.45; //
  double min_order_param = 0.0;   // 0.20;   // 0.45; //
  double gamma_angle     = 0.0;
  double eta_angle       = 0.0;

  switch (problemID) {
    case 0:
      gamma_angle = 0.0;
      eta_angle   = 0.0;
      break;
    case 1:
      gamma_angle = M_PI_2;
      eta_angle   = 0.0;
      break;
    case 2:
      gamma_angle = M_PI_2;
      eta_angle   = 0.0;
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(pmesh, L2<0>{}, "gammaParam");
  bool               heterogeneousGammaField = problemID == 2 ? true : false;
  auto               gammaFunc = [heterogeneousGammaField, gamma_angle](const mfem::Vector& x, double) -> double {
    if (heterogeneousGammaField) {
      double d    = 5.0e-3;
      double t    = 0.525e-3;

      // vertical
      if ( (x[0] < 0.3e-3) || (x[0] > 9.20e-3) || ((x[0] > 4.45e-3)&&(x[0] < 5.05e-3)  ) ){
        return M_PI_2;
      }
      // forward incline
      else if ( x[0] <= 4.5e-3 + 0.0*d*t ){
        return -0.1920;
      }
      // backward incline
      else{ // if ( x[0] >= 5.05e-3 + 0.0*d*t ){
        return +0.1920;
      }
    }
    return gamma_angle;
  };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState etaParam(pmesh, L2<0>{}, "etaParam");
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
  LiquidCrystalElastomerBertoldi lceMat(density, young_modulus, possion_ratio, max_order_param, beta_param);

  solid_solver.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat);

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({1}, zeroFunc, 0);  // left face x-dir disp = 0
  solid_solver.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
  solid_solver.setDisplacementBCs({3}, zeroFunc, 2);  // back face z-dir disp = 0

  double iniDispVal = 5.0e-6;
  if (problemID == 4) {
    iniDispVal = 5.0e-8;
  }
  auto ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename = "sol_lce_bertoldi_amr_test_no_border_with_qoi_ref_0";
  solid_solver.outputStateToDisk(outputFilename);

  // QoI for output
  // --------------
  // auto& pmesh = serac::StateManager::mesh();
  Functional<double(H1<p, dim>, serac::L2<p>, serac::L2<p>, serac::L2<p>)> strainEnergyQoI(
      {&solid_solver.displacement().space(), &orderParam.space(), &gammaParam.space(), &etaParam.space()});
  strainEnergyQoI.AddDomainIntegral(
      serac::Dimension<dim>{},
      DependsOn<0, 1, 2, 3>{},
      [=](double /*t*/, auto /*x*/, auto displacement, auto order_param_tuple, auto gamma_param_tuple, auto eta_param_tuple) {
        auto du_dx = serac::get<1>(displacement);
        serac::LiquidCrystalElastomerBertoldi::State state{};
        // auto strain = serac::sym(du_dx);
        // auto stress = lceMat(state, du_dx, order_param_tuple, gamma_param_tuple, eta_param_tuple);
        // return 0.5 * serac::double_dot(strain, stress);
        auto strainEnergy = lceMat.calculateStrainEnergy(state, du_dx, order_param_tuple, gamma_param_tuple, eta_param_tuple);
        return strainEnergy;
      },
      pmesh);

  // Time stepping
  // --------------    
  int num_steps = 10;
  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;
  // double gblDispYmin;

  for (int i = 0; i < num_steps; i++) {

    t += dt;
    // orderParam = max_order_param * (tmax - t) / tmax;
    orderParam = min_order_param + (max_order_param-min_order_param) * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
                << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

    // Compute QoI
    double current_qoi = strainEnergyQoI(t, solid_solver.displacement(), orderParam, gammaParam, etaParam);

    // Construct adjoint load
    serac::FiniteElementDual adjoint_load(solid_solver.displacement().space(), "adjoint_load");
    auto dqoi_du = get<1>(strainEnergyQoI(DifferentiateWRT<0>{}, t, solid_solver.displacement(), orderParam, gammaParam, etaParam));
    adjoint_load = *assemble(dqoi_du);

    // Solve adjoint problem
    solid_solver.setAdjointLoad({{"displacement", adjoint_load}});
    solid_solver.reverseAdjointTimestep();

    // Output data
    auto&                 fes             = solid_solver.displacement().space();
    mfem::ParGridFunction displacement_gf = solid_solver.displacement().gridFunction();
    int                   numDofs         = fes.GetNDofs();
    mfem::Vector          dispVecX(numDofs);
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
    MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double gblDispZmin, lclDispZmin = dispVecZ.Min();
    double gblDispZmax, lclDispZmax = dispVecZ.Max();
    MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
                << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
                << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
                << std::endl;

    std::cout << "\n... The QoIVal is: " << current_qoi << std::endl;
    
      if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
        std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
        exit(1);
      }
    }
  }

  // -------------------
  // -------------------
  // AMR mesh generation
  // -------------------
  // -------------------

  // std::string   filename2 = SERAC_REPO_DIR "/data/meshes/reEntrantHoneycomb_3D_2x1_no_border_amr.g";
  auto new_mesh = buildMeshFromFile(filename);
  int numElems = new_mesh.GetNE();

  auto dQoIdp = solid_solver.computeTimestepSensitivity(GAMMA_INDEX);
  mfem::HypreParVector* assembledVector(const_cast<mfem::ParLinearForm &>(dQoIdp.linearForm()).ParallelAssemble());

  // Construct grid function from hypre vector
  mfem::ParGridFunction dQoIdp_pgf(dQoIdp.linearForm().ParFESpace(), assembledVector);
  mfem::FiniteElementCollection *fec = new mfem::H1_FECollection(1, 3);
  mfem::FiniteElementSpace fespace_h1(&new_mesh, fec);
  mfem::GridFunction temp_dQoIdp_pgf(&fespace_h1);

  // Get data
  for (int j = 0; j < fespace_h1. GetVSize(); j++) { temp_dQoIdp_pgf(j) = dQoIdp_pgf(j)/ 1.0e-10;}

  // Generate coefficient 
  mfem::GridFunctionCoefficient sensitivities(&temp_dQoIdp_pgf);

  // Generate L2 space for evaluations
  mfem::FiniteElementCollection *l2_fec = new mfem::L2_FECollection(0, 3);
  mfem::FiniteElementSpace fespace_l2(&new_mesh, l2_fec);
  assert(temp_dQoIdp_pgf.Size()==numElems);

  mfem::GridFunction sensitivities_l2_proj(&fespace_l2);
  sensitivities_l2_proj = 0.0;

  // Bounds of field for refinement
  int numRef = 2;
  double minVal = temp_dQoIdp_pgf.Min();
  double maxVal = temp_dQoIdp_pgf.Max();
  double deltaVal = maxVal-minVal;

  // Assign attributes based on sensitivities
  for (int e = 0; e < numElems; e++) {
    ::mfem::Vector eval(fespace_l2.GetFE(e)->GetDof()); 
    fespace_l2.GetFE(e)->Project(sensitivities, *(fespace_l2.GetElementTransformation(e)), eval);
    sensitivities_l2_proj(e) = eval.Sum()/eval.Size();
    int attr = 1;
    if (sensitivities_l2_proj(e) > minVal + 0.5/(numRef+1) * deltaVal){attr = 2;}
    if (sensitivities_l2_proj(e) > minVal + 1.0/(numRef+1) * deltaVal){attr = 3;}
    if (sensitivities_l2_proj(e) > minVal + 1.5/(numRef+1) * deltaVal){attr = 4;}
    new_mesh.GetElement(e)->SetAttribute(attr);
  }

  // Output original mesh for verification purposes
  mfem::ParaViewDataCollection vis1("originalMeshAMRTest", &new_mesh);
  vis1.SetCycle(0);
  vis1.SetTime(1.0);
  mfem::GridFunction temp_dQoIdp_pgf_l2(&fespace_l2);
  temp_dQoIdp_pgf_l2.ProjectCoefficient(sensitivities);
  vis1.RegisterField("sensitivities", &temp_dQoIdp_pgf_l2);
  vis1.RegisterField("sensitivities_l2_proj", &sensitivities_l2_proj);
  vis1.Save();

  // Refine mesh as necessary
  for (int iRef=1; iRef<(numRef+1); iRef++){
    mfem::Array<int> new_elem_to_refine;
    for (int iElem=0; iElem<new_mesh.GetNE(); iElem++) {
      if (new_mesh.GetElement(iElem)->GetAttribute() > iRef) { 
        new_elem_to_refine.Append(iElem);
      }
    }
    new_mesh.GeneralRefinement(new_elem_to_refine);
    fespace_h1.Update();
    fespace_l2.Update();
  }

  mfem::ParaViewDataCollection vis("refinedMeshAMRTest", &new_mesh);
  vis.SetCycle(0);
  vis.SetTime(1.0);
  vis.Save();

  std::ostringstream mesh_name;
  mesh_name << "refinedMeshAMRTest.mesh";
  std::ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(8);
  new_mesh.Print(mesh_ofs);

  std::cout<<"......... Finished refining the mesh ........."<<std::endl;

  // --------------
  // --------------
  // AMR mesh rerun
  // --------------
  // --------------

//   // Create DataStore
//   // axom::sidre::DataStore datastore_amr;
//   auto amr_mesh_name = "solid_lce_functional_amr";
//   serac::StateManager::initialize(datastore, amr_mesh_name);
//   std::cout<<"......... New refined mesh initialized........."<<std::endl;

//   auto mesh_amr = mesh::refineAndDistribute(std::move(new_mesh), serial_refinement, parallel_refinement);
//   std::cout<<"......... New refined mesh refined and distributed........."<<std::endl;

//   serac::StateManager::setMesh(std::move(mesh_amr), amr_mesh_name);
//   auto& pmesh_amr = serac::StateManager::mesh(amr_mesh_name);
//   std::cout<<"......... New refined mesh moved to state manager........."<<std::endl;

//   SolidMechanics<p, dim, Parameters<L2<p>, L2<p>, L2<p> > > solid_solver_amr(
//       nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "lce_solid_functional_amr", &pmesh_amr);
// std::cout<<"......... 1 ........."<<std::endl;
//   // Parameter 1
//   FiniteElementState orderParam_amr(StateManager::newState(
//     FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "orderParam_amr"}, amr_mesh_name));
//   orderParam_amr = max_order_param;

//   // Parameter 2
//   FiniteElementState gammaParam_amr(StateManager::newState(
//       FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "gammaParam_amr"}, amr_mesh_name));
//   gammaParam_amr.project(gammaCoef);

//   // Paremetr 3
//   FiniteElementState        etaParam_amr(StateManager::newState(
//       FiniteElementState::Options{.order = p, .element_type = ElementType::L2, .name = "etaParam_amr"}, amr_mesh_name));
//   etaParam_amr.project(etaCoef);
// std::cout<<"......... 2 ........."<<std::endl;
//   // Set parameters
//   solid_solver_amr.setParameter(ORDER_INDEX, orderParam_amr);
//   solid_solver_amr.setParameter(GAMMA_INDEX, gammaParam_amr);
//   solid_solver_amr.setParameter(ETA_INDEX, etaParam_amr);

//   // Set material
//   LiquidCrystalElastomerBertoldi lceMat_amr(density, young_modulus, possion_ratio, max_order_param, beta_param);
// std::cout<<"......... 3 ........."<<std::endl;
//   solid_solver_amr.setMaterial(DependsOn<ORDER_INDEX, GAMMA_INDEX, ETA_INDEX>{}, lceMat_amr);

//   solid_solver_amr.setDisplacementBCs({1}, zeroFunc, 0);  // left face x-dir disp = 0
//   solid_solver_amr.setDisplacementBCs({2}, zeroFunc, 1);  // bottom face y-dir disp = 0
//   solid_solver_amr.setDisplacementBCs({3}, zeroFunc, 2);  // back face z-dir disp = 0
//   solid_solver_amr.setDisplacement(ini_displacement);

//   // Finalize the data structures
//   solid_solver_amr.completeSetup();
// std::cout<<"......... 4 ........."<<std::endl;
//   // Perform the quasi-static solve
//   std::string outputFilename_amr = "sol_lce_bertoldi_amr_test_no_border_with_qoi_ref_0_refined_mesh";
//   solid_solver_amr.outputStateToDisk(outputFilename_amr);

//   // QoI for output
//   // --------------
//   Functional<double(H1<p, dim>, serac::L2<p>, serac::L2<p>, serac::L2<p>)> strainEnergyQoI_amr(
//       {&solid_solver_amr.displacement().space(), &orderParam_amr.space(), &gammaParam_amr.space(), &etaParam_amr.space()});
//   strainEnergyQoI_amr.AddDomainIntegral(
//       serac::Dimension<dim>{},
//       DependsOn<0, 1, 2, 3>{},
//       [=](auto /*x*/, auto displacement, auto order_param_tuple, auto gamma_param_tuple, auto eta_param_tuple) {
//         auto du_dx = serac::get<1>(displacement);
//         auto strain = serac::sym(du_dx);
//         serac::LiquidCrystalElastomerBertoldi::State state{};
//         auto stress = lceMat_amr(state, du_dx, order_param_tuple, gamma_param_tuple, eta_param_tuple);
//         return 0.5 * serac::double_dot(strain, stress);
//       },
//       pmesh_amr);
// std::cout<<"......... 5 ........."<<std::endl;
//   // Time stepping
//   // -------------- 
//   for (int i = 0; i < num_steps; i++) {

//     t += dt;
//     // orderParam_amr = max_order_param * (tmax - t) / tmax;
//     orderParam_amr = min_order_param + (max_order_param-min_order_param) * std::pow((tmax - t) / tmax, 1.0);

//     if (rank == 0) {
//       std::cout << "\n\n............................"
//                 << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
//                 << "\n............................\n"
//                 << "\n... Using order parameter: " << max_order_param * (tmax - t) / tmax
//                 << "\n... Using gamma = " << gamma_angle << ", and eta = " << eta_angle << std::endl;
//     }

//     solid_solver_amr.advanceTimestep(dt);
//     solid_solver_amr.outputStateToDisk(outputFilename_amr);

//     // Compute QoI
//     double current_qoi = strainEnergyQoI_amr(solid_solver_amr.displacement(), orderParam_amr, gammaParam_amr, etaParam_amr);

//     // Construct adjoint load
//     serac::FiniteElementDual adjoint_load(solid_solver_amr.displacement().space(), "adjoint_load");
//     auto dqoi_du = get<1>(strainEnergyQoI_amr(DifferentiateWRT<0>{}, solid_solver_amr.displacement(), orderParam_amr, gammaParam_amr, etaParam_amr));
//     adjoint_load = *assemble(dqoi_du);

//     // Solve adjoint problem
//     solid_solver_amr.solveAdjoint({{"displacement", adjoint_load}});

//     // Output data
//     auto&                 fes             = solid_solver_amr.displacement().space();
//     mfem::ParGridFunction displacement_gf = solid_solver_amr.displacement().gridFunction();
//     int                   numDofs         = fes.GetNDofs();
//     mfem::Vector          dispVecX(numDofs);
//     dispVecX = 0.0;
//     mfem::Vector dispVecY(numDofs);
//     dispVecY = 0.0;
//     mfem::Vector dispVecZ(numDofs);
//     dispVecZ = 0.0;

//     for (int k = 0; k < numDofs; k++) {
//       dispVecX(k) = displacement_gf(0 * numDofs + k);
//       dispVecY(k) = displacement_gf(1 * numDofs + k);
//       dispVecZ(k) = displacement_gf(2 * numDofs + k);
//     }
//     double gblDispXmin, lclDispXmin = dispVecX.Min();
//     double gblDispXmax, lclDispXmax = dispVecX.Max();
//     double gblDispYmin, lclDispYmin = dispVecY.Min();
//     double gblDispYmax, lclDispYmax = dispVecY.Max();
//     MPI_Allreduce(&lclDispXmin, &gblDispXmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(&lclDispXmax, &gblDispXmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     MPI_Allreduce(&lclDispYmin, &gblDispYmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(&lclDispYmax, &gblDispYmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//     double gblDispZmin, lclDispZmin = dispVecZ.Min();
//     double gblDispZmax, lclDispZmax = dispVecZ.Max();
//     MPI_Allreduce(&lclDispZmin, &gblDispZmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(&lclDispZmax, &gblDispZmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//     if (rank == 0) {
//       std::cout << "\n... In time step: " << i + 1 << " (/" << num_steps << ")"
//                 << "\n... Min X displacement: " << gblDispXmin << "\n... Max X displacement: " << gblDispXmax
//                 << "\n... Min Y displacement: " << gblDispYmin << "\n... Max Y displacement: " << gblDispYmax
//                 << "\n... Min Z displacement: " << gblDispZmin << "\n... Max Z displacement: " << gblDispZmax
//                 << std::endl;

//     std::cout << "\n... The QoIVal is: " << current_qoi << std::endl;
    
//       if (std::isnan(gblDispXmax) || gblDispXmax > 1.0e3) {
//         std::cout << "... Solution blew up... Check boundary and initial conditions." << std::endl;
//         exit(1);
//       }
//     }
//   }

  serac::exitGracefully();
}