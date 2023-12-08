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

#include "serac/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "serac/physics/boundary_conditions/boundary_condition_helper.hpp"

using namespace serac;

// #define ALT_ITER_SOLVER
#undef ALT_ITER_SOLVER 

const static int problemID = 0;

////////////////////////////////////////////////////////////////////////////////////////
class StdFunctionVectorCoefficient : public mfem::VectorCoefficient {
public:
  StdFunctionVectorCoefficient(int dim, std::function<void(mfem::Vector &, mfem::Vector &)> func): 
  mfem::VectorCoefficient(dim), func_(func)
  {}

  void Eval(mfem::Vector &V, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    double x[T.GetSpaceDim()];
    mfem::Vector transip(x, T.GetSpaceDim());

    T.Transform(ip, transip);
    func_(transip, V);
    V[0] = 0.0;
    V[2] = 0.0;
  }

private:
  std::function<void(mfem::Vector &, mfem::Vector &)> func_;
};

////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  constexpr int p         = 1;
  constexpr int dim       = 3;
  int serial_refinement   = 0;
  int parallel_refinement = 0;

  if (problemID<3){parallel_refinement = 1;}

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_lce_functional");

  // Get mesh
  std::string inputFilename;
  switch (problemID) {
    case 0:
      inputFilename = SERAC_REPO_DIR "/data/meshes/dbgLogPileNoSymm.g";
      // inputFilename = SERAC_REPO_DIR "/data/meshes/dbgLogPileNoSymm_thicker.g";
      break;
    case 1:
      inputFilename = SERAC_REPO_DIR "/data/meshes/dbgLogPileNoSymm.g";
      break;
    case 2:
      inputFilename = SERAC_REPO_DIR "/data/meshes/dbgLogPileNoSymm.g";
      std::cout << "...... Problem used for debugging. do not run in current configuration ......" << std::endl;
      exit(0);
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  auto initial_mesh = buildMeshFromFile(inputFilename);
  auto mesh = mesh::refineAndDistribute(std::move(initial_mesh), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"}; 
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a functional-based solid mechanics solver
  // LinearSolverOptions linear_options = {.linear_solver = LinearSolver::SuperLU};
  const LinearSolverOptions linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};

  NonlinearSolverOptions nonlinear_options = {.nonlin_solver  = serac::NonlinearSolver::Newton,
                                              .relative_tol   = 1.0e-8,
                                              .absolute_tol   = 1.0e-12,
                                              .max_iterations = 1,
                                              .print_level    = 1};
  SolidMechanics<p, dim, Parameters<L2<0>, L2<0>, L2<0> > > solid_solver(
      nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, 
      "lce_solid_free_swelling", mesh_tag, {"orderParam", "gammaParam", "etaParam"});

  // Material properties
  double density         = 1.0;    // [Kg / mm3]
  double young_modulus   = 9.34e6;  // 4.0e5 [Kg /s2 / mm]
  double possion_ratio   = 0.49;   // 0.49;   // 0.48 // 
  double beta_param      = 5.75e5; // 5.20e5; // 2.31e5; // [Kg /s2 / mm] 
  double max_order_param = 0.40;   // 0.20;   // 0.45; //
  double gamma_angle     = M_PI_2;
  double eta_angle       = 0.0;

  // Parameter 1
  FiniteElementState orderParam(pmesh, L2<0>{}, "orderParam");
  orderParam = max_order_param;

  // Parameter 2
  FiniteElementState gammaParam(pmesh, L2<0>{}, "gammaParam");
  auto gammaFunc = [=](const mfem::Vector& /*x*/, double) -> double {
    double alignmentAngle = 0.0;
    return alignmentAngle;
  };
  mfem::FunctionCoefficient gammaCoef(gammaFunc);
  gammaParam.project(gammaCoef);

  // Paremetr 3
  FiniteElementState        etaParam(pmesh, L2<0>{}, "etaParam");
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

  auto zeroDisp = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs({2}, zeroDisp); 

  auto zeroFunc = [](const mfem::Vector /*x*/) { return 0.0; };
  solid_solver.setDisplacementBCs({5}, zeroFunc, 2);  // back face z-dir disp = 0

  double targetDisp = -1.0e-3;
  double topBoundaryYCoord = 6.374999e-3;
  if (problemID==1)
  {
    targetDisp = -2.4e-3;
  }

  auto is_on_top = [=](const mfem::Vector& x) {
    bool tag = false;
    if (x(1) > topBoundaryYCoord) {
    // if (x(1) > 4e-3) {
      tag = true;
    }
    return tag;
  };

  auto scalar_offset = [=](const mfem::Vector&, double t) { return targetDisp*(t+0.01); };
  solid_solver.setDisplacementBCs(is_on_top, scalar_offset, 1);
  solid_solver.setDisplacementBCs({4}, zeroFunc, 0);
  // solid_solver.setDisplacementBCs({4}, zeroFunc, 2);

  // auto compression_disp = [=](const mfem::Vector&) { return 0.01*targetDisp; }; // { return targetDisp*(t+0.1); };
  // solid_solver.setDisplacementBCs({4}, compression_disp, 1);  // back face z-dir disp = 0
  
  // auto nonZeroFunc = [](const mfem::Vector /*x*/) { return -3.6e-3; };
  // solid_solver.setDisplacementBCs({4}, nonZeroFunc, 1);  // back face z-dir disp = 0

  double iniDispVal = -1.0e-6;
  auto ini_displacement = [=](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  std::string outputFilename;
  switch (problemID) {
    case 0:
      outputFilename = "sol_log_pile_rect_dbg";
      break;
    case 1:
      outputFilename = "sol_log_pile_rect_3D";
      break;
    default:
      std::cout << "...... Wrong problem ID ......" << std::endl;
      exit(0);
  }

  solid_solver.outputStateToDisk(outputFilename);

  int num_steps = 50;
  if (problemID==1)
  {
    num_steps = 20;
  }

  double t    = 0.0;
  double tmax = 1.0;
  double dt   = tmax / num_steps;

  if (rank == 0) {
    std::cout << "\n\n###############################" 
    << "\n... problemID: " << problemID 
    << "\n###############################" << std::endl;
  }

/////////////////////////////////////////////////////////////////////////////////////

  Functional<double(H1<p, dim>)> area({&solid_solver.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{}, 
      [=](auto position) { 
          auto [X, dX_dxi] = position;
          return (X[1] > topBoundaryYCoord) ? 1.0 : 0.0; 
          // return (X[1] > 6.0) ? 1.0 : 0.0; 
        }, 
        pmesh);

  double initial_area = area(solid_solver.displacement());
  if (rank == 0) {
    std::cout << "... Initial Area of the top surface: " << initial_area << std::endl;
  }
/////////////////////////////////////////////////////////////////////////////////////

  for (int i = 0; i < num_steps; i++) {
    // orderParam = max_order_param * (tmax - t) / tmax;
    // orderParam = max_order_param * std::pow((tmax - t) / tmax, 1.0);

    if (rank == 0) {
      std::cout << "\n\n............................"
                << "\n... Entering time step: " << i + 1 << " (/" << num_steps << ")"
                << "\n............................\n"
                << "\n... Using order parameter = " << max_order_param << ", gamma = " << gamma_angle << ", and eta = " << eta_angle
                << "\n... Using target displacement =  " << targetDisp * (t+0.01) // maxYDisp * t / tmax 
                << "\n... At total time =  " << t 
                << std::endl << std::endl;
    }

    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(outputFilename);

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

/////////////////////////////////////////////////////////////////////////////////////
    // QoI for output:
    mfem::ParFiniteElementSpace reactions_fes(solid_solver.reactions().space());
    mfem::ParGridFunction       reactionsAtTop(&reactions_fes);
    auto topTag = [=](const mfem::Vector& x, mfem::Vector& output) {
      output= 0.0;
      if (x(1) > topBoundaryYCoord) {
        output = 1.0;
      }
    };
    // mfem::FunctionCoefficient topTagCoeff(topTag);
    StdFunctionVectorCoefficient topTagCoeff(3, topTag);

    reactionsAtTop.ProjectCoefficient(topTagCoeff);
    
    mfem::ParLinearForm reactionLF(&reactions_fes);
    solid_solver.reactions().fillLinearForm(reactionLF);

    double totalReaction = reactionLF(reactionsAtTop);

    if (rank == 0) {
      // std::cout << "... Initial Area of the top surface: " << initial_area << std::endl;
      std::cout << "... Total reaction foce along the top boundary: " << totalReaction << std::endl;
      std::cout << "... Max reactionLF value: " << reactionLF.Max() << std::endl;
      // std::cout << "... Total reaction force along the top boundary: " << totalReaction*initial_area << std::endl;
    }

/////////////////////////////////////////////////////////////////////////////////////

    auto temp_mesh = buildMeshFromFile(inputFilename);
    mfem::ParMesh *temp_pmesh = new mfem::ParMesh(MPI_COMM_WORLD, temp_mesh);
    // auto temp_pmesh = mesh::refineAndDistribute(std::move(temp_mesh), serial_refinement, parallel_refinement);
    
    auto h1_fec = mfem::H1_FECollection(1, 3);
    mfem::ParFiniteElementSpace h1_fes(temp_pmesh, &h1_fec, 1);
    mfem::ParGridFunction tempGF(&h1_fes);
    // mfem::FunctionCoefficient tempTopTagCoeff(
    //   [=](const mfem::Vector& x) {
    //   double output= 0.0;
    //   if (x(1) > topBoundaryYCoord) {
    //     output = 1.0;
    //   }
    //   return output;
    // });
    mfem::VectorFunctionCoefficient tempTopTagCoeff(3, 
      [=](const mfem::Vector& x, mfem::Vector& f) {
      f(0) = 0.0;
      f(2) = 0.0;
      if (x(1) > topBoundaryYCoord) {
        f(1) = 1.0;
      }
    });
    tempGF.ProjectCoefficient(tempTopTagCoeff);

    // mfem::HypreParVector* assembledVector(const_cast<mfem::ParLinearForm &> (reactionLF.ParallelAssemble()));
    // mfem::ParGridFunction tempReactionGF(reactionLF.ParFESpace(), assembledVector);

    mfem::Vector tempReactionGF(reactions_fes.GetTrueVSize());
    const_cast<mfem::ParLinearForm &> (reactionLF).ParallelAssemble(tempReactionGF);
    
    mfem::ParGridFunction tempReactions(&h1_fes);
    for (int k = 0; k < 3*numDofs; k++) {
      tempReactions(k) = tempReactionGF(k);
    }

    mfem::ParaViewDataCollection vis("tempCheckMesh", temp_pmesh);
    vis.RegisterField("tempGF", &tempGF);
    vis.RegisterField("reactions", &tempReactions);
    vis.SetCycle(1);
    vis.SetTime(1);
    vis.Save();
    exit(0);
/////////////////////////////////////////////////////////////////////////////////////

    t += dt;
  }

  serac::exitGracefully();
}