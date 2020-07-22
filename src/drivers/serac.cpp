// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

//***********************************************************************
//
//   SERAC - Nonlinear Implicit Contact Proxy App
//
//   Description: The purpose of this code is to act as a proxy app
//                for nonlinear implicit mechanics codes at LLNL. This
//                initial version is copied from a previous version
//                of the ExaConsist AM miniapp.
//
//
//***********************************************************************

#include <fstream>
#include <iostream>
#include <memory>

#include "CLI11/CLI11.hpp"
#include "coefficients/loading_functions.hpp"
#include "coefficients/traction_coefficient.hpp"
#include "mfem.hpp"
#include "serac_config.hpp"
#include "solvers/nonlinear_solid_solver.hpp"
#include "common/logger.hpp"

int main(int argc, char *argv[])
{
  // Initialize MPI.
  int num_procs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize SLIC logger
  if(!serac::logger::Initialize(MPI_COMM_WORLD))
  {
    serac::ExitGracefully(true);
  }

  // mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/beam-hex.mesh";

  // serial and parallel refinement levels
  int ser_ref_levels = 0;
  int par_ref_levels = 0;

  // polynomial interpolation order
  int order = 1;

  // Solver parameters
  serac::NonlinearSolverParameters nonlin_params;
  nonlin_params.rel_tol     = 1.0e-2;
  nonlin_params.abs_tol     = 1.0e-4;
  nonlin_params.max_iter    = 500;
  nonlin_params.print_level = 0;

  serac::LinearSolverParameters lin_params;
  lin_params.rel_tol     = 1.0e-6;
  lin_params.abs_tol     = 1.0e-8;
  lin_params.max_iter    = 5000;
  lin_params.print_level = 0;

  // solver input args
  bool gmres_solver = true;
  bool slu_solver   = false;

  // neo-Hookean material parameters
  double mu = 0.25;
  double K  = 5.0;

  // loading parameters
  double tx = 0.0;
  double ty = 1.0e-3;
  double tz = 0.0;

  double t_final = 1.0;
  double dt      = 0.25;

  // specify all input arguments
  CLI::App app{"serac: a high order nonlinear thermomechanical simulation code"};
  app.add_option("-m, --mesh", mesh_file,  "Mesh file to use.", true);
  app.add_option("--rs, --refine-serial", ser_ref_levels,  "Number of times to refine the mesh uniformly in serial.", true);
  app.add_option("--rp, --refine-parallel", par_ref_levels, 
                  "Number of times to refine the mesh uniformly in parallel.", true);
  app.add_option("-o, --order", order,  "Order degree of the finite elements.", true);
  app.add_option("--mu, --shear-modulus", mu,  "Shear modulus in the Neo-Hookean hyperelastic model.", true);
  app.add_option("-K, --bulk-modulus", K,  "Bulk modulus in the Neo-Hookean hyperelastic model.", true);
  app.add_option("--tx, --traction-x", tx,  "Cantilever tip traction in the x direction.", true);
  app.add_option("--ty, --traction-y", ty,  "Cantilever tip traction in the y direction.", true);
  app.add_option("--tz, --traction-z", tz,  "Cantilever tip traction in the z direction.", true);
  app.add_flag("--slu, --superlu, !--no-slu, !--no-superlu", slu_solver,  "Use the SuperLU Solver.");
  app.add_option("--lrel, --linear-relative-tolerance", lin_params.rel_tol, 
                  "Relative tolerance for the lienar solve.", true);
  app.add_option("--labs, --linear-absolute-tolerance", lin_params.abs_tol, 
                  "Absolute tolerance for the linear solve.", true);
  app.add_option("--lit, --linear-iterations", lin_params.max_iter,  "Maximum iterations for the linear solve.", true);
  app.add_option("--lpl, --linear-print-level", lin_params.print_level,  "Linear print level.", true);
  app.add_option("--nrel, --newton-relative-tolerance", nonlin_params.rel_tol, 
                  "Relative tolerance for the Newton solve.", true);
  app.add_option("--nabs, --newton-absolute-tolerance", nonlin_params.abs_tol, 
                  "Absolute tolerance for the Newton solve.", true);
  app.add_option("--nit, --newton-iterations", nonlin_params.max_iter,  "Maximum iterations for the Newton solve.", true);
  app.add_option("--npl, --newton-print-level", nonlin_params.print_level,  "Newton print level.", true);
  app.add_option("--dt, --time-step", dt,  "Time step.", true);

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    if (rank == 0) {
      app.exit(e);
      // Don't reprint the usage if CLI11 already has
      if (e.get_name() != "CallForHelp") {
        std::cout << app.help() << '\n';
      }
    }
    serac::ExitGracefully(true);
  }
  if (rank == 0) {
    std::cout << app.config_to_str(true, true) << '\n';
  }

  // Open the mesh
  std::string msg = fmt::format("Opening mesh file: {0}", mesh_file);
  SLIC_INFO_MASTER(rank, msg);
  std::ifstream imesh(mesh_file);
  if (!imesh) {
    serac::logger::Flush();
    std::string msg = fmt::format("Can not open mesh file: {0}", mesh_file);
    SLIC_ERROR_MASTER(rank, msg);
    serac::ExitGracefully();
  }

  auto mesh = std::make_unique<mfem::Mesh>(imesh, 1, 1, true);
  imesh.close();

  // mesh refinement if specified in input
  for (int lev = 0; lev < ser_ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  // create the parallel mesh
  auto pmesh = std::make_shared<mfem::ParMesh>(MPI_COMM_WORLD, *mesh);
  for (int lev = 0; lev < par_ref_levels; lev++) {
    pmesh->UniformRefinement();
  }

  int dim = pmesh->Dimension();

  // Define the solid solver object
  NonlinearSolidSolver solid_solver(order, pmesh);

  // Project the initial and reference configuration functions onto the
  // appropriate grid functions
  mfem::VectorFunctionCoefficient defo_coef(dim, InitialDeformation);

  mfem::Vector velo(dim);
  velo = 0.0;

  mfem::VectorConstantCoefficient velo_coef(velo);

  // initialize x_cur, boundary condition, deformation, and
  // incremental nodal displacment grid functions by projection the
  // VectorFunctionCoefficient function onto them
  solid_solver.SetDisplacement(defo_coef);
  solid_solver.SetVelocity(velo_coef);

  std::set<int> ess_bdr = {1};

  // define the displacement vector
  mfem::Vector disp(dim);
  disp           = 0.0;
  auto disp_coef = std::make_shared<mfem::VectorConstantCoefficient>(disp);

  std::set<int> trac_bdr = {2};

  // define the traction vector
  mfem::Vector traction(dim);
  traction(0) = tx;
  traction(1) = ty;
  if (dim == 3) {
    traction(2) = tz;
  }

  auto traction_coef = std::make_shared<VectorScaledConstantCoefficient>(traction);

  // Set the boundary condition information
  solid_solver.SetDisplacementBCs(ess_bdr, disp_coef);
  solid_solver.SetTractionBCs(trac_bdr, traction_coef);

  // Set the material parameters
  solid_solver.SetHyperelasticMaterialParameters(mu, K);

  // Set the linear solver parameters
  if (gmres_solver == true) {
    lin_params.prec       = serac::Preconditioner::BoomerAMG;
    lin_params.lin_solver = serac::LinearSolver::GMRES;
  } else {
    lin_params.prec       = serac::Preconditioner::Jacobi;
    lin_params.lin_solver = serac::LinearSolver::MINRES;
  }
  solid_solver.SetSolverParameters(lin_params, nonlin_params);

  // Set the time step method
  solid_solver.SetTimestepper(serac::TimestepMethod::QuasiStatic);

  // Complete the solver setup
  solid_solver.CompleteSetup();

  // initialize/set the time
  double t = 0.0;

  bool last_step = false;

  solid_solver.InitializeOutput(serac::OutputType::VisIt, "serac");

  // enter the time step loop. This was modeled after example 10p.
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    // compute current time
    t = t + dt_real;

    if (rank == 0) {
      std::cout << "step " << ti << ", t = " << t << std::endl;
    }

    traction_coef->SetScale(t);

    // Solve the Newton system
    solid_solver.AdvanceTimestep(dt_real);

    solid_solver.OutputState();

    last_step = (t >= t_final - 1e-8 * dt);
  }

  serac::ExitGracefully();
} 
