// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Nonlinear implicit proxy app driver
 *
 * The purpose of this code is to act as a proxy app for nonlinear implicit mechanics codes at LLNL.
 */

#include <fstream>
#include <iostream>
#include <memory>

#include "CLI11/CLI11.hpp"
#include "coefficients/loading_functions.hpp"
#include "coefficients/traction_coefficient.hpp"
#include "infrastructure/initialize.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"
#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid_solver.hpp"
#include "serac_config.hpp"

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

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
  app.add_option("-m, --mesh", mesh_file, "Mesh file to use.", true);
  app.add_option("--rs, --refine-serial", ser_ref_levels, "Number of times to refine the mesh uniformly in serial.",
                 true);
  app.add_option("--rp, --refine-parallel", par_ref_levels, "Number of times to refine the mesh uniformly in parallel.",
                 true);
  app.add_option("-o, --order", order, "Order degree of the finite elements.", true);
  app.add_option("--mu, --shear-modulus", mu, "Shear modulus in the Neo-Hookean hyperelastic model.", true);
  app.add_option("-K, --bulk-modulus", K, "Bulk modulus in the Neo-Hookean hyperelastic model.", true);
  app.add_option("--tx, --traction-x", tx, "Cantilever tip traction in the x direction.", true);
  app.add_option("--ty, --traction-y", ty, "Cantilever tip traction in the y direction.", true);
  app.add_option("--tz, --traction-z", tz, "Cantilever tip traction in the z direction.", true);
  app.add_flag("--slu, --superlu, !--no-slu, !--no-superlu", slu_solver, "Use the SuperLU Solver.");
  app.add_flag("--gmres, !--no-gmres", gmres_solver, "Use gmres, otherwise minimum residual is used.");
  app.add_option("--lrel, --linear-relative-tolerance", lin_params.rel_tol, "Relative tolerance for the lienar solve.",
                 true);
  app.add_option("--labs, --linear-absolute-tolerance", lin_params.abs_tol, "Absolute tolerance for the linear solve.",
                 true);
  app.add_option("--lit, --linear-iterations", lin_params.max_iter, "Maximum iterations for the linear solve.", true);
  app.add_option("--lpl, --linear-print-level", lin_params.print_level, "Linear print level.", true);
  app.add_option("--nrel, --newton-relative-tolerance", nonlin_params.rel_tol,
                 "Relative tolerance for the Newton solve.", true);
  app.add_option("--nabs, --newton-absolute-tolerance", nonlin_params.abs_tol,
                 "Absolute tolerance for the Newton solve.", true);
  app.add_option("--nit, --newton-iterations", nonlin_params.max_iter, "Maximum iterations for the Newton solve.",
                 true);
  app.add_option("--npl, --newton-print-level", nonlin_params.print_level, "Newton print level.", true);
  app.add_option("--dt, --time-step", dt, "Time step.", true);

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    serac::logger::flush();
    auto err_msg = (e.get_name() == "CallForHelp") ? app.help() : CLI::FailureMessage::simple(&app, e);
    SLIC_ERROR_ROOT(rank, err_msg);
    serac::exitGracefully();
  }

  auto config_msg = app.config_to_str(true, true);
  SLIC_INFO_ROOT(rank, config_msg);

  auto mesh = serac::buildMeshFromFile(mesh_file, ser_ref_levels, par_ref_levels);

  int dim = mesh->Dimension();

  // Define the solid solver object
  serac::NonlinearSolidSolver solid_solver(order, mesh);

  // Project the initial and reference configuration functions onto the
  // appropriate grid functions
  mfem::VectorFunctionCoefficient defo_coef(dim, serac::initialDeformation);

  mfem::Vector velo(dim);
  velo = 0.0;

  mfem::VectorConstantCoefficient velo_coef(velo);

  // initialize x_cur, boundary condition, deformation, and
  // incremental nodal displacment grid functions by projection the
  // VectorFunctionCoefficient function onto them
  solid_solver.setDisplacement(defo_coef);
  solid_solver.setVelocity(velo_coef);

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

  auto traction_coef = std::make_shared<serac::VectorScaledConstantCoefficient>(traction);

  // Set the boundary condition information
  solid_solver.setDisplacementBCs(ess_bdr, disp_coef);
  solid_solver.setTractionBCs(trac_bdr, traction_coef);

  // Set the material parameters
  solid_solver.setHyperelasticMaterialParameters(mu, K);

  // Set the linear solver parameters
  if (gmres_solver == true) {
    lin_params.prec       = serac::Preconditioner::BoomerAMG;
    lin_params.lin_solver = serac::LinearSolver::GMRES;
  } else {
    lin_params.prec       = serac::Preconditioner::Jacobi;
    lin_params.lin_solver = serac::LinearSolver::MINRES;
  }
  solid_solver.setSolverParameters(lin_params, nonlin_params);

  // Set the time step method
  solid_solver.setTimestepper(serac::TimestepMethod::QuasiStatic);

  // Complete the solver setup
  solid_solver.completeSetup();

  // initialize/set the time
  double t = 0.0;

  bool last_step = false;

  solid_solver.initializeOutput(serac::OutputType::VisIt, "serac");

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
    solid_solver.advanceTimestep(dt_real);

    solid_solver.outputState();

    last_step = (t >= t_final - 1e-8 * dt);
  }

  serac::exitGracefully();
}
