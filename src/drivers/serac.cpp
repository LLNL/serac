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
#include <string>

#include "axom/core.hpp"
#include "coefficients/loading_functions.hpp"
#include "coefficients/traction_coefficient.hpp"
#include "infrastructure/cli.hpp"
#include "infrastructure/initialize.hpp"
#include "infrastructure/input.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"
#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid.hpp"
#include "physics/utilities/equation_solver.hpp"
#include "serac_config.hpp"

namespace serac {

//------- Input file -------

void defineInputFileSchema(std::shared_ptr<axom::inlet::Inlet> inlet, int rank)
{
  // mesh
  serac::mesh::defineInputFileSchema(inlet);

  // Simulation time parameters
  inlet->addDouble("t_final", "Final time for simulation.")->defaultValue(1.0);
  inlet->addDouble("dt", "Time step.")->defaultValue(0.25);

  // Physics
  serac::NonlinearSolid::defineInputFileSchema(inlet);

  // Verify input file
  if (!inlet->verify()) {
    SLIC_ERROR_ROOT(rank, "Input file failed to verify.");
    serac::exitGracefully(true);
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts = serac::cli::defineAndParse(argc, argv, rank);
  serac::cli::printGiven(cli_opts, rank);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");
  if (search != cli_opts.end()) {
    input_file_path = search->second;
  }

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet, rank);

  // Save input values to file
  datastore.getRoot()->save("serac_input.json", "json");

  // serial and parallel refinement levels
  int ser_ref_levels;
  inlet->get("ser_ref_levels", ser_ref_levels);  // has default value
  int par_ref_levels;
  inlet->get("par_ref_levels", par_ref_levels);  // has default value

  // Build mesh
  std::string mesh_file_path;
  inlet->get("mesh", mesh_file_path);  // required in input file
  mesh_file_path = serac::input::findMeshFilePath(mesh_file_path, input_file_path);
  auto mesh      = serac::buildMeshFromFile(mesh_file_path, ser_ref_levels, par_ref_levels);

  // Solver parameters
  auto lin_params = serac::NonlinearSolid::default_qs_linear_params;
  inlet->get("nonlinear_solid/solver/linear/rel_tol", lin_params.rel_tol);          // has default value
  inlet->get("nonlinear_solid/solver/linear/abs_tol", lin_params.abs_tol);          // has default value
  inlet->get("nonlinear_solid/solver/linear/max_iter", lin_params.max_iter);        // has default value
  inlet->get("nonlinear_solid/solver/linear/print_level", lin_params.print_level);  // has default value

  serac::NonlinearSolverParameters nonlin_params;
  inlet->get("nonlinear_solid/solver/nonlinear/rel_tol", nonlin_params.rel_tol);          // has default value
  inlet->get("nonlinear_solid/solver/nonlinear/abs_tol", nonlin_params.abs_tol);          // has default value
  inlet->get("nonlinear_solid/solver/nonlinear/max_iter", nonlin_params.max_iter);        // has default value
  inlet->get("nonlinear_solid/solver/nonlinear/print_level", nonlin_params.print_level);  // has default value

  // solver input args
  std::string solver_type;
  inlet->get("nonlinear_solid/solver/linear/solver_type", solver_type);  // has default value
  if (solver_type == "gmres") {
    lin_params.prec       = serac::HypreBoomerAMGPrec{};
    lin_params.lin_solver = serac::LinearSolver::GMRES;
  } else if (solver_type == "minres") {
    lin_params.prec       = serac::HypreSmootherPrec{mfem::HypreSmoother::l1Jacobi};
    lin_params.lin_solver = serac::LinearSolver::MINRES;
  } else {
    serac::logger::flush();
    std::string msg = fmt::format("Unknown Linear solver type given: {0}", solver_type);
    SLIC_ERROR_ROOT(rank, msg);
    serac::exitGracefully(true);
  }

  // Define the solid solver object
  int order;
  inlet->get("nonlinear_solid/order", order);  // has default value
  serac::NonlinearSolid solid_solver(order, mesh, {lin_params, nonlin_params});

  // Project the initial and reference configuration functions onto the
  // appropriate grid functions
  int dim = mesh->Dimension();

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

  // loading parameters
  double tx, ty, tz;
  inlet->get("nonlinear_solid/tx", tx);  // has default value
  inlet->get("nonlinear_solid/ty", ty);  // has default value
  inlet->get("nonlinear_solid/tz", tz);  // has default value

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

  // neo-Hookean material parameters
  double mu, K;
  inlet->get("nonlinear_solid/mu", mu);  // has default value
  inlet->get("nonlinear_solid/K", K);    // has default value

  // Set the material parameters
  solid_solver.setHyperelasticMaterialParameters(mu, K);

  // Complete the solver setup
  solid_solver.completeSetup();

  // initialize/set the time
  double t = 0;
  double t_final, dt;
  inlet->get("t_final", t_final);  // has default value
  inlet->get("dt", dt);            // has default value

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
