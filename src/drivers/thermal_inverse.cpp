// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file serac.cpp
 *
 * @brief Serac: nonlinear implicit thermal-structural driver
 *
 * The purpose of this code is to act as a proxy app for nonlinear implicit mechanics codes at LLNL.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "axom/core.hpp"
#include "mfem.hpp"
#include "serac/coefficients/loading_functions.hpp"
#include "serac/infrastructure/cli.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/thermal_solid.hpp"
#include "serac/physics/utilities/equation_solver.hpp"
#include "serac/coefficients/coefficient_extensions.hpp"
#include "serac/serac_config.hpp"

namespace serac {

//------- Input file -------
//
// This defines what we expect to extract from the input file
void defineInputFileSchema(axom::inlet::Inlet& inlet, int rank)
{
  // Initial guess for the design field
  auto& initial_guess_table =
      inlet.addStruct("initial_top_flux_guess", "The initial guess for the top flux").required();
  serac::input::CoefficientInputOptions::defineInputFileSchema(initial_guess_table);

  auto& experimental_flux_table = inlet.addStruct("experimental_flux_coef", "The measured heat flux").required();
  serac::input::CoefficientInputOptions::defineInputFileSchema(experimental_flux_table);

  // Boundary marker for the designed flux
  inlet.addInt("measured_boundary", "Boundary indicator for the measured side");
  inlet.addInt("unknown_boundary", "Boundary indicator for the unknown side");

  // Epsilon for the regularization term
  inlet.addDouble("epsilon", "Scaling factor for the regularization parameter").required();

  // The output type (visit, glvis, paraview, etc)
  serac::input::defineOutputTypeInputFileSchema(inlet.getGlobalTable());

  // The mesh options
  auto& mesh_table = inlet.addStruct("main_mesh", "The main mesh for the problem").required();
  serac::mesh::InputOptions::defineInputFileSchema(mesh_table);

  // The thermal conduction options
  auto& thermal_solver_table = inlet.addStruct("thermal_conduction", "Thermal conduction module").required();
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_solver_table);

  // Verify the input file
  if (!inlet.verify()) {
    SLIC_ERROR_ROOT(rank, "Input file failed to verify.");
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts =
      serac::cli::defineAndParse(argc, argv, rank, "Inverse thermal problem driver");
  serac::cli::printGiven(cli_opts, rank);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");

  SLIC_ERROR_ROOT_IF(search == cli_opts.end(), rank, "Input file must be specified in the command line.");
  input_file_path = search->second;

  // Create DataStore
  axom::sidre::DataStore datastore;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet, rank);

  // Save input values to file
  datastore.getRoot()->save("thermal_inverse_input.json", "json");

  int unknown_boundary  = inlet["unknown_boundary"];
  int measured_boundary = inlet["measured_boundary"];

  std::shared_ptr<mfem::ParMesh> mesh;
  // Build the mesh
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
    auto full_mesh_path = serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
    mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);
  }

  // BUILD AND SETUP THE FORWARD SOLVER

  // Create the forward physics solver object
  auto thermal_solver_options = inlet["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();
  serac::ThermalConduction thermal_solver(mesh, thermal_solver_options);

  // Create and initialize the design field
  auto initial_flux_guess = inlet["initial_top_flux_guess"].get<serac::input::CoefficientInputOptions>();
  auto flux_guess_coef    = initial_flux_guess.constructScalar();

  serac::FiniteElementState designed_flux(
      *mesh,
      serac::FiniteElementState::Options{
          .order = 1, .coll = std::make_unique<mfem::H1_FECollection>(1, mesh->Dimension()), .name = "designed_flux"});
  designed_flux.project(*flux_guess_coef);

  auto designed_flux_coef = std::make_shared<mfem::GridFunctionCoefficient>(&designed_flux.gridFunc());

  // Load up the forward solver with the design parameter
  thermal_solver.setFluxBCs({unknown_boundary}, designed_flux_coef);

  // DO THE FORWARD SOLVE

  // Complete the solver setup
  thermal_solver.completeSetup();

  // Initialize the output
  thermal_solver.initializeOutput(inlet.getGlobalTable().get<serac::OutputType>(), "thermal_inverse");

  // Solve the physics module appropriately
  double dt = 1.0;
  thermal_solver.advanceTimestep(dt);

  // COMPUTE THE ADJOINT LOAD
  // NOTE: this should be replaced by weak_form

  // Make grid function coefficient for the temperature field
  mfem::GradientGridFunctionCoefficient grad_temp_coef(&thermal_solver.temperature().gridFunc());
  mfem::Vector                          vec(mesh->Dimension());
  vec    = 0.0;
  vec(1) = -1.0;
  mfem::VectorConstantCoefficient normal_coef(vec);
  auto computed_flux_coef = std::make_shared<mfem::InnerProductCoefficient>(normal_coef, grad_temp_coef);

  auto exact_flux_coef_options = inlet["experimental_flux_coef"].get<serac::input::CoefficientInputOptions>();
  std::shared_ptr<mfem::Coefficient> exact_flux_coef(exact_flux_coef_options.constructScalar());

  auto misfit_coef = serac::mfem_ext::TransformedScalarCoefficient(
      exact_flux_coef, computed_flux_coef,
      [](double exact_flux, double computed_flux) { return -2.0 * (computed_flux - exact_flux); });

  thermal_solver.setAdjointEssentialBCs({measured_boundary}, misfit_coef);
  // could have a similar setAdjointLoad call

  // SOLVE THE ADJOINT PROBLEM
  thermal_solver.solveAdjoint();

  // Turn the adjoint and state field into a coefficient
  auto adjoint_coef = std::make_shared<mfem::GridFunctionCoefficient>(&thermal_solver.adjointTemperature().gridFunc());
  auto temp_coef    = std::make_shared<mfem::GridFunctionCoefficient>(&thermal_solver.temperature().gridFunc());

  // COMPUTE THE SENSITIVITY
  // This should also get replaced by a weak form

  // Make the sensitivity coefficient
  double                                        epsilon = inlet["epsilon"];
  serac::mfem_ext::TransformedScalarCoefficient sensitivity_coef(
      adjoint_coef, designed_flux_coef, [epsilon](double adjoint_value, double designed_flux_value) {
        return 2.0 * epsilon * designed_flux_value - adjoint_value;
      });

  mfem::ParLinearForm sensitivity_form(&designed_flux.space());
  mfem::Array<int>    markers(mesh->bdr_attributes.Max());
  markers                       = 0;
  markers[unknown_boundary - 1] = 1;
  sensitivity_form.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(sensitivity_coef));

  std::unique_ptr<mfem::HypreParVector> discrete_sensitivity(sensitivity_form.ParallelAssemble());

  serac::exitGracefully();
}