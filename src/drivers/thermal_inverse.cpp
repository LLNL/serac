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
#include <tuple>

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

axom::inlet::Inlet initializeInletAndMesh(int argc, char* argv[], int rank, axom::sidre::DataStore& datastore,
                                          std::shared_ptr<mfem::ParMesh>& mesh)
{
  // Handle Command line
  std::unordered_map<std::string, std::string> cli_opts =
      serac::cli::defineAndParse(argc, argv, rank, "Inverse thermal problem driver");
  serac::cli::printGiven(cli_opts, rank);

  // Read input file
  std::string input_file_path = "";
  auto        search          = cli_opts.find("input_file");

  SLIC_ERROR_ROOT_IF(search == cli_opts.end(), rank, "Input file must be specified in the command line.");
  input_file_path = search->second;

  // Initialize Inlet and read input file
  auto inlet = serac::input::initialize(datastore, input_file_path);
  serac::defineInputFileSchema(inlet, rank);

  // Save input values to file
  datastore.getRoot()->save("thermal_inverse_input.json", "json");

  // Build the mesh
  auto mesh_options = inlet["main_mesh"].get<serac::mesh::InputOptions>();
  if (const auto file_opts = std::get_if<serac::mesh::FileInputOptions>(&mesh_options.extra_options)) {
    auto full_mesh_path = serac::input::findMeshFilePath(file_opts->relative_mesh_file_name, input_file_path);
    mesh = serac::buildMeshFromFile(full_mesh_path, mesh_options.ser_ref_levels, mesh_options.par_ref_levels);
  }

  return inlet;
}

mfem::VectorConstantCoefficient makeNormalCoef(std::shared_ptr<mfem::ParMesh> mesh)
{
  mfem::Vector vec(mesh->Dimension());
  vec    = 0.0;
  vec(1) = -1.0;
  return mfem::VectorConstantCoefficient(vec);
}

std::unique_ptr<serac::ThermalConduction> setupForwardSolver(axom::inlet::Inlet&            inlet,
                                                             std::shared_ptr<mfem::ParMesh> mesh,
                                                             FiniteElementState&            designed_flux)
{
  auto thermal_solver_options = inlet["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();
  auto thermal_solver         = std::make_unique<serac::ThermalConduction>(mesh, thermal_solver_options);

  int unknown_boundary = inlet["unknown_boundary"];

  // Load up the forward solver with the design parameter
  thermal_solver->setFluxBCs({unknown_boundary}, designed_flux.scalarCoef());

  // Complete the solver setup
  thermal_solver->completeSetup();

  // Initialize the output
  thermal_solver->initializeOutput(inlet.getGlobalTable().get<serac::OutputType>(), "thermal_inverse");

  return thermal_solver;
}

}  // namespace serac

namespace lido {

serac::FiniteElementState setupDesignedFlux(axom::inlet::Inlet& inlet, std::shared_ptr<mfem::ParMesh> mesh)
{
  // Set up the finite element state for the optimized (inverse) flux
  auto initial_flux_guess = inlet["initial_top_flux_guess"].get<serac::input::CoefficientInputOptions>();
  auto flux_guess_coef    = initial_flux_guess.constructScalar();

  serac::FiniteElementState designed_flux(
      *mesh,
      serac::FiniteElementState::Options{
          .order = 1, .coll = std::make_unique<mfem::H1_FECollection>(1, mesh->Dimension()), .name = "designed_flux"});
  designed_flux.project(*flux_guess_coef);

  return designed_flux;
}

serac::mfem_ext::TransformedScalarCoefficient computeAdjointLoad(axom::inlet::Inlet& inlet, serac::FiniteElementState&,
                                                                 mfem::GradientGridFunctionCoefficient& grad_temp,
                                                                 mfem::VectorCoefficient&               mesh_normal)
{
  auto computed_flux_coef = std::make_shared<mfem::InnerProductCoefficient>(mesh_normal, grad_temp);

  auto exact_flux_coef_options = inlet["experimental_flux_coef"].get<serac::input::CoefficientInputOptions>();
  std::shared_ptr<mfem::Coefficient> exact_flux_coef(exact_flux_coef_options.constructScalar());

  auto misfit_coef = serac::mfem_ext::TransformedScalarCoefficient(
      exact_flux_coef, computed_flux_coef,
      [](double exact_flux, double computed_flux) { return -2.0 * (computed_flux - exact_flux); });

  return misfit_coef;
}

std::unique_ptr<mfem::ParLinearForm> assembleSensitivityForm(axom::inlet::Inlet& inlet, serac::FiniteElementState& temp,
                                                             serac::FiniteElementState&         adjoint,
                                                             serac::FiniteElementState&         designed_flux,
                                                             std::shared_ptr<mfem::Coefficient> sensitivity_coef)
{
  // Make the sensitivity coefficient
  double epsilon          = inlet["epsilon"];
  int    unknown_boundary = inlet["unknown_boundary"];

  sensitivity_coef = std::make_shared<serac::mfem_ext::TransformedScalarCoefficient>(
      adjoint.scalarCoef(), designed_flux.scalarCoef(), [epsilon](double adjoint_value, double designed_flux_value) {
        return 2.0 * epsilon * designed_flux_value - adjoint_value;
      });

  auto             sensitivity_form = std::make_unique<mfem::ParLinearForm>(&designed_flux.space());
  mfem::Array<int> markers(temp.mesh().bdr_attributes.Max());
  markers                       = 0;
  markers[unknown_boundary - 1] = 1;
  sensitivity_form->AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(*sensitivity_coef));

  return sensitivity_form;
}

}  // namespace lido

int main(int argc, char* argv[])
{
  auto [num_procs, rank] = serac::initialize(argc, argv);

  // Read the command line and fill inlet
  axom::sidre::DataStore         datastore;
  std::shared_ptr<mfem::ParMesh> mesh;
  auto                           inlet = serac::initializeInletAndMesh(argc, argv, rank, datastore, mesh);

  // Set up the finite element state containing the designed flux field
  auto designed_flux = lido::setupDesignedFlux(inlet, mesh);

  // Create the forward physics solver object
  auto thermal_solver = serac::setupForwardSolver(inlet, mesh, designed_flux);

  // Solve the forward physics
  double dt = 1.0;
  thermal_solver->advanceTimestep(dt);

  // Compute the adjoint load
  //
  // NOTE: this should be replaced by weak_form
  // NOTE: this computes an essential boundary coefficient, but it could also assemble a load vector for other
  // opimization scenario

  // NOTE: These coefficients have to be defined here for MFEM memory management reasons. It should get
  // cleaned up when we move away from coefficients and instead use weak_forms
  mfem::GradientGridFunctionCoefficient grad_temp_coef(&thermal_solver->temperature().gridFunc());

  auto normal_coef = serac::makeNormalCoef(mesh);

  auto misfit_coef = lido::computeAdjointLoad(inlet, thermal_solver->temperature(), grad_temp_coef, normal_coef);

  // Set the adjoint solve boundary conditions and/or loads
  int measured_boundary = inlet["measured_boundary"];
  thermal_solver->setAdjointEssentialBCs({measured_boundary}, misfit_coef);

  // could have a similar setAdjointLoad or possibly solveAdjoint(adjoint_load)
  thermal_solver->solveAdjoint();

  // Generate the form of the sensitivity
  // This should also get replaced by a weak form
  std::shared_ptr<mfem::Coefficient> sensitivity_coef;

  auto sensitivity_form = lido::assembleSensitivityForm(
      inlet, thermal_solver->temperature(), thermal_solver->adjointTemperature(), designed_flux, sensitivity_coef);

  // Compute the discrete sensitivity
  std::unique_ptr<mfem::HypreParVector> discrete_sensitivity_vector(sensitivity_form->ParallelAssemble());

  serac::exitGracefully();
}