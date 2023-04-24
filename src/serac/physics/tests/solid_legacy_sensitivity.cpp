// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include <mfem.hpp>
#include <mpi.h>

#include <serac/physics/solid_legacy.hpp>
#include <serac/physics/state/state_manager.hpp>
#include <serac/physics/state/finite_element_state.hpp>
#include <serac/physics/state/finite_element_dual.hpp>

TEST(SolidLegacy, FiniteDiff)
{
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_sensitivity");

  // Build a mesh
  ::mfem::Mesh cuboid = ::mfem::Mesh::MakeCartesian2D(3, 3, ::mfem::Element::Type::QUADRILATERAL, true, 0.1, 0.1);

  // Parallelize mesh and give it to serac
  auto* mesh = serac::StateManager::setMesh(std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid));

  // Setup some solver options
  serac::LinearSolverOptions const default_linear_options = {
      .linear_solver  = serac::LinearSolver::GMRES,
      .preconditioner = serac::Preconditioner::HypreAMG,
      .relative_tol   = 1.0e-8,
      .absolute_tol   = 1.0e-12,
      .max_iterations = 500,
      .print_level    = 0,
  };

  serac::NonlinearSolverOptions const default_nonlinear_options = {
      .relative_tol = 1.0e-6, .absolute_tol = 1.0e-11, .max_iterations = 500, .print_level = 1};

  serac::SolidLegacy::SolverOptions const solverOptions = {default_linear_options, default_nonlinear_options};

  // Make the serac SolidLegacy object
  int                order = 1;
  serac::SolidLegacy solid(order, solverOptions, serac::GeometricNonlinearities::Off, serac::FinalMeshOption::Reference,
                           "hi", mesh);

  // Set up some fixed displacement conditions
  std::set<int> fixedBC{1};
  mfem::Vector  zeroDisplacement(mesh->SpaceDimension());
  zeroDisplacement = 0.0;
  solid.setDisplacementBCs(fixedBC, std::make_unique<mfem::VectorConstantCoefficient>(zeroDisplacement));

  // Set up a load
  mfem::Vector appliedTraction(2);
  appliedTraction[0]               = 1.0;
  appliedTraction[1]               = 0.0;
  bool          computeOnReference = true;
  std::set<int> tractionBC{4};
  solid.setTractionBCs(tractionBC, std::make_unique<mfem::VectorConstantCoefficient>(appliedTraction),
                       computeOnReference);

  // Set up a material properties defined by a L2 degree 0 finite element space
  mfem::L2_FECollection       l2fec(0, mesh->Dimension());
  mfem::ParFiniteElementSpace l2fespace_scalar(mesh, &l2fec, 1);

  // Constant bulk modulus everywhere
  double                bulkModulusValue = 1.1;
  mfem::ParGridFunction bulkModulus(&l2fespace_scalar);
  bulkModulus = bulkModulusValue;

  // Constant shear modulus everywhere
  double shearModulusValue = 1.2;

  mfem::ParGridFunction shearModulus(&l2fespace_scalar);
  shearModulus = shearModulusValue;

  bool materialNonlinearity = false;
  solid.setMaterialParameters(std::make_unique<mfem::GridFunctionCoefficient>(&shearModulus),
                              std::make_unique<mfem::GridFunctionCoefficient>(&bulkModulus), materialNonlinearity);
  // Tell serac I am done
  solid.completeSetup();

  // Solve for displacement
  double timestep = 1.0;
  solid.advanceTimestep(timestep);

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjointLoad(&solid.displacement().space());
  adjointLoad = 1.0;

  // Solve adjoint system given this made up adjoint load
  serac::FiniteElementDual assembledAdjointLoad(solid.displacement().space(), "adjointLoad");
  mfem::HypreParVector*    assembledVector = adjointLoad.ParallelAssemble();
  assembledAdjointLoad                     = *assembledVector;
  delete assembledVector;

  solid.solveAdjoint({{"displacement", assembledAdjointLoad}});

  // Ask for bulk modulus sensitivity
  serac::FiniteElementDual const& bulkModulusSensitivity  = solid.bulkModulusSensitivity(&l2fespace_scalar);
  serac::FiniteElementDual const& shearModulusSensitivity = solid.shearModulusSensitivity(&l2fespace_scalar);

  // Perform finite difference on each bulk modulus value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the displacement
  double eps = 1.0E-6;
  for (int ix = 0; ix < bulkModulus.Size(); ++ix) {
    // Perturb bulk sensitivity
    bulkModulus[ix] = bulkModulusValue + eps;
    solid.advanceTimestep(timestep);
    mfem::ParGridFunction displacementPlus = solid.displacement().gridFunction();

    bulkModulus[ix] = bulkModulusValue - eps;
    solid.advanceTimestep(timestep);
    mfem::ParGridFunction displacementMinus = solid.displacement().gridFunction();

    // Reset to the original bulk modulus value
    bulkModulus[ix] = bulkModulusValue;

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction du_dbulk(&solid.displacement().space());
    for (int ix2 = 0; ix2 < displacementPlus.Size(); ++ix2) {
      du_dbulk(ix2) = (displacementPlus(ix2) - displacementMinus(ix2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dbulk = adjointLoad(du_dbulk);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dbulk: {}", dqoi_dbulk));
    SLIC_INFO(axom::fmt::format("bulkModulusSensitivity: {}", bulkModulusSensitivity(ix)));

    EXPECT_NEAR((bulkModulusSensitivity(ix) - dqoi_dbulk) / dqoi_dbulk, 0.0, 1.0e-3);
  }

  bulkModulus = bulkModulusValue;

  for (int ix = 0; ix < shearModulus.Size(); ++ix) {
    // Perturb bulk sensitivity
    shearModulus[ix] = shearModulusValue + eps;
    solid.advanceTimestep(timestep);
    auto displacementPlus = solid.displacement().gridFunction();

    shearModulus[ix] = shearModulusValue - eps;
    solid.advanceTimestep(timestep);
    auto displacementMinus = solid.displacement().gridFunction();

    // Reset to the original shear modulus value
    shearModulus[ix] = shearModulusValue;

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction du_dbulk(&solid.displacement().space());
    for (int ix2 = 0; ix2 < displacementPlus.Size(); ++ix2) {
      du_dbulk(ix2) = (displacementPlus(ix2) - displacementMinus(ix2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dshear = adjointLoad(du_dbulk);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dshear: {}", dqoi_dshear));
    SLIC_INFO(axom::fmt::format("shearModulusSensitivity: {}", shearModulusSensitivity(ix)));
    EXPECT_NEAR((shearModulusSensitivity(ix) - dqoi_dshear) / dqoi_dshear, 0.0, 1.0e-3);
  }
}

TEST(SolidLegacy, MultipleDesignSpaces)
{
  // Initialize the datastore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_sensitivity_design_spaces");

  // Create a mesh and pass it to the datastore
  ::mfem::Mesh cuboid = ::mfem::Mesh::MakeCartesian3D(2, 2, 2, ::mfem::Element::Type::HEXAHEDRON, 4.0, 2.0, 10);
  auto         mesh   = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, cuboid);
  serac::StateManager::setMesh(std::move(mesh));

  // Setup setup the solid module
  serac::LinearSolverOptions const default_linear_options = {.linear_solver  = serac::LinearSolver::GMRES,
                                                             .preconditioner = serac::Preconditioner::HypreAMG,
                                                             .relative_tol   = 1.0e-12,
                                                             .absolute_tol   = 1.0e-12,
                                                             .max_iterations = 500,
                                                             .print_level    = 0};

  serac::NonlinearSolverOptions const default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-6, .max_iterations = 3, .print_level = 1};

  serac::SolidLegacy::SolverOptions const solverOptions = {default_linear_options, default_nonlinear_options};
  int                                     order         = 1;
  serac::SolidLegacy                      solid(order, solverOptions, serac::GeometricNonlinearities::Off,
                           serac::FinalMeshOption::Reference);

  // Set the fixed Dirichlet BCs
  std::set<int> fixedBCs{1};
  mfem::Vector  zeroDisplacement(serac::StateManager::mesh().SpaceDimension());
  zeroDisplacement = 0.0;
  solid.setDisplacementBCs(fixedBCs, std::make_unique<mfem::VectorConstantCoefficient>(zeroDisplacement));

  // Set a fixed traction BC
  std::set<int> tractionBCs{2};
  bool          computeOnReference = true;
  mfem::Vector  appliedTraction(3);
  appliedTraction[0] = 1.0;
  appliedTraction[1] = 1.0;
  appliedTraction[2] = 1.0;
  solid.setTractionBCs(tractionBCs, std::make_unique<mfem::VectorConstantCoefficient>(appliedTraction),
                       computeOnReference);

  // Use a linear elastic material
  bool materialNonlinearity = false;

  // Create various FE spaces for the parameter and adjoint fields
  mfem::H1_FECollection       h1fec(1, serac::StateManager::mesh().Dimension());
  mfem::ParFiniteElementSpace h1fespace_scalar(&serac::StateManager::mesh(), &h1fec, 1, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace h1fespace_vector(&serac::StateManager::mesh(), &h1fec,
                                               serac::StateManager::mesh().SpaceDimension(), mfem::Ordering::byNODES);
  mfem::L2_FECollection       l2fec(0, serac::StateManager::mesh().Dimension());
  mfem::ParFiniteElementSpace l2fespace_scalar(&serac::StateManager::mesh(), &l2fec, 1, mfem::Ordering::byNODES);
  std::vector<std::reference_wrapper<mfem::ParFiniteElementSpace>> materialSpaces{h1fespace_scalar, l2fespace_scalar};

  // Compute the sensitivities for both H1 and L2 fields
  for (mfem::ParFiniteElementSpace& materialSpace : materialSpaces) {
    // Create grid function representations of the shear and bulk modulus on the current parameter FE space
    mfem::ParGridFunction shearModulus(&materialSpace);
    shearModulus = 1.1;
    mfem::ParGridFunction bulkModulus(&materialSpace);
    bulkModulus = 1.2;

    // Set the material parameters in the solid solver
    solid.setMaterialParameters(std::make_unique<mfem::GridFunctionCoefficient>(&shearModulus),
                                std::make_unique<mfem::GridFunctionCoefficient>(&bulkModulus), materialNonlinearity);

    // Run the solid solver
    solid.completeSetup();
    double timestep = 1.0;
    solid.advanceTimestep(timestep);

    // Create a dummy adjoint load and compute the sensitivities
    serac::FiniteElementDual adjointLoad(h1fespace_vector, "adjoint_load");
    adjointLoad = 1.1;
    solid.solveAdjoint({{"displacement", adjointLoad}});
    solid.shearModulusSensitivity(shearModulus.ParFESpace());
    solid.shearModulusSensitivity(bulkModulus.ParFESpace());
  }
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
