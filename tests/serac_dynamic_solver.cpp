// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <fstream>

#include "mfem.hpp"
#include "numerics/mesh_utils.hpp"
#include "physics/nonlinear_solid.hpp"
#include "serac_config.hpp"

namespace serac {

void initialDeformation(const mfem::Vector& x, mfem::Vector& y);

void initialVelocity(const mfem::Vector& x, mfem::Vector& v);

const IterativeSolverParameters default_dyn_linear_params = {.rel_tol     = 1.0e-4,
                                                             .abs_tol     = 1.0e-8,
                                                             .print_level = 0,
                                                             .max_iter    = 500,
                                                             .lin_solver  = LinearSolver::GMRES,
                                                             .prec        = HypreBoomerAMGPrec{}};

const IterativeSolverParameters default_dyn_oper_linear_params = {
    .rel_tol     = 1.0e-4,
    .abs_tol     = 1.0e-8,
    .print_level = 0,
    .max_iter    = 500,
    .lin_solver  = LinearSolver::GMRES,
    .prec        = HypreSmootherPrec{mfem::HypreSmoother::Jacobi}};

const NonlinearSolverParameters default_dyn_nonlinear_params = {
    .rel_tol = 1.0e-4, .abs_tol = 1.0e-8, .max_iter = 500, .print_level = 1};

const NonlinearSolid::SolverParameters default_dynamic = {
    default_dyn_linear_params, default_dyn_nonlinear_params,
    NonlinearSolid::DynamicSolverParameters{TimestepMethod::AverageAcceleration,
                                            DirichletEnforcementMethod::RateControl, default_dyn_oper_linear_params}};

TEST(dynamic_solver, dyn_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  std::set<int> ess_bdr = {1};

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // initialize the dynamic solver object
  NonlinearSolid dyn_solver(1, pmesh, default_dynamic);
  dyn_solver.setDisplacementBCs(ess_bdr, deform);
  dyn_solver.setHyperelasticMaterialParameters(0.25, 5.0);
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 1.0;

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(1.4225, x_norm, 0.0001);
  EXPECT_NEAR(0.2252, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(dynamic_solver, dyn_direct_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  std::set<int> ess_bdr = {1};

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // initialize the dynamic solver object
  auto solver_params                 = default_dynamic;
  solver_params.H_lin_params         = DirectSolverParameters{0};
  solver_params.dyn_params->M_params = DirectSolverParameters{0};
  NonlinearSolid dyn_solver(1, pmesh, solver_params);
  dyn_solver.setDisplacementBCs(ess_bdr, deform);
  dyn_solver.setHyperelasticMaterialParameters(0.25, 5.0);
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 1.0;

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(1.4225, x_norm, 0.0001);
  EXPECT_NEAR(0.2252, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef MFEM_USE_SUNDIALS
TEST(dynamic_solver, dyn_linesearch_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  std::set<int> ess_bdr = {1};

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // Set the nonlinear solver parameters
  auto params                          = default_dynamic;
  params.H_nonlin_params.nonlin_solver = NonlinearSolver::KINBacktrackingLineSearch;

  // initialize the dynamic solver object
  NonlinearSolid dyn_solver(1, pmesh, params);
  dyn_solver.setDisplacementBCs(ess_bdr, deform);
  dyn_solver.setHyperelasticMaterialParameters(0.25, 5.0);
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 1.0;

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(1.4225, x_norm, 0.0001);
  EXPECT_NEAR(0.2252, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_SUNDIALS

#ifdef MFEM_USE_AMGX
TEST(dynamic_solver, dyn_amgx_solve)
{
  MPI_Barrier(MPI_COMM_WORLD);

  // Open the mesh
  std::string mesh_file = std::string(SERAC_REPO_DIR) + "/data/meshes/beam-hex.mesh";

  auto pmesh = buildMeshFromFile(mesh_file, 1, 0);

  int dim = pmesh->Dimension();

  std::set<int> ess_bdr = {1};

  auto visc   = std::make_unique<mfem::ConstantCoefficient>(0.0);
  auto deform = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialDeformation);
  auto velo   = std::make_shared<mfem::VectorFunctionCoefficient>(dim, initialVelocity);

  // initialize the dynamic solver object
  auto  dyn_solver_params = default_dynamic;
  auto& H_iter_params     = std::get<IterativeSolverParameters>(dyn_solver_params.H_lin_params);
  H_iter_params.prec      = AMGXPrec{};
  NonlinearSolid dyn_solver(1, pmesh, dyn_solver_params);
  dyn_solver.setDisplacementBCs(ess_bdr, deform);
  dyn_solver.setHyperelasticMaterialParameters(0.25, 5.0);
  dyn_solver.setViscosity(std::move(visc));
  dyn_solver.setDisplacement(*deform);
  dyn_solver.setVelocity(*velo);

  // Initialize the VisIt output
  dyn_solver.initializeOutput(serac::OutputType::VisIt, "dynamic_solid");

  // Construct the internal dynamic solver data structures
  dyn_solver.completeSetup();

  double t       = 0.0;
  double t_final = 6.0;
  double dt      = 3.0;

  // Ouput the initial state
  dyn_solver.outputState();

  // Perform time-integration
  // (looping over the time iterations, ti, with a time-step dt).
  bool last_step = false;
  for (int ti = 1; !last_step; ti++) {
    double dt_real = std::min(dt, t_final - t);
    t += dt_real;
    last_step = (t >= t_final - 1e-8 * dt);

    dyn_solver.advanceTimestep(dt_real);
  }

  // Output the final state
  dyn_solver.outputState();

  // Check the final displacement and velocity L2 norms
  mfem::Vector zero(dim);
  zero = 0.0;
  mfem::VectorConstantCoefficient zerovec(zero);

  double v_norm = dyn_solver.velocity().gridFunc().ComputeLpError(2.0, zerovec);
  double x_norm = dyn_solver.displacement().gridFunc().ComputeLpError(2.0, zerovec);

  EXPECT_NEAR(12.86733, x_norm, 0.0001);
  EXPECT_NEAR(0.22298, v_norm, 0.0001);

  MPI_Barrier(MPI_COMM_WORLD);
}
#endif  // MFEM_USE_AMGX

void initialDeformation(const mfem::Vector& /*x*/, mfem::Vector& u) { u = 0.0; }

void initialVelocity(const mfem::Vector& x, mfem::Vector& v)
{
  const int    dim = x.Size();
  const double s   = 0.1 / 64.;

  v          = 0.0;
  v(dim - 1) = s * x(0) * x(0) * (8.0 - x(0));
  v(0)       = -s * x(0) * x(0);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  UnitTestLogger logger;  // create & initialize test logger, finalized when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
