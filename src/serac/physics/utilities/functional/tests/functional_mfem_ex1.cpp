// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/utilities/functional/functional.hpp"

using namespace serac;

constexpr int dim = 2;
constexpr int p   = 1;

int main(int argc, char*argv[])
{

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Initialize the logger
  axom::slic::SimpleLogger logger;

  // Open, refine, and distribute the star mesh
  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  auto        mesh = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  // Create standard MFEM FE collections and space for H1
  auto fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(mesh.get(), &fec);

  // Create a grid function to contain the solution of the PDE
  mfem::ParGridFunction u_global(&fespace);
  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, &fespace);

  // Set the essential boundaries
  mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 1;
  residual.SetEssentialBC(ess_bdr);

  // Add the total domain residual term to the functional
  residual.AddDomainIntegral(
      Dimension<dim>{},
      [=]([[maybe_unused]] auto x, auto temperature) {
        // Get the value and the gradient from the input tuple
        auto [u, du_dx] = temperature;

        // Set the uniform source term (comes from the linear form in MFEM ex1)
        auto source     = -1.0;

        // Set the flux term (comes from the bilinear form in MFEM ex1)
        auto flux       = du_dx;

        // Return the source and the flux as a tuple 
        return serac::tuple{source, flux};
      },
      *mesh);

  // Initialize the solution vector with the essential boundary values
  U = 0.0;

  // Create a zero vector to drive the nonlinear residual to zero
  mfem::Vector zero(U);
  zero = 0.0;

  // Set up the linear and nonlinear solvers
  mfem::CGSolver lin_solver(MPI_COMM_WORLD);
  lin_solver.SetPrintLevel(1);
  lin_solver.SetAbsTol(1.0e-8);
  lin_solver.SetRelTol(1.0e-8);
  lin_solver.SetMaxIter(1000);

  mfem::NewtonSolver nonlin_solver(MPI_COMM_WORLD);
  nonlin_solver.SetOperator(residual);
  nonlin_solver.SetSolver(lin_solver);
  nonlin_solver.SetPrintLevel(1);
  nonlin_solver.SetAbsTol(1.0e-12);
  nonlin_solver.SetRelTol(1.0e-7);

  // Drive the residual determined by functional to zero
  nonlin_solver.Mult(zero, U);

  /* 

  // This shows how functional could be used in a purely linear context

  // Calculate the initial residual for the RHS of the linear solve
  mfem::Vector res = residual(U);
  res *= -1.0;

  // Get the gradient of the residual evaluation
  mfem::Operator& grad = residual.GetGradient(U);

  // Solve the linear system using CG
  mfem::CG(grad, res, U, 1);

  */

  // Output the grid function
  u_global.SetFromTrueDofs(U);
  u_global.Save("sol");
  mesh->Save("mesh");

  mfem::Vector res = residual(U);
  fmt::print("L2 norm of residual post-solve = {}\n", res.Norml2());

  // Close out MPI
  MPI_Finalize();
}
