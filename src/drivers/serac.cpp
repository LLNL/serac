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

#include "mfem.hpp"
#include "solvers/nonlinear_solid_solver.hpp"
#include "coefficients/loading_functions.hpp"
#include "coefficients/traction_coefficient.hpp"
#include <memory>
#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
  // Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // mesh
  const char *mesh_file = "../../data/beam-hex.mesh";

  // serial and parallel refinement levels
  int ser_ref_levels = 0;
  int par_ref_levels = 0;

  // polynomial interpolation order
  int order = 1;

  // newton input args
  double newton_rel_tol = 1.0e-2;
  double newton_abs_tol = 1.0e-4;
  int newton_iter = 500;

  // solver input args
  bool gmres_solver = true;
  bool slu_solver = false;

  // neo-Hookean material parameters
  double mu = 0.25;
  double K = 5.0;

  // loading parameters
  double tx = 0.0;
  double ty = 1.0e-3;
  double tz = 0.0;

  double t_final = 1.0;
  double dt = 0.25;

  // specify all input arguments
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                 "Number of times to refine the mesh uniformly in serial.");
  args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                 "Number of times to refine the mesh uniformly in parallel.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&mu, "-mu", "--shear-modulus",
                 "Shear modulus in the Neo-Hookean hyperelastic model.");
  args.AddOption(&K, "-K", "--bulk-modulus",
                 "Bulk modulus in the Neo-Hookean hyperelastic model.");
  args.AddOption(&tx, "-tx", "--traction-x",
                 "Cantilever tip traction in the x direction.");
  args.AddOption(&ty, "-ty", "--traction-y",
                 "Cantilever tip traction in the y direction.");
  args.AddOption(&tz, "-tz", "--traction-z",
                 "Cantilever tip traction in the z direction.");
  args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                 "--no-superlu", "Use the SuperLU Solver.");
  args.AddOption(&gmres_solver, "-gmres", "--gmres", "-no-gmres", "--no-gmres",
                 "Use gmres, otherwise minimum residual is used.");
  args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                 "Relative tolerance for the Newton solve.");
  args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                 "Absolute tolerance for the Newton solve.");
  args.AddOption(&newton_iter, "-it", "--newton-iterations",
                 "Maximum iterations for the Newton solve.");
  args.AddOption(&t_final, "-tf", "--t-final",
                 "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step",
                 "Time step.");

  // Parse the arguments and check if they are good
  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(std::cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(std::cout);
  }

  // Open the mesh
  mfem::Mesh *mesh;
  std::ifstream imesh(mesh_file);
  if (!imesh) {
    if (myid == 0) {
      std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
    }
    MPI_Finalize();
    return 2;
  }

  mesh = new mfem::Mesh(imesh, 1, 1, true);
  imesh.close();

  // declare pointer to parallel mesh object
  mfem::ParMesh *pmesh = NULL;

  // mesh refinement if specified in input
  for (int lev = 0; lev < ser_ref_levels; lev++) {
    mesh->UniformRefinement();
  }

  pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
  for (int lev = 0; lev < par_ref_levels; lev++) {
    pmesh->UniformRefinement();
  }

  delete mesh;

  int dim = pmesh->Dimension();

  // Define the solid solver object
  NonlinearSolidSolver solid_solver(order, pmesh);

  // Project the initial and reference configuration functions onto the appropriate grid functions
  mfem::VectorCoefficient *defo_coef = new mfem::VectorFunctionCoefficient(dim, InitialDeformation);

  mfem::Vector velo(dim);
  velo = 0.0;
  mfem::VectorConstantCoefficient *velo_coef = new mfem::VectorConstantCoefficient(velo);

  mfem::Array<mfem::VectorCoefficient*> coefs(2);
  coefs[0] = defo_coef;
  coefs[1] = velo_coef;

  // initialize x_cur, boundary condition, deformation, and
  // incremental nodal displacment grid functions by projection the
  // VectorFunctionCoefficient function onto them
  solid_solver.ProjectState(coefs);

  // define a boundary attribute array and initialize to 0
  mfem::Array<int> ess_bdr;
  ess_bdr.SetSize(pmesh->bdr_attributes.Max());
  ess_bdr = 0;

  // boundary attribute 1 (index 0) is fixed (Dirichlet)
  ess_bdr[0] = 1;

  // define the displacement vector
  mfem::Vector disp(dim);
  disp = 0.0;
  mfem::VectorConstantCoefficient disp_coef(disp);

  mfem::Array<int> trac_bdr;
  trac_bdr.SetSize(pmesh->bdr_attributes.Max());

  trac_bdr = 0;
  trac_bdr[1] = 1;

  // define the traction vector
  mfem::Vector traction(dim);
  traction(0) = tx;
  traction(1) = ty;
  if (dim == 3) {
    traction(2) = tz;
  }

  VectorScaledConstantCoefficient traction_coef(traction);

  // Set the boundary condition information
  solid_solver.SetDisplacementBCs(ess_bdr, &disp_coef);
  solid_solver.SetTractionBCs(trac_bdr, &traction_coef);

  // Set the material parameters
  solid_solver.SetHyperelasticMaterialParameters(mu, K);

  // Set the linear solver parameters
  LinearSolverParameters params;
  params.rel_tol = newton_rel_tol;
  params.abs_tol = newton_abs_tol;
  params.print_level = 0;
  params.max_iter = newton_iter;
  if (gmres_solver == true) {
    params.prec = Preconditioner::BoomerAMG;
    params.lin_solver = LinearSolver::GMRES;
  } else {
    params.prec = Preconditioner::Jacobi;
    params.lin_solver = LinearSolver::MINRES;
  }
  solid_solver.SetLinearSolverParameters(params);
  
  // Set the time step method
  solid_solver.SetTimestepper(TimestepMethod::QuasiStatic);

  // Complete the solver setup
  solid_solver.CompleteSetup();

  // initialize/set the time
  double t = 0.0;

  bool last_step = false;

  mfem::Array<std::string> names(2);
  names[0] = "Deformation";
  names[1] = "Velocity";

  solid_solver.InitializeOutput(OutputType::VisIt, "serac", names);

  // enter the time step loop. This was modeled after example 10p.
  for (int ti = 1; !last_step; ti++) {

    double dt_real = std::min(dt, t_final - t);
    // compute current time
    t = t + dt_real;

    if (myid == 0) {
      std::cout << "step " << ti << ", t = " << t << std::endl;
    }

    traction_coef.SetScale(t);

    // Solve the Newton system
    solid_solver.AdvanceTimestep(dt_real);

    solid_solver.OutputState();

    last_step = (t >= t_final - 1e-8*dt);
  }

  // Free the used memory.
  delete pmesh;

  MPI_Finalize();

  return 0;
}
