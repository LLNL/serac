// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <fstream>
#include <functional>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/equation_solver.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"

using namespace serac;
using namespace serac::mfem_ext;

using param_t = std::tuple<NonlinearSolver, LinearSolver, Preconditioner>;

class EquationSolverSuite : public testing::TestWithParam<param_t> {
protected:
  void            SetUp() override { std::tie(nonlin_solver, lin_solver, precond) = GetParam(); }
  NonlinearSolver nonlin_solver;
  LinearSolver    lin_solver;
  Preconditioner  precond;
};

TEST_P(EquationSolverSuite, All)
{
  auto mesh  = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL);
  auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

  constexpr int p   = 1;
  constexpr int dim = 2;

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec);

  mfem::HypreParVector x_exact(&fes);
  mfem::HypreParVector x_computed(&fes);

  std::unique_ptr<mfem::HypreParMatrix> J;

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = H1<p>;
  using trial_space = H1<p>;

  // Construct the new functional object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fes, {&fes});

  x_exact.Randomize(0);

  residual.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0>{},
      [&](auto, auto scalar) {
        auto [u, du_dx] = scalar;
        auto source     = 0.5 * sin(u);
        auto flux       = du_dx;
        return serac::tuple{source, flux};
      },
      pmesh);

  StdFunctionOperator residual_opr(
      fes.TrueVSize(),
      [&x_exact, &residual](const mfem::Vector& x, mfem::Vector& r) {
        // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
        // tracking strategy
        // See https://github.com/mfem/mfem/issues/3531
        const mfem::Vector res = residual(x);

        r = res;
        r -= residual(x_exact);
      },
      [&residual, &J](const mfem::Vector& x) -> mfem::Operator& {
        auto [val, grad] = residual(differentiate_wrt(x));
        J                = assemble(grad);
        return *J;
      });

  const LinearSolverOptions lin_opts = {.linear_solver  = lin_solver,
                                        .preconditioner = precond,
                                        .relative_tol   = 1.0e-10,
                                        .absolute_tol   = 1.0e-12,
                                        .max_iterations = 500,
                                        .print_level    = 1};

  const NonlinearSolverOptions nonlin_opts = {.nonlin_solver  = nonlin_solver,
                                              .relative_tol   = 1.0e-10,
                                              .absolute_tol   = 1.0e-12,
                                              .max_iterations = 100,
                                              .print_level    = 1};

  EquationSolver eq_solver(nonlin_opts, lin_opts);

  eq_solver.setOperator(residual_opr);

  eq_solver.solve(x_computed);

  for (int i = 0; i < x_computed.Size(); ++i) {
    EXPECT_LT(std::abs((x_computed(i) - x_exact(i))) / x_exact(i), 1.0e-6);
  }
}

#ifdef MFEM_USE_SUNDIALS
INSTANTIATE_TEST_SUITE_P(
    AllEquationSolverTests, EquationSolverSuite,
    testing::Combine(testing::Values(NonlinearSolver::Newton, NonlinearSolver::LBFGS, NonlinearSolver::KINFullStep,
                                     NonlinearSolver::KINBacktrackingLineSearch, NonlinearSolver::KINPicard),
                     testing::Values(LinearSolver::CG, LinearSolver::GMRES, LinearSolver::SuperLU),
                     testing::Values(Preconditioner::HypreJacobi, Preconditioner::HypreL1Jacobi,
                                     Preconditioner::HypreGaussSeidel, Preconditioner::HypreAMG,
                                     Preconditioner::HypreILU)));
#else
INSTANTIATE_TEST_SUITE_P(AllEquationSolverTests, EquationSolverSuite,
                         testing::Combine(testing::Values(NonlinearSolver::Newton, NonlinearSolver::LBFGS),
                                          testing::Values(LinearSolver::CG, LinearSolver::GMRES, LinearSolver::SuperLU),
                                          testing::Values(Preconditioner::HypreJacobi, Preconditioner::HypreL1Jacobi,
                                                          Preconditioner::HypreGaussSeidel, Preconditioner::HypreAMG,
                                                          Preconditioner::HypreILU)));
#endif

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
