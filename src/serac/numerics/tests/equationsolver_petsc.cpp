// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <fstream>
#include <ostream>
#include <functional>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/numerics/equation_solver.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/infrastructure/initialize.hpp"

using namespace serac;
using namespace serac::mfem_ext;

using param_t = std::tuple<NonlinearSolver, LinearSolver, Preconditioner, PetscPCType>;

class EquationSolverSuite : public testing::TestWithParam<param_t> {
protected:
  void            SetUp() override { std::tie(nonlin_solver, lin_solver, precond, pc_type) = GetParam(); }
  NonlinearSolver nonlin_solver;
  LinearSolver    lin_solver;
  Preconditioner  precond;
  PetscPCType     pc_type;
};

TEST_P(EquationSolverSuite, All)
{
  auto mesh  = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL);
  auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

  pmesh.EnsureNodes();
  pmesh.ExchangeFaceNbrData();

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
      [&](double /*t*/, auto, auto scalar) {
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

        double dummy_time = 0.0;

        const mfem::Vector res = residual(dummy_time, x);

        r = res;
        r -= residual(dummy_time, x_exact);
      },
      [&residual, &J](const mfem::Vector& x) -> mfem::Operator& {
        double dummy_time = 0.0;
        auto [val, grad]  = residual(dummy_time, differentiate_wrt(x));
        J                 = assemble(grad);
        return *J;
      });

  const LinearSolverOptions lin_opts = {.linear_solver        = lin_solver,
                                        .preconditioner       = precond,
                                        .petsc_preconditioner = pc_type,
                                        .relative_tol         = 1.0e-10,
                                        .absolute_tol         = 1.0e-12,
                                        .max_iterations       = 500,
                                        .print_level          = 1};

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

std::string nonlinearName(const NonlinearSolver& s)
{
  if (s == NonlinearSolver::Newton) return "Newton";
  if (s == NonlinearSolver::KINFullStep) return "KINFullStep";
  if (s == NonlinearSolver::KINBacktrackingLineSearch) return "KINBacktrackingLineSearch";
  if (s == NonlinearSolver::KINPicard) return "KINPicard";
  if (s == NonlinearSolver::PetscNewton) return "PetscNewton";
  if (s == NonlinearSolver::PetscNewtonBacktracking) return "PetscNewtonBacktracking";
  if (s == NonlinearSolver::PetscNewtonCriticalPoint) return "PetscNewtonCriticalPoint";
  if (s == NonlinearSolver::PetscTrustRegion) return "PetscTrustRegion";
  return "";
}

std::string linearName(const LinearSolver& s)
{
  if (s == LinearSolver::CG) return "CG";
  if (s == LinearSolver::GMRES) return "GMRES";
  if (s == LinearSolver::PetscCG) return "PetscCG";
  if (s == LinearSolver::PetscGMRES) return "PetscGMRES";
  return "";
}

std::string PetscPCName(const PetscPCType& s)
{
  if (s == PetscPCType::JACOBI) return "JACOBI";
  if (s == PetscPCType::JACOBI_L1) return "JACOBI_L1";
  if (s == PetscPCType::JACOBI_ROWSUM) return "JACOBI_ROWSUM";
  if (s == PetscPCType::JACOBI_ROWMAX) return "JACOBI_ROWMAX";
  if (s == PetscPCType::PBJACOBI) return "PBJACOBI";
  if (s == PetscPCType::BJACOBI) return "BJACOBI";
  if (s == PetscPCType::LU) return "LU";
  if (s == PetscPCType::ILU) return "ILU";
  if (s == PetscPCType::CHOLESKY) return "CHOLESKY";
  if (s == PetscPCType::SVD) return "SVD";
  if (s == PetscPCType::ASM) return "ASM";
  if (s == PetscPCType::GASM) return "GASM";
  if (s == PetscPCType::GAMG) return "GAMG";
  if (s == PetscPCType::HMG) return "HMG";
  return "";
}

#ifdef SERAC_USE_SUNDIALS
INSTANTIATE_TEST_SUITE_P(
    AllEquationSolverTests, EquationSolverSuite,
    testing::Combine(testing::Values(NonlinearSolver::Newton, NonlinearSolver::KINFullStep,
                                     NonlinearSolver::KINBacktrackingLineSearch, NonlinearSolver::KINPicard,
                                     NonlinearSolver::PetscNewton, NonlinearSolver::PetscNewtonBacktracking,
                                     NonlinearSolver::PetscNewtonCriticalPoint, NonlinearSolver::PetscTrustRegion),
                     testing::Values(LinearSolver::CG, LinearSolver::GMRES, LinearSolver::PetscCG,
                                     LinearSolver::PetscGMRES),
                     testing::Values(Preconditioner::Petsc),
                     testing::Values(PetscPCType::JACOBI, PetscPCType::JACOBI_L1, PetscPCType::JACOBI_ROWSUM,
                                     PetscPCType::JACOBI_ROWMAX, PetscPCType::PBJACOBI, PetscPCType::BJACOBI,
                                     PetscPCType::LU, PetscPCType::ILU, PetscPCType::CHOLESKY, PetscPCType::SVD,
                                     PetscPCType::ASM, PetscPCType::GASM, PetscPCType::GAMG)),  //, PetscPCType::HMG)),
    [](const testing::TestParamInfo<EquationSolverSuite::ParamType>& test_info) {
      std::string name = nonlinearName(std::get<0>(test_info.param)) + "_" + linearName(std::get<1>(test_info.param)) +
                         "_" + PetscPCName(std::get<3>(test_info.param));
      return name;
    });
#else
INSTANTIATE_TEST_SUITE_P(
    AllEquationSolverTests, EquationSolverSuite,
    testing::Combine(
        testing::Values(NonlinearSolver::Newton, NonlinearSolver::PetscNewton, NonlinearSolver::PetscNewtonBacktracking,
                        NonlinearSolver::PetscNewtonCriticalPoint, NonlinearSolver::PetscTrustRegion),
        testing::Values(LinearSolver::CG, LinearSolver::GMRES, LinearSolver::PetscCG, LinearSolver::PetscGMRES),
        testing::Values(Preconditioner::Petsc),
        testing::Values(PetscPCType::JACOBI, PetscPCType::JACOBI_L1, PetscPCType::JACOBI_ROWSUM,
                        PetscPCType::JACOBI_ROWMAX, PetscPCType::PBJACOBI, PetscPCType::BJACOBI, PetscPCType::LU,
                        PetscPCType::ILU, PetscPCType::CHOLESKY, PetscPCType::SVD, PetscPCType::ASM, PetscPCType::GASM,
                        PetscPCType::GAMG)),  //, PetscPCType::HMG)));
    [](const testing::TestParamInfo<EquationSolverSuite::ParamType>& test_info) {
      std::string name = nonlinearName(std::get<0>(test_info.param)) + "_" + linearName(std::get<1>(test_info.param)) +
                         "_" + PetscPCName(std::get<3>(test_info.param));
      return name;
    });
#endif

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
