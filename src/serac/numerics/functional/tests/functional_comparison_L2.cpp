#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"
#include <gtest/gtest.h>

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;

constexpr bool                 verbose = true;
std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

// this test sets up a toy "thermal" problem where the residual includes contributions
// from a temperature-dependent source term and a temperature-gradient-dependent flux
//
// the same problem is expressed with mfem and functional, and their residuals and gradient action
// are compared to ensure the implementations are in agreement.
template <int p, int dim>
void functional_test(mfem::ParMesh& mesh, L2<p> test, L2<p> trial, Dimension<dim>)
{
  [[maybe_unused]] static constexpr double a       = 1.7;
  [[maybe_unused]] static constexpr double b       = 0.0;
  std::string                              postfix = concat("_L2<", p, ">");

  serac::profiling::initializeCaliper();

  // Create standard MFEM bilinear and linear forms on H1
  auto                        fec = mfem::L2_FECollection(p, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParBilinearForm A(&fespace);

  // Add the mass term using the standard MFEM method
  mfem::ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new mfem::MassIntegrator(a_coef));

  // Assemble the bilinear form into a matrix
  SERAC_PROFILE_EXPR(concat("mfem_localAssemble", postfix), A.Assemble(0));
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  // Create a linear form for the load term using the standard MFEM method
  mfem::ParLinearForm       f(&fespace);
  mfem::FunctionCoefficient load_func([&](const mfem::Vector& coords) { return 100 * coords(0) * coords(1); });
  // FunctionCoefficient load_func([&]([[maybe_unused]] const Vector& coords) { return 1.0; });

  // Create and assemble the linear load term into a vector
  f.AddDomainIntegrator(new mfem::DomainLFIntegrator(load_func));
  SERAC_PROFILE_EXPR(serac::profiling::concat("mfem_fAssemble", postfix), f.Assemble());
  std::unique_ptr<mfem::HypreParVector> F(
      SERAC_PROFILE_EXPR(concat("mfem_fParallelAssemble", postfix), f.ParallelAssemble()));

  // Set a random state to evaluate the residual
  mfem::ParGridFunction u_global(&fespace);
  u_global.Randomize();

  mfem::Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  // Set up the same problem using weak form

  // Define the types for the test and trial spaces using the function arguments
  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  // Construct the new weak form object using the known test and trial spaces
  Functional<test_space(trial_space)> residual(&fespace, {&fespace});

  // Add the total domain residual term to the weak form
  residual.AddDomainIntegral(
      Dimension<dim>{},
      [&]([[maybe_unused]] auto x, [[maybe_unused]] auto temperature) {
        // get the value and the gradient from the input tuple
        auto [u, du_dx] = temperature;
        auto source     = a * u - (100 * x[0] * x[1]);
        auto flux       = b * du_dx;
        return serac::tuple{source, flux};
      },
      mesh);

  // Compute the residual using standard MFEM methods
  // mfem::Vector r1 = (*J) * U - (*F);
  mfem::Vector r1 = SERAC_PROFILE_EXPR(concat("mfem_Apply", postfix), A * U - (*F));

  // Compute the residual using weak form
  mfem::Vector r2 = SERAC_PROFILE_EXPR(concat("functional_Apply", postfix), residual(U));

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  // Test that the two residuals are equivalent
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  // Compute the gradient using weak form
  mfem::Operator& grad2 = SERAC_PROFILE_EXPR(concat("functional_GetGradient", postfix), residual.GetGradient(U));

  // Compute the gradient action using standard MFEM and Functional
  mfem::Vector g1 = SERAC_PROFILE_EXPR(concat("mfem_ApplyGradient", postfix), (*J) * U);
  mfem::Vector g2 = SERAC_PROFILE_EXPR(concat("functional_ApplyGradient", postfix), grad2 * U);

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  }

  // Ensure the two methods generate the same result
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);

  serac::profiling::terminateCaliper();
}

TEST(L2, 2D_linear) { functional_test(*mesh2D, L2<1>{}, L2<1>{}, Dimension<2>{}); }
TEST(L2, 2D_quadratic) { functional_test(*mesh2D, L2<2>{}, L2<2>{}, Dimension<2>{}); }
TEST(L2, 2D_cubic) { functional_test(*mesh2D, L2<3>{}, L2<3>{}, Dimension<2>{}); }

TEST(L2, 3D_linear) { functional_test(*mesh3D, L2<1>{}, L2<1>{}, Dimension<3>{}); }
TEST(L2, 3D_quadratic) { functional_test(*mesh3D, L2<2>{}, L2<2>{}, Dimension<3>{}); }
TEST(L2, 3D_cubic) { functional_test(*mesh3D, L2<3>{}, L2<3>{}, Dimension<3>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
