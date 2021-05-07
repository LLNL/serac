#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/numerics/mesh_utils.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace mfem;
using namespace serac;

int         num_procs, myid;
int         refinements = 0;
constexpr bool verbose     = false;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

template <int p, int dim>
void boundary_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
{
  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  auto                  fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);

  ParBilinearForm A(&fespace);

  ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new MassIntegrator(a_coef));

  ConstantCoefficient b_coef(b);
  A.AddDomainIntegrator(new DiffusionIntegrator(b_coef));
  A.Assemble(0);
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  LinearForm          f(&fespace);
  FunctionCoefficient load_func([&](const Vector& coords) { return 100 * coords(0) * coords(1); });

  f.AddDomainIntegrator(new DomainLFIntegrator(load_func));
  f.Assemble();

  ParGridFunction u_global(&fespace);
  u_global.Randomize();

  Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddSurfaceIntegral([&](auto x, auto /* u */) { return 1.0; }, mesh);

  mfem::Vector r1 = A * u_global - f;
  mfem::Vector r2 = residual * u_global;

  if (verbose) {
    std::cout << "||r1||: " << r1.Norml2() << std::endl;
    std::cout << "||r2||: " << r2.Norml2() << std::endl;
    std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  mfem::Operator& grad2 = residual.GetGradient(u_global);

  mfem::Vector g1 = (*J) * u_global;
  mfem::Vector g2 = grad2 * u_global;

  if (verbose) {
    std::cout << "||g1||: " << g1.Norml2() << std::endl;
    std::cout << "||g2||: " << g2.Norml2() << std::endl;
    std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  }

  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);

}

TEST(boundary, 3D_linear) { boundary_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(boundary, 2D_linear) { boundary_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string mesh_file3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/star.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return 0;
}
