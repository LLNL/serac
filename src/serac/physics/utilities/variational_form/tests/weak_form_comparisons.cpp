#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace mfem;

int         num_procs, myid;
int         refinements = 0;
const char* mesh_file   = SERAC_REPO_DIR "/data/meshes/star.mesh";

auto setup(int argc, char* argv[])
{
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&refinements, "-r", "--ref", "");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    exit(1);
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  mfem::Mesh mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }

  return mfem::ParMesh(MPI_COMM_WORLD, mesh);
}

template <int p, int dim>
void weak_form_test(mfem::ParMesh& mesh, H1<p> test, H1<p> trial, Dimension<dim>)
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

  ParLinearForm       f(&fespace);
  FunctionCoefficient load_func([&](const Vector& coords) { return 100 * coords(0) * coords(1); });

  f.AddDomainIntegrator(new DomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  ParGridFunction u_global(&fespace);
  u_global.Randomize();

  Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddDomainIntegral(
      Dimension<dim>{},
      [&](auto x, auto temperature) {
        auto [u, du_dx] = temperature;
        auto f0         = a * u - (100 * x[0] * x[1]);
        auto f1         = b * du_dx;
        return std::tuple{f0, f1};
      },
      mesh);

  mfem::Vector r1 = A * U - (*F);
  mfem::Vector r2 = residual(U);

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  mfem::Operator& grad2 = residual.GetGradient(u_global);

  mfem::Vector g1 = (*J) * U;
  mfem::Vector g2 = grad2 * U;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);
}

template <int p, int dim>
void weak_form_test(mfem::ParMesh& mesh, H1<p, dim> test, H1<p, dim> trial, Dimension<dim>)
{
  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  auto                  fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec, dim);

  ParBilinearForm A(&fespace);

  ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new VectorMassIntegrator(a_coef));

  ConstantCoefficient lambda_coef(b);
  ConstantCoefficient mu_coef(b);
  A.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  A.Assemble(0);
  A.Finalize();

  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  ParLinearForm             f(&fespace);
  VectorFunctionCoefficient load_func(dim, [&](const Vector& /*coords*/, Vector& force) {
    force    = 0.0;
    force(0) = -1.0;
  });

  f.AddDomainIntegrator(new VectorDomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  ParGridFunction u_global(&fespace);
  u_global.Randomize();

  Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  [[maybe_unused]] static constexpr auto I = Identity<dim>();

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddDomainIntegral(
      Dimension<dim>{},
      [&](auto /*x*/, auto displacement) {
        auto [u, du_dx] = displacement;
        auto f0         = a * u + I[0];
        auto strain     = 0.5 * (du_dx + transpose(du_dx));
        auto f1         = b * tr(strain) * I + 2.0 * b * strain;
        return std::tuple{f0, f1};
      },
      mesh);

  mfem::Vector r1 = A * U - (*F);
  mfem::Vector r2 = residual(U);

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  mfem::Operator& grad = residual.GetGradient(U);

  mfem::Vector g1 = (*J) * U;
  mfem::Vector g2 = grad * U;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);
}

template <int p, int dim>
void weak_form_test(mfem::ParMesh& mesh, Hcurl<p> test, Hcurl<p> trial, Dimension<dim>)
{
  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  auto                  fec = ND_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);

  ParBilinearForm B(&fespace);

  ConstantCoefficient a_coef(a);
  B.AddDomainIntegrator(new VectorFEMassIntegrator(a_coef));

  ConstantCoefficient b_coef(b);
  B.AddDomainIntegrator(new CurlCurlIntegrator(b_coef));
  B.Assemble(0);
  B.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(B.ParallelAssemble());

  ParLinearForm             f(&fespace);
  VectorFunctionCoefficient load_func(dim, [&](const Vector& coords, Vector& output) {
    double x  = coords(0);
    double y  = coords(1);
    output    = 0.0;
    output(0) = 10 * x * y;
    output(1) = -5 * (x - y) * y;
  });

  f.AddDomainIntegrator(new VectorFEDomainLFIntegrator(load_func));
  f.Assemble();
  std::unique_ptr<mfem::HypreParVector> F(f.ParallelAssemble());

  ParGridFunction u_global(&fespace);
  u_global.Randomize();

  Vector U(fespace.TrueVSize());
  u_global.GetTrueDofs(U);

  using test_space  = decltype(test);
  using trial_space = decltype(trial);

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddDomainIntegral(
      Dimension<dim>{},
      [&](auto x, auto vector_potential) {
        auto [A, curl_A] = vector_potential;
        auto f0          = a * A - tensor<double, dim>{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]};
        auto f1          = b * curl_A;
        return std::tuple{f0, f1};
      },
      mesh);

  mfem::Vector r1 = B * U - f;
  mfem::Vector r2 = residual(U);

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(r1 - r2).Norml2() / r1.Norml2(), 1.e-14);

  mfem::Operator& grad = residual.GetGradient(U);

  mfem::Vector g1 = (*J) * U;
  mfem::Vector g2 = grad * U;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;
  EXPECT_NEAR(0., mfem::Vector(g1 - g2).Norml2() / g1.Norml2(), 1.e-14);
}

template <int dim>
void run_tests(mfem::ParMesh& mesh)
{
  Dimension<dim> d;

  std::cout << "H1/H1 tests" << std::endl;
  weak_form_test(mesh, H1<1>{}, H1<1>{}, d);
  weak_form_test(mesh, H1<2>{}, H1<2>{}, d);
  weak_form_test(mesh, H1<3>{}, H1<3>{}, d);

  std::cout << "H1/H1 tests (elasticity)" << std::endl;
  weak_form_test(mesh, H1<1, dim>{}, H1<1, dim>{}, d);
  weak_form_test(mesh, H1<2, dim>{}, H1<2, dim>{}, d);
  weak_form_test(mesh, H1<3, dim>{}, H1<3, dim>{}, d);

  std::cout << "Hcurl/Hcurl tests" << std::endl;
  weak_form_test(mesh, Hcurl<1>{}, Hcurl<1>{}, d);
  weak_form_test(mesh, Hcurl<2>{}, Hcurl<2>{}, d);
  weak_form_test(mesh, Hcurl<3>{}, Hcurl<3>{}, d);
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  auto mesh = setup(argc, argv);

  if (mesh.Dimension() == 2) {
    run_tests<2>(mesh);
  }
  if (mesh.Dimension() == 3) {
    run_tests<3>(mesh);
  }

  MPI_Finalize();

  return 0;
}
