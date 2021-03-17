#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"

#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"

using namespace std;
using namespace mfem;

static constexpr int dim = 2;

int num_procs, myid;
int refinements = 0;
const char * mesh_file = SERAC_REPO_DIR"/data/meshes/star.mesh";

auto parse_arguments(int argc, char* argv[]) {

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

}


// for now, each test recreates the mesh from scratch 
// to work around a bug in mfem::Mesh::GetGeometricFactors()
//
// see https://github.com/mfem/mfem/issues/2106
auto create_mesh() {
  mfem::Mesh mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }

  return mfem::ParMesh(MPI_COMM_WORLD, mesh);
}

template < int p >
void weak_form_test(H1<p> test, H1<p> trial) {

  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  mfem::ParMesh mesh = create_mesh();

  auto fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);

  ParBilinearForm A(&fespace);

  ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new MassIntegrator(a_coef));

  ConstantCoefficient b_coef(b);
  A.AddDomainIntegrator(new DiffusionIntegrator(b_coef));
  A.Assemble(0);
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  LinearForm f(&fespace);
  FunctionCoefficient load_func([&](const Vector& coords) {
    return 100 * coords(0) * coords(1);
  });

  f.AddDomainIntegrator(new DomainLFIntegrator(load_func));
  f.Assemble();

  ParGridFunction x(&fespace);
  x.Randomize();

  Vector X(fespace.TrueVSize());
  x.GetTrueDofs(X);

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  WeakForm< test_space(trial_space) > residual(&fespace, &fespace);

  residual.AddVolumeIntegral([&](auto x, auto temperature) {
    auto [u, du_dx] = temperature;
    auto f0 = a * u - (100 * x[0] * x[1]);
    auto f1 = b * du_dx;
    return std::tuple{f0, f1};
  }, mesh);

  mfem::Vector r1 = A * x - f;
  mfem::Vector r2 = residual * x;

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;

  mfem::Operator & grad2 = residual.GetGradient(x);

  mfem::Vector g1 = (*J) * x;
  mfem::Vector g2 = grad2 * x;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;

}

template < int p >
void weak_form_test(H1<p, 2> test, H1<p, 2> trial) {

  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  mfem::ParMesh mesh = create_mesh();

  auto fec = H1_FECollection(p, dim);
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

  LinearForm f(&fespace);
  VectorFunctionCoefficient load_func(dim, [&](const Vector& /*coords*/, Vector & force) {
    force = 0.0;
    force(0) = -1.0;
  });

  f.AddDomainIntegrator(new VectorDomainLFIntegrator(load_func));
  f.Assemble();

  ParGridFunction x(&fespace);
  x.Randomize();

  Vector X(fespace.TrueVSize());
  x.GetTrueDofs(X);

  static constexpr auto I = Identity<dim>();

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  WeakForm< test_space(trial_space) > residual(&fespace, &fespace);

  residual.AddVolumeIntegral([&](auto /*x*/, auto displacement) {
    auto [u, du_dx] = displacement;
    auto f0 = a * u - tensor{{-1.0, 0.0}};
    auto strain = 0.5 * (du_dx + transpose(du_dx));
    auto f1 = b * tr(strain) * I + 2.0 * b * strain;
    return std::tuple{f0, f1};
  }, mesh);

  mfem::Vector r1 = A * x - f;
  mfem::Vector r2 = residual * x;

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;

  mfem::Operator & grad = residual.GetGradient(x);

  mfem::Vector g1 = (*J) * x;
  mfem::Vector g2 = grad * x;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;

}

template < int p >
void weak_form_test(Hcurl<p> test, Hcurl<p> trial) {

  static constexpr double a = 1.7;
  static constexpr double b = 2.1;

  mfem::ParMesh mesh = create_mesh();

  auto fec = ND_FECollection(p, dim);
  ParFiniteElementSpace fespace(&mesh, &fec);

  ParBilinearForm A(&fespace);

  ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new VectorFEMassIntegrator(a_coef));

  ConstantCoefficient b_coef(b);
  A.AddDomainIntegrator(new CurlCurlIntegrator(b_coef));
  A.Assemble(0);
  A.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  LinearForm f(&fespace);
  VectorFunctionCoefficient load_func(dim, [&](const Vector& coords, Vector& output) {
    double x = coords(0);
    double y = coords(1);
    output = 0.0;
    output(0) = 10 * x * y;
    output(1) = -5 * (x - y) * y;
  });

  f.AddDomainIntegrator(new VectorFEDomainLFIntegrator(load_func));
  f.Assemble();

  ParGridFunction x(&fespace);
  x.Randomize();

  Vector X(fespace.TrueVSize());
  x.GetTrueDofs(X);

  using test_space = decltype(test);
  using trial_space = decltype(trial);

  WeakForm< test_space(trial_space) > residual(&fespace, &fespace);

  residual.AddVolumeIntegral([&](auto x, auto vector_potential) {
    auto [A, curl_A] = vector_potential;
    auto f0 = a * A - tensor{{10 * x[0] * x[1], -5 * (x[0] - x[1]) * x[1]}};
    auto f1 = b * curl_A;
    return std::tuple{f0, f1};
  }, mesh);

  mfem::Vector r1 = A * x - f;
  mfem::Vector r2 = residual * x;

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;

  mfem::Operator & grad = residual.GetGradient(x);

  mfem::Vector g1 = (*J) * x;
  mfem::Vector g2 = grad * x;

  std::cout << "||g1||: " << g1.Norml2() << std::endl;
  std::cout << "||g2||: " << g2.Norml2() << std::endl;
  std::cout << "||g1-g2||/||g1||: " << mfem::Vector(g1 - g2).Norml2() / g1.Norml2() << std::endl;

}

int main(int argc, char* argv[])
{

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  parse_arguments(argc, argv);

  std::cout << "H1/H1 tests" << std::endl;
  weak_form_test(H1<1>{}, H1<1>{});
  weak_form_test(H1<2>{}, H1<2>{});
  weak_form_test(H1<3>{}, H1<3>{});

  std::cout << "H1/H1 tests (elasticity)" << std::endl;
  weak_form_test(H1<1, dim>{}, H1<1, dim>{});
  weak_form_test(H1<2, dim>{}, H1<2, dim>{});
  weak_form_test(H1<3, dim>{}, H1<3, dim>{});

  std::cout << "Hcurl/Hcurl tests" << std::endl;
  weak_form_test(Hcurl<1>{}, Hcurl<1>{});
  weak_form_test(Hcurl<2>{}, Hcurl<2>{});
  weak_form_test(Hcurl<3>{}, Hcurl<3>{});


  MPI_Finalize();

  return 0;
}
