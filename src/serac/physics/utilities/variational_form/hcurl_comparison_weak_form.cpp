#include "mfem.hpp"
#include "parvariationalform.hpp"
#include "qfuncintegrator.hpp"
#include "weak_form.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"

using namespace std;
using namespace mfem;

#include "axom/slic/core/SimpleLogger.hpp"

// solve an equation of the form
// (a * M + b * K) x == f
// 
// where M is the H1 mass matrix
//       K is the H1 stiffness matrix
//       f is some load term
// 
int main(int argc, char* argv[])
{
  #if 1
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  const char * mesh_file = SERAC_REPO_DIR"/data/meshes/star.mesh";

  constexpr int p = 1;
  constexpr int dim = 2;
  int         refinements = 0;
  double a = 1.0;
  double b = 1.0;
  // SERAC EDIT BEGIN
  // double p = 5.0;
  // SERAC EDIT END

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&refinements, "-r", "--ref", "");

  args.Parse();
  if (!args.Good()) {
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0) {
    args.PrintOptions(cout);
  }

  Mesh mesh(mesh_file, 1, 1);
  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);

  auto fec = ND_FECollection(p, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);

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

  using test_space = Hcurl<p>;
  using trial_space = Hcurl<p>;

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

  MPI_Finalize();

  #endif
  return 0;
}
