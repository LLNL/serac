#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"

#include "serac/physics/utilities/variational_form/weak_form.hpp"
#include "serac/physics/utilities/variational_form/tensor.hpp"

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
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  const char * mesh_file = SERAC_REPO_DIR"/data/meshes/star.mesh";

  constexpr int p = 2;
  constexpr int dim = 2;
  int         refinements = 0;
  double a = 1.0;
  double b = 1.0;

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

  auto fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);

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

  using test_space = H1<p>;
  using trial_space = H1<p>;

  WeakForm< test_space(trial_space) > residual(&fespace, &fespace);

  residual.AddIntegral([&](auto x, auto temperature) {
    auto [u, du_dx] = temperature;
    auto f0 = a * u - (100 * x[0] * x[1]);
    auto f1 = b * du_dx;
    return std::tuple{f0, f1};
  }, pmesh);

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

  MPI_Finalize();

  return 0;
}
