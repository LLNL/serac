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

  const char* mesh_file = SERAC_REPO_DIR "/data/meshes/star.mesh";

  constexpr int p           = 1;
  constexpr int dim         = 2;
  int           refinements = 0;
  double        a           = 1.0;
  double        b           = 1.0;
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

  if (mesh.Dimension() != dim) {
    std::cout << "invalid mesh dimension" << std::endl;
  }

  for (int l = 0; l < refinements; l++) {
    mesh.UniformRefinement();
  }

  ParMesh pmesh(MPI_COMM_WORLD, mesh);

  auto                  fec = H1_FECollection(p, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec, dim);

  ParBilinearForm A(&fespace);

  ConstantCoefficient a_coef(a);
  A.AddDomainIntegrator(new VectorMassIntegrator(a_coef));

  ConstantCoefficient lambda_coef(b);
  ConstantCoefficient mu_coef(b);
  A.AddDomainIntegrator(new ElasticityIntegrator(lambda_coef, mu_coef));
  A.Assemble(0);
  A.Finalize();

  std::unique_ptr<mfem::HypreParMatrix> J(A.ParallelAssemble());

  LinearForm                f(&fespace);
  VectorFunctionCoefficient load_func(dim, [&](const Vector& /*coords*/, Vector& force) {
    force    = 0.0;
    force(0) = -1.0;
  });

  f.AddDomainIntegrator(new VectorDomainLFIntegrator(load_func));
  f.Assemble();

  ParGridFunction x(&fespace);
  x.Randomize();

  Vector X(fespace.TrueVSize());
  x.GetTrueDofs(X);

  static constexpr auto I = Identity<2>();

  using test_space  = H1<p, dim>;
  using trial_space = H1<p, dim>;

  WeakForm<test_space(trial_space)> residual(&fespace, &fespace);

  residual.AddIntegral(
      [&](auto /*x*/, auto displacement) {
        auto [u, du_dx] = displacement;
        auto f0         = a * u - tensor{{-1.0, 0.0}};
        auto strain     = 0.5 * (du_dx + transpose(du_dx));
        auto f1         = b * tr(strain) * I + 2.0 * b * strain;
        return std::tuple{f0, f1};
      },
      pmesh);

  mfem::Vector r1 = A * x - f;
  mfem::Vector r2 = residual * x;

  std::cout << "||r1||: " << r1.Norml2() << std::endl;
  std::cout << "||r2||: " << r2.Norml2() << std::endl;
  std::cout << "||r1-r2||/||r1||: " << mfem::Vector(r1 - r2).Norml2() / r1.Norml2() << std::endl;

  MPI_Finalize();

  return 0;
}
