#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/physics/utilities/functional/array.hpp"
#include "serac/physics/utilities/functional/dof_numbering.hpp"
#include "serac/numerics/mesh_utils_base.hpp"

#include <gtest/gtest.h>

using namespace serac;

int            num_procs, myid;
int            refinements = 0;
constexpr bool verbose     = true;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

bool operator==(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B)
{
  if (A.Height() != B.Height()) return false;
  if (A.Width() != B.Width()) return false;

  constexpr double tolerance       = 1.0e-15;
  double           fnorm_A_plus_B  = 0.0;
  double           fnorm_A_minus_B = 0.0;

  for (int r = 0; r < A.Height(); r++) {
    auto columns = A.GetRowColumns(r);
    for (int j = 0; j < A.RowSize(r); j++) {
      int c = columns[j];
      if (c < 0) std::cout << "??" << std::endl;
      fnorm_A_plus_B += (A(r, c) + B(r, c)) * (A(r, c) + B(r, c));
      fnorm_A_minus_B += (A(r, c) - B(r, c)) * (A(r, c) - B(r, c));
    }
  }

  return sqrt(fnorm_A_minus_B) < (tolerance * sqrt(fnorm_A_plus_B));
}

bool operator!=(const mfem::SparseMatrix& A, const mfem::SparseMatrix& B) { return !(A == B); }

template <typename space>
auto to_mfem_fecollection(mfem::ParMesh& mesh)
{
  if constexpr (space::family == Family::H1) {
    return mfem::H1_FECollection(space::order, mesh.Dimension());
  }

  if constexpr (space::family == Family::HCURL) {
    return mfem::ND_FECollection(space::order, mesh.Dimension());
  }
}

void apply_permutation(mfem::Array<int>& input, const mfem::Array<int>& permutation)
{
  auto output = input;
  for (int i = 0; i < permutation.Size(); i++) {
    if (permutation[i] >= 0) {
      output[i] = input[permutation[i]];
    } else {
      output[i] = -input[-permutation[i] - 1] - 1;
    }
  }
  input = output;
}

template <typename test, typename trial, int dimension, typename lambda>
void sparsity(mfem::ParMesh& mesh, lambda qf, std::string prefix = "")
{
  auto                        test_fec  = to_mfem_fecollection<test>(mesh);
  auto                        trial_fec = to_mfem_fecollection<trial>(mesh);
  mfem::ParFiniteElementSpace test_fespace(&mesh, &test_fec, test::components);
  mfem::ParFiniteElementSpace trial_fespace(&mesh, &trial_fec, trial::components);

  GradientAssemblyLookupTables gradient_LUT(test_fespace, trial_fespace);

  mfem::Vector            U(trial_fespace.TrueVSize());
  Functional<test(trial)> f(&test_fespace, &trial_fespace);
  f.AddDomainIntegral(Dimension<dimension>{}, qf, mesh);
  f(U);
  mfem::Vector element_matrices = f.ComputeElementGradients();

  auto & element_LUT = gradient_LUT.element_nonzero_LUT;
  auto K_elem = mfem::Reshape(element_matrices.HostReadWrite(), element_LUT.size(1), element_LUT.size(2), element_LUT.size(0));

  std::vector<double> values(gradient_LUT.nnz, 0.0);
  for (int e = 0; e < element_LUT.size(0); e++) {
    for (int i = 0; i < element_LUT.size(1); i++) {
      for (int j = 0; j < element_LUT.size(2); j++) {
        auto [index, sign] = element_LUT(e, i, j);
        values[index] += sign * K_elem(i, j, e);
      }
    }
  }

  int num_rows = test_fespace.GetNDofs() * test_fespace.GetVDim();
  int num_cols = trial_fespace.GetNDofs() * trial_fespace.GetVDim();

  mfem::SparseMatrix B(gradient_LUT.row_ptr.data(), gradient_LUT.col_ind.data(), values.data(), num_rows, num_cols, false, false, true);

  std::ofstream      outfile;
  outfile.open(prefix + "B.mtx");
  B.PrintMM(outfile);
  outfile.close();

  auto& A = f.GetAssembledSparseMatrix();

  A.Finalize();
  A.SortColumnIndices();

  outfile.open(prefix + "A.mtx");
  A.PrintMM(outfile);
  outfile.close();

  if (A != B) {
    std::cout << "test " + prefix + " failed " << std::endl;
    exit(1);
  }

  mfem::SparseMatrix C = grad(f);
  outfile.open(prefix + "C.mtx");
  C.PrintMM(outfile);
  outfile.close();

  if (A != C) {
    std::cout << "test " + prefix + " failed " << std::endl;
    exit(1);
  }
}

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
  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  auto default_qf = [](auto /*x*/, auto field) { return field; };

  sparsity<H1<1>, H1<1>, 2>(*mesh2D, default_qf, "h1h1_2D_");
  sparsity<H1<1>, H1<1>, 3>(*mesh3D, default_qf, "h1h1_3D_");

  //sparsity<Hcurl<1>, Hcurl<1>, 2>(*mesh2D, default_qf, "hcurlhcurl_2D_");
  //sparsity<Hcurl<1>, Hcurl<1>, 3>(*mesh3D, default_qf, "hcurlhcurl_3D_");

  //auto hcurl_h1_3D_qf = [](auto /*x*/, auto field) {
  //  auto [u, grad_u] = field;
  //  return serac::tuple{grad_u, grad_u};
  //};

  //sparsity<Hcurl<1>, H1<1>, 3>(*mesh3D, hcurl_h1_3D_qf, "hcurlh1_3D_");

  //sparsity<H1<1, 2>, H1<1, 2>, 2>(*mesh2D, default_qf, "h1vh1v_2D_");
  //sparsity<H1<1, 3>, H1<1, 3>, 3>(*mesh3D, default_qf, "h1vh1v_3D_");

}
