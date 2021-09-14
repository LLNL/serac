#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "axom/slic/core/SimpleLogger.hpp"

#include "serac/serac_config.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/numerics/mesh_utils_base.hpp"

#include <gtest/gtest.h>

using namespace serac;

int            num_procs, myid;
int            refinements = 0;
constexpr bool verbose     = true;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

struct elem_info {
  int  global_row;
  int  global_col;
  int  local_row;
  int  local_col;
  int  element_id;
  int  sign;
  bool on_boundary;
};

// for sorting lexicographically by {global_row, global_col}
bool operator<(const elem_info& x, const elem_info& y)
{
  return (x.global_row < y.global_row) || (x.global_row == y.global_row && x.global_col < y.global_col);
}

bool operator!=(const elem_info& x, const elem_info& y)
{
  return (x.global_row != y.global_row) || (x.global_col != y.global_col);
}

auto& operator<<(std::ostream& out, elem_info e)
{
  out << e.global_row << ", ";
  out << e.global_col << ", ";
  out << e.local_row << ", ";
  out << e.local_col << ", ";
  out << e.element_id << ", ";
  out << e.on_boundary;
  return out;
}

int mfem_sign(int i) { return (i >= 0) ? 1 : -1; }
int mfem_index(int i) { return (i >= 0) ? i : -1 - i; }

struct signed_index {
  int index;
  int sign;
      operator int() { return index; }
};

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

template <typename T>
struct Array3D {
  Array3D(int n1, int n2, int n3) : strides{n2 * n3, n3, 1}, data(n1 * n2 * n3) {}
  auto&          operator()(int i, int j, int k) { return data[i * strides[0] + j * strides[1] + k * strides[2]]; }
  int            strides[3];
  std::vector<T> data;
};

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

  // auto test_elem_restriction  = test_fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC));
  // auto test_face_restriction  = test_fespace.GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
  // mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);

  // auto trial_elem_restriction  = trial_fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC));
  // auto trial_face_restriction  = trial_fespace.GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
  // mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);

  mfem::Array<int> test_dofs;
  mfem::Array<int> trial_dofs;

  int test_vdim  = test_fespace.GetVDim();
  int trial_vdim = trial_fespace.GetVDim();

  test_fespace.GetElementDofs(0, test_dofs);
  trial_fespace.GetElementDofs(0, trial_dofs);
  int num_elements           = test_fespace.GetNE();
  int dofs_per_test_element  = test_dofs.Size();
  int dofs_per_trial_element = trial_dofs.Size();
  int entries_per_element    = dofs_per_test_element * dofs_per_trial_element;

  test_fespace.GetBdrElementDofs(0, test_dofs);
  trial_fespace.GetBdrElementDofs(0, trial_dofs);
  int num_boundary_elements           = test_fespace.GetNBE();
  int dofs_per_test_boundary_element  = test_dofs.Size();
  int dofs_per_trial_boundary_element = trial_dofs.Size();
  int entries_per_boundary_element =
      dofs_per_test_boundary_element * test_vdim * dofs_per_trial_boundary_element * trial_vdim;

  int num_infos[2] = {entries_per_element * num_elements, entries_per_boundary_element * num_boundary_elements};

  std::vector<elem_info> infos;
  infos.reserve(num_infos[0] + num_infos[1]);

  {
    bool                    on_boundary = false;
    const mfem::Array<int>& test_native_to_lexicographic =
        dynamic_cast<const mfem::TensorBasisElement*>(test_fespace.GetFE(0))->GetDofMap();
    const mfem::Array<int>& trial_native_to_lexicographic =
        dynamic_cast<const mfem::TensorBasisElement*>(trial_fespace.GetFE(0))->GetDofMap();

    for (int e = 0; e < num_elements; e++) {
      test_fespace.GetElementDofs(e, test_dofs);
      trial_fespace.GetElementDofs(e, trial_dofs);
      apply_permutation(test_dofs, test_native_to_lexicographic);
      apply_permutation(trial_dofs, trial_native_to_lexicographic);
      for (int i = 0; i < dofs_per_test_element; i++) {
        for (int j = 0; j < dofs_per_trial_element; j++) {
          for (int k = 0; k < test_vdim; k++) {
            int test_vdof = test_fespace.DofToVDof(mfem_index(test_dofs[i]), k);
            for (int l = 0; l < trial_vdim; l++) {
              int trial_vdof = trial_fespace.DofToVDof(mfem_index(trial_dofs[j]), l);
              infos.push_back(elem_info{test_vdof, trial_vdof, i + dofs_per_test_element * k,
                                        j + dofs_per_trial_element * l, e,
                                        mfem_sign(test_dofs[i]) * mfem_sign(trial_dofs[j]), on_boundary});
            }
          }
        }
      }
    }
  }

  // mfem doesn't implement GetDofMap for some of its Nedelec elements (??),
  // so we have to temporarily disable boundary terms for Hcurl until they do
  if (test::family != Family::HCURL && trial::family != Family::HCURL && mesh.Dimension() == 3) {
    bool on_boundary = true;

    const mfem::Array<int>& test_native_to_lexicographic =
        dynamic_cast<const mfem::TensorBasisElement*>(test_fespace.GetBE(0))->GetDofMap();
    const mfem::Array<int>& trial_native_to_lexicographic =
        dynamic_cast<const mfem::TensorBasisElement*>(trial_fespace.GetBE(0))->GetDofMap();

    for (int b = 0; b < num_boundary_elements; b++) {
      test_fespace.GetBdrElementDofs(b, test_dofs);
      trial_fespace.GetBdrElementDofs(b, trial_dofs);
      apply_permutation(test_dofs, test_native_to_lexicographic);
      apply_permutation(trial_dofs, trial_native_to_lexicographic);
      for (int i = 0; i < dofs_per_test_boundary_element; i++) {
        for (int j = 0; j < dofs_per_test_boundary_element; j++) {
          for (int k = 0; k < test_vdim; k++) {
            int test_vdof = test_fespace.DofToVDof(mfem_index(test_dofs[i]), k);
            for (int l = 0; l < trial_vdim; l++) {
              int trial_vdof = trial_fespace.DofToVDof(mfem_index(trial_dofs[j]), l);
              infos.push_back(elem_info{test_vdof, trial_vdof, i + dofs_per_test_boundary_element * k,
                                        j + dofs_per_trial_boundary_element * l, b,
                                        mfem_sign(test_dofs[i]) * mfem_sign(trial_dofs[j]), on_boundary});
            }
          }
        }
      }
    }
  }

  std::sort(infos.begin(), infos.end());

  std::vector<int>          row_ptr(test_fespace.GetNDofs() * test_fespace.GetVDim() + 1);
  std::vector<int>          col_ind;
  std::vector<signed_index> nonzero_ids(infos.size());

  int nnz    = 0;
  row_ptr[0] = 0;
  col_ind.push_back(infos[0].global_col);
  nonzero_ids[0] = {0, infos[0].sign};

  for (size_t i = 1; i < infos.size(); i++) {
    // increment the nonzero count every time we find a new (i,j) pair
    nnz += (infos[i - 1] != infos[i]);

    nonzero_ids[i] = signed_index{nnz, infos[i].sign};

    if (infos[i - 1] != infos[i]) {
      col_ind.push_back(infos[i].global_col);
    }

    for (int j = infos[i - 1].global_row; j < infos[i].global_row; j++) {
      row_ptr[j + 1] = nonzero_ids[i];
    }
  }

  row_ptr.back() = ++nnz;

  Array3D<signed_index> element_nonzero_LUT(num_elements, dofs_per_test_element * test_vdim,
                                            dofs_per_trial_element * trial_vdim);
  Array3D<signed_index> boundary_element_nonzero_LUT(num_boundary_elements, dofs_per_test_boundary_element * test_vdim,
                                                     dofs_per_trial_boundary_element * trial_vdim);

  for (size_t i = 0; i < infos.size(); i++) {
    auto [_1, _2, local_row, local_col, element_id, _3, on_boundary] = infos[i];
    if (on_boundary) {
      boundary_element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
    } else {
      element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
    }
  }

  mfem::Vector            U(trial_fespace.TrueVSize());
  Functional<test(trial)> f(&test_fespace, &trial_fespace);
  f.AddDomainIntegral(Dimension<dimension>{}, qf, mesh);
  f(U);
  mfem::Vector element_matrices = f.ComputeElementMatrices();
  auto         K_elem           = mfem::Reshape(element_matrices.HostReadWrite(), dofs_per_test_element * test_vdim,
                              dofs_per_trial_element * trial_vdim, num_elements);

  std::vector<double> values(nnz, 0.0);
  int                 num_rows = test_fespace.GetNDofs() * test_fespace.GetVDim();
  int                 num_cols = trial_fespace.GetNDofs() * trial_fespace.GetVDim();
  for (int e = 0; e < num_elements; e++) {
    for (int i = 0; i < dofs_per_test_element * test_vdim; i++) {
      for (int j = 0; j < dofs_per_trial_element * trial_vdim; j++) {
        auto [index, sign] = element_nonzero_LUT(e, i, j);
        values[index] += sign * K_elem(i, j, e);
      }
    }
  }

  mfem::SparseMatrix B(row_ptr.data(), col_ind.data(), values.data(), num_rows, num_cols, false, false, true);
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

  sparsity<Hcurl<1>, Hcurl<1>, 2>(*mesh2D, default_qf, "hcurlhcurl_2D_");
  sparsity<Hcurl<1>, Hcurl<1>, 3>(*mesh3D, default_qf, "hcurlhcurl_3D_");

  auto hcurl_h1_3D_qf = [](auto /*x*/, auto field) {
    auto [u, grad_u] = field;
    return serac::tuple{grad_u, grad_u};
  };

  sparsity<Hcurl<1>, H1<1>, 3>(*mesh3D, hcurl_h1_3D_qf, "hcurlh1_3D_");

  sparsity<H1<1, 2>, H1<1, 2>, 2>(*mesh2D, default_qf, "h1vh1v_2D_");
  sparsity<H1<1, 3>, H1<1, 3>, 3>(*mesh3D, default_qf, "h1vh1v_3D_");
}
