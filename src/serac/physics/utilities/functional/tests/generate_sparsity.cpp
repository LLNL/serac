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

struct elem_info{
  int global_row;
  int global_col;
  int local_row;
  int local_col;
  int element_id;
  bool on_boundary;
};

// for sorting lexicographically by {global_row, global_col}
bool operator<(const elem_info & x, const elem_info & y) {
  return (x.global_row < y.global_row) || (x.global_row == y.global_row && x.global_col < y.global_col);
}

bool operator!=(const elem_info & x, const elem_info & y) {
  return (x.global_row != y.global_row) || (x.global_col != y.global_col);
}

auto & operator<<(std::ostream & out, elem_info e) {
  out << e.global_row << ", ";
  out << e.global_col << ", ";
  out << e.local_row << ", ";
  out << e.local_col << ", ";
  out << e.element_id << ", ";
  out << e.on_boundary;
  return out;
}

template < typename T >
struct Array3D{
  Array3D(int n1, int n2, int n3) : strides{n2 * n3, n3, 1}, data(n1 * n2 * n3) {}
  auto & operator()(int i, int j, int k) { return data[i * strides[0] + j * strides[1] + k * strides[2]]; }
  int strides[3];
  std::vector < T > data;
};

template < typename test_collection, typename trial_collection >
void sparsity(mfem::ParMesh & mesh) {

  auto test_fec = test_collection(1, mesh.Dimension());
  auto trial_fec = trial_collection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace test_fespace(&mesh, &test_fec);
  mfem::ParFiniteElementSpace trial_fespace(&mesh, &trial_fec);

  mfem::Array<int> test_dofs;
  mfem::Array<int> trial_dofs;

  test_fespace.GetElementDofs(0, test_dofs);
  trial_fespace.GetElementDofs(0, trial_dofs);
  int num_elements = test_fespace.GetNE();
  int dofs_per_test_element = test_dofs.Size();
  int dofs_per_trial_element = trial_dofs.Size();
  int entries_per_element = dofs_per_test_element * dofs_per_trial_element;

  test_fespace.GetBdrElementDofs(0, test_dofs);
  trial_fespace.GetBdrElementDofs(0, trial_dofs);
  int num_boundary_elements = test_fespace.GetNBE();
  int dofs_per_test_boundary_element = test_dofs.Size();
  int dofs_per_trial_boundary_element = trial_dofs.Size();
  int entries_per_boundary_element = dofs_per_test_boundary_element * dofs_per_trial_boundary_element;

  int num_infos[2] = {
    entries_per_element * num_elements,
    entries_per_boundary_element * num_boundary_elements
  };

  std::vector < elem_info > infos (num_infos[0] + num_infos[1]);

  int info_id = 0;
  for (int e = 0; e < num_elements; e++) {
    test_fespace.GetElementDofs(e, test_dofs);
    trial_fespace.GetElementDofs(e, trial_dofs);
    for (int i = 0; i < dofs_per_test_element; i++) {
      for (int j = 0; j < dofs_per_trial_element; j++) {
        infos[info_id++] = elem_info{test_dofs[i], trial_dofs[j], i, j, e, false};
      }
    }
  }

  for (int b = 0; b < num_boundary_elements; b++) {
    test_fespace.GetBdrElementDofs(b, test_dofs);
    trial_fespace.GetBdrElementDofs(b, trial_dofs);
    for (int i = 0; i < dofs_per_test_boundary_element; i++) {
      for (int j = 0; j < dofs_per_test_boundary_element; j++) {
        infos[info_id++] = elem_info{test_dofs[i], trial_dofs[j], i, j, b, true};
      }
    }
  }

  std::sort(infos.begin(), infos.end());

  std::vector < int > row_ptr(test_fespace.GetNDofs() + 1);
  std::vector < int > col_ind(infos.size(), 0);
  std::vector < int > nonzero_ids(infos.size(), 0);

  row_ptr[0] = 0;
  col_ind[0] = infos[0].global_col;
  nonzero_ids[0] = 0;

  for (size_t i = 1; i < infos.size(); i++) {
    col_ind[i] = infos[i].global_col;
    nonzero_ids[i] = nonzero_ids[i-1] + (infos[i-1] != infos[i]);

    for (int j = infos[i-1].global_row; j < infos[i].global_row; j++) {
      row_ptr[j+1] = nonzero_ids[i];
    }
  }

  row_ptr.back() = nonzero_ids.back();

  for (size_t i = 0; i < infos.size(); i++) {
    std::cout << infos[i] << "     " << col_ind[i] << " " << nonzero_ids[i] << std::endl;
  }

  for (size_t i = 0; i < row_ptr.size(); i++) {
    std::cout << row_ptr[i] << std::endl;
  }

  Array3D<int> element_nonzero_LUT(num_elements, dofs_per_test_element, dofs_per_trial_element);
  Array3D<int> boundary_element_nonzero_LUT(num_boundary_elements, dofs_per_test_boundary_element, dofs_per_trial_boundary_element);

  for (size_t i = 0; i < infos.size(); i++) {
    if (infos[i].on_boundary) {
      boundary_element_nonzero_LUT(infos[i].element_id, infos[i].local_row, infos[i].local_col) = nonzero_ids[i];
    } else {
      element_nonzero_LUT(infos[i].element_id, infos[i].local_row, infos[i].local_col) = nonzero_ids[i];
    }
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


  sparsity< mfem::H1_FECollection, mfem::H1_FECollection >(*mesh2D);

}
