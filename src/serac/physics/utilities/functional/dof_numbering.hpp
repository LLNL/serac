#pragma once

#include "mfem.hpp"

#include "serac/physics/utilities/functional/array.hpp"

namespace serac {

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

struct DofNumbering {
  DofNumbering(const mfem::ParFiniteElementSpace& fespace)
      : element_dofs(fespace.GetNE(), fespace.GetFE(0)->GetDof() * fespace.GetVDim()),
        boundary_element_dofs(fespace.GetNFbyType(mfem::FaceType::Boundary),
                              fespace.GetBE(0)->GetDof() * fespace.GetVDim())
  {
    {
      auto elem_restriction = fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);

      mfem::Vector iota(elem_restriction->Width());
      mfem::Vector dof_ids(elem_restriction->Height());
      dof_ids = 0.0;
      for (int i = 0; i < iota.Size(); i++) {
        iota[i] = i;
      }

      elem_restriction->Mult(iota, dof_ids);
      const double * dof_ids_h = dof_ids.HostRead();

      for (size_t e = 0; e < element_dofs.size(0); e++) {
        for (size_t i = 0; i < element_dofs.size(1); i++) {
          int mfem_id        = static_cast<int>(dof_ids_h[element_dofs.index(e, i)]);
          element_dofs(e, i) = signed_index{mfem_index(mfem_id), mfem_sign(mfem_id)};
        }
      }
    }

    {
      auto face_restriction = fespace.GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                                         mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);

      mfem::Vector iota(face_restriction->Width());
      mfem::Vector dof_ids(face_restriction->Height());
      for (int i = 0; i < iota.Size(); i++) {
        iota[i] = i;
      }

      face_restriction->Mult(iota, dof_ids);
      const double * dof_ids_h = dof_ids.HostRead();

      for (size_t e = 0; e < boundary_element_dofs.size(0); e++) {
        for (size_t i = 0; i < boundary_element_dofs.size(1); i++) {
          int mfem_id                 = static_cast<int>(dof_ids_h[boundary_element_dofs.index(e, i)]);
          boundary_element_dofs(e, i) = signed_index{mfem_index(mfem_id), mfem_sign(mfem_id)};
        }
      }
    }
  }

  serac::CPUArray<signed_index, 2> element_dofs;
  serac::CPUArray<signed_index, 2> boundary_element_dofs;
};

struct GradientAssemblyLookupTables {
  GradientAssemblyLookupTables(mfem::ParFiniteElementSpace& test_fespace, mfem::ParFiniteElementSpace& trial_fespace)
      : element_nonzero_LUT(trial_fespace.GetNE(), test_fespace.GetFE(0)->GetDof() * test_fespace.GetVDim(),
                            trial_fespace.GetFE(0)->GetDof() * trial_fespace.GetVDim()),
        boundary_element_nonzero_LUT(trial_fespace.GetNFbyType(mfem::FaceType::Boundary),
                                     test_fespace.GetBE(0)->GetDof() * test_fespace.GetVDim(),
                                     trial_fespace.GetBE(0)->GetDof() * trial_fespace.GetVDim())
  {
    DofNumbering test_dofs_(test_fespace);
    DofNumbering trial_dofs_(trial_fespace);

    int num_elements          = trial_fespace.GetNE();
    int num_boundary_elements = trial_fespace.GetNFbyType(mfem::FaceType::Boundary);

    std::vector<elem_info> infos;

    bool on_boundary = false;
    for (int e = 0; e < num_elements; e++) {
      for (int i = 0; i < int(test_dofs_.element_dofs.size(1)); i++) {
        auto test_dof = test_dofs_.element_dofs(e, i);
        for (int j = 0; j < int(trial_dofs_.element_dofs.size(1)); j++) {
          auto trial_dof = trial_dofs_.element_dofs(e, j);
          infos.push_back(
              elem_info{test_dof, trial_dof, i, j, e, mfem_sign(test_dof) * mfem_sign(trial_dof), on_boundary});
        }
      }
    }

    // note: mfem doesn't implement FaceRestrictions for some of its function spaces,
    // so until those are implemented, DofNumbering::boundary_element_dofs will be
    // an empty 2D array, so these loops will not do anything
    on_boundary = true;
    for (int e = 0; e < num_boundary_elements; e++) {
      for (int i = 0; i < int(test_dofs_.boundary_element_dofs.size(1)); i++) {
        auto test_dof = test_dofs_.boundary_element_dofs(e, i);
        for (int j = 0; j < int(trial_dofs_.boundary_element_dofs.size(1)); j++) {
          auto trial_dof = trial_dofs_.boundary_element_dofs(e, j);
          infos.push_back(
              elem_info{test_dof, trial_dof, i, j, e, mfem_sign(test_dof) * mfem_sign(trial_dof), on_boundary});
        }
      }
    }

    std::sort(infos.begin(), infos.end());

    row_ptr.resize(test_fespace.GetNDofs() * test_fespace.GetVDim() + 1);

    std::vector<signed_index> nonzero_ids(infos.size());

    nnz        = 0;
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

    for (size_t i = 0; i < infos.size(); i++) {
      auto [_1, _2, local_row, local_col, element_id, _3, is_on_boundary] = infos[i];
      if (is_on_boundary) {
        boundary_element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
      } else {
        element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
      }
    }
  }

  int              nnz;
  std::vector<int> row_ptr;
  std::vector<int> col_ind;

  serac::CPUArray<signed_index, 3> element_nonzero_LUT;
  serac::CPUArray<signed_index, 3> boundary_element_nonzero_LUT;
};

}  // namespace serac
