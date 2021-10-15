#pragma once

#include "mfem.hpp"

#include "serac/physics/utilities/functional/array.hpp"

namespace serac {

/**
 * @brief a (poorly named) tuple of quantities used to discover the sparsity
 * pattern associated with element and boundary element matrices.
 *
 * It stores information about how the entries of an element "stiffness" matrix
 * map to the global stiffness. The operator< definition allows us to sort them
 * lexicographically, to facilitate creating the CSR matrix graph.
 */
struct ElemInfo {
  int  global_row_;   ///< The global row number
  int  global_col_;   ///< The global column number
  int  local_row_;    ///< The local row number
  int  local_col_;    ///< The global column number
  int  element_id_;   ///< The element ID
  int  sign_;         ///< The orientation of the element
  bool on_boundary_;  ///< True if element is on the boundary
};

/**
 * @brief operator for sorting lexicographically by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
bool operator<(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row_ < y.global_row_) || (x.global_row_ == y.global_row_ && x.global_col_ < y.global_col_);
}

/**
 * @brief operator determining inequality by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
bool operator!=(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row_ != y.global_row_) || (x.global_col_ != y.global_col_);
}

auto& operator<<(std::ostream& out, const ElemInfo& e)
{
  out << e.global_row_ << ", ";
  out << e.global_col_ << ", ";
  out << e.local_row_ << ", ";
  out << e.local_col_ << ", ";
  out << e.element_id_ << ", ";
  out << e.on_boundary_;
  return out;
}

/**
 * @brief mfem will frequently encode {sign, index} into a single int32_t.
 * This function decodes the sign from such a type.
 */
int mfem_sign(int i) { return (i >= 0) ? 1 : -1; }

/**
 * @brief mfem will frequently encode {sign, index} into a single int32_t.
 * This function decodes the index from such a type.
 */
int mfem_index(int i) { return (i >= 0) ? i : -1 - i; }

/**
 * @brief this type explicitly stores sign (typically used conveying edge/face orientation) and index values
 *
 * TODO: investigate implementation via bitfield (should have smaller memory footprint, better readability than mfem's
 * {sign, index} int32_t encoding)
 */
struct SignedIndex {
  /// the actual index of some quantity
  int index_;

  /// whether or not the value associated with this index is positive or negative
  int sign_;

  /// the implicit conversion to int extracts only the index
  operator int() { return index_; }
};

/**
 * @brief this object extracts the dofs for each element in a FiniteElementSpace as a 2D array such that
 *   element_dofs_(e, i) will be the `i`th dof of element `e`.
 *
 * Note: due to an internal inconsistency between mfem::FiniteElementSpace and mfem::FaceRestriction,
 *    we choose to use the Restriction operator as the "source of truth", since we are also using its
 *    convention for quadrature point numbering.
 */
struct DofNumbering {
  DofNumbering(const mfem::ParFiniteElementSpace& fespace)
      : element_dofs_(fespace.GetNE(), fespace.GetFE(0)->GetDof() * fespace.GetVDim()),
        bdr_element_dofs_(fespace.GetNFbyType(mfem::FaceType::Boundary), fespace.GetBE(0)->GetDof() * fespace.GetVDim())
  {
    {
      auto elem_restriction = fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);

      mfem::Vector iota(elem_restriction->Width());
      mfem::Vector dof_ids(elem_restriction->Height());
      dof_ids = 0.0;
      for (int i = 0; i < iota.Size(); i++) {
        iota[i] = i;
      }

      // we're using Mult() to reveal the locations nonzero entries
      // in the restriction operator, since that information is not
      // made available through its public interface
      //
      // TODO: investigate refactoring mfem's restriction operators
      // to provide this information in more natural way.
      elem_restriction->Mult(iota, dof_ids);
      const double* dof_ids_h = dof_ids.HostRead();

      for (size_t e = 0; e < element_dofs_.size(0); e++) {
        for (size_t i = 0; i < element_dofs_.size(1); i++) {
          int mfem_id         = static_cast<int>(dof_ids_h[element_dofs_.index(e, i)]);
          element_dofs_(e, i) = SignedIndex{mfem_index(mfem_id), mfem_sign(mfem_id)};
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
      const double* dof_ids_h = dof_ids.HostRead();

      for (size_t e = 0; e < bdr_element_dofs_.size(0); e++) {
        for (size_t i = 0; i < bdr_element_dofs_.size(1); i++) {
          int mfem_id             = static_cast<int>(dof_ids_h[bdr_element_dofs_.index(e, i)]);
          bdr_element_dofs_(e, i) = SignedIndex{mfem_index(mfem_id), mfem_sign(mfem_id)};
        }
      }
    }
  }

  /// @brief element_dofs_(e, i) stores the `i`th dof of element `e`.
  serac::CPUArray<SignedIndex, 2> element_dofs_;

  /// @brief bdr_element_dofs_(b, i) stores the `i`th dof of boundary element `b`.
  serac::CPUArray<SignedIndex, 2> bdr_element_dofs_;
};

/**
 * @brief this object figures out the sparsity pattern associated with a finite element discretization
 *   of the given test and trial function spaces, and records which nonzero each element "stiffness"
 *   matrix maps to, to facilitate assembling the element matrices into the global sparse matrix. e.g.
 *
 *   element_nonzero_LUT(e, i, j) says where (in the global sparse matrix)
 *   to put the (i,j) component of the matrix associated with element element matrix `e`
 *
 * Note: due to an internal inconsistency between mfem::FiniteElementSpace and mfem::FaceRestriction,
 *    we choose to use the Restriction operator as the "source of truth", since we are also using its
 *    convention for quadrature point numbering.
 */
struct GradientAssemblyLookupTables {
  GradientAssemblyLookupTables(mfem::ParFiniteElementSpace& test_fespace, mfem::ParFiniteElementSpace& trial_fespace)
      : element_nonzero_LUT(trial_fespace.GetNE(), test_fespace.GetFE(0)->GetDof() * test_fespace.GetVDim(),
                            trial_fespace.GetFE(0)->GetDof() * trial_fespace.GetVDim()),
        bdr_element_nonzero_LUT(trial_fespace.GetNFbyType(mfem::FaceType::Boundary),
                                test_fespace.GetBE(0)->GetDof() * test_fespace.GetVDim(),
                                trial_fespace.GetBE(0)->GetDof() * trial_fespace.GetVDim())
  {
    DofNumbering test_dofs(test_fespace);
    DofNumbering trial_dofs(trial_fespace);

    int num_elements     = trial_fespace.GetNE();
    int num_bdr_elements = trial_fespace.GetNFbyType(mfem::FaceType::Boundary);

    std::vector<ElemInfo> infos;

    // we start by having each element and boundary element emit the (i,j) entry that it
    // touches in the global "stiffness matrix", and also keep track of some metadata about
    // which element and which dof are associated with that particular nonzero entry
    bool on_boundary = false;
    for (int e = 0; e < num_elements; e++) {
      for (int i = 0; i < int(test_dofs.element_dofs_.size(1)); i++) {
        auto test_dof = test_dofs.element_dofs_(e, i);
        for (int j = 0; j < int(trial_dofs.element_dofs_.size(1)); j++) {
          auto trial_dof = trial_dofs.element_dofs_(e, j);
          infos.push_back(
              ElemInfo{test_dof, trial_dof, i, j, e, mfem_sign(test_dof) * mfem_sign(trial_dof), on_boundary});
        }
      }
    }

    // note: mfem doesn't implement FaceRestrictions for some of its function spaces,
    // so until those are implemented, DofNumbering::bdr_element_dofs will be
    // an empty 2D array, so these loops will not do anything
    on_boundary = true;
    for (int e = 0; e < num_bdr_elements; e++) {
      for (int i = 0; i < int(test_dofs.bdr_element_dofs_.size(1)); i++) {
        auto test_dof = test_dofs.bdr_element_dofs_(e, i);
        for (int j = 0; j < int(trial_dofs.bdr_element_dofs_.size(1)); j++) {
          auto trial_dof = trial_dofs.bdr_element_dofs_(e, j);
          infos.push_back(
              ElemInfo{test_dof, trial_dof, i, j, e, mfem_sign(test_dof) * mfem_sign(trial_dof), on_boundary});
        }
      }
    }

    // sorting the ElemInfos by row and column groups the different contributions
    // to the same location of the global stiffness matrix, and makes it easy to identify
    // the unique entries
    std::sort(infos.begin(), infos.end());

    // the row_ptr array size only depends on the number of rows in the global stiffness matrix,
    // so we already know its size before processing the ElemInfo array
    row_ptr.resize(test_fespace.GetNDofs() * test_fespace.GetVDim() + 1);

    // the other CSR matrix arrays are formed incrementally by going through the sorted ElemInfo values
    std::vector<SignedIndex> nonzero_ids(infos.size());

    nnz        = 0;
    row_ptr[0] = 0;
    col_ind.push_back(infos[0].global_col_);
    nonzero_ids[0] = {0, infos[0].sign_};

    for (size_t i = 1; i < infos.size(); i++) {
      // increment the nonzero count every time we find a new (i,j) entry
      nnz += (infos[i - 1] != infos[i]);

      // record the index, sign, and column of this particular (i,j) entry
      nonzero_ids[i] = SignedIndex{nnz, infos[i].sign_};
      if (infos[i - 1] != infos[i]) {
        col_ind.push_back(infos[i].global_col_);
      }

      // if the new entry has a different row, then the row_ptr offsets must be set as well
      for (int j = infos[i - 1].global_row_; j < infos[i].global_row_; j++) {
        row_ptr[j + 1] = nonzero_ids[i];
      }
    }

    row_ptr.back() = ++nnz;

    // once we've finished processing the ElemInfo array, we go back and fill in our lookup tables
    // so that each element can know precisely where to place its element stiffness matrix contributions
    for (size_t i = 0; i < infos.size(); i++) {
      auto [_1, _2, local_row, local_col, element_id, _3, from_bdr_element] = infos[i];
      if (from_bdr_element) {
        bdr_element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
      } else {
        element_nonzero_LUT(element_id, local_row, local_col) = nonzero_ids[i];
      }
    }
  }

  /// @brief how many nonzero entries appear in the sparse matrix
  int nnz;

  /**
   * @brief array holding the offsets for a given row of the sparse matrix
   * i.e. row r corresponds to the indices [row_ptr[r], row_ptr[r+1])
   */
  std::vector<int> row_ptr;

  /// @brief array holding the column associated with each nonzero entry
  std::vector<int> col_ind;

  /**
   * @brief element_nonzero_LUT(e, i, j) says where (in the global sparse matrix)
   * to put the (i, j) component of the matrix associated with element element matrix `e`
   */
  serac::CPUArray<SignedIndex, 3> element_nonzero_LUT;

  /**
   * @brief bdr_element_nonzero_LUT(b, i, j) says where (in the global sparse matrix)
   * to put the (i, j) component of the matrix associated with boundary element element matrix `b`
   */
  serac::CPUArray<SignedIndex, 3> bdr_element_nonzero_LUT;
};

}  // namespace serac
