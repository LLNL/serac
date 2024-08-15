// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"

#include "serac/numerics/functional/integral.hpp"
#include "serac/numerics/functional/element_restriction.hpp"

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
  uint32_t     global_row_;  ///< The global row number
  uint32_t     global_col_;  ///< The global column number
  uint32_t     local_row_;   ///< The local row number
  uint32_t     local_col_;   ///< The global column number
  uint32_t     element_id_;  ///< The element ID
  int          sign_;        ///< The orientation of the element
  Domain::Type type;         ///< Which kind of Integral this entry comes from
};

/**
 * @brief operator for sorting lexicographically by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
inline bool operator<(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row_ < y.global_row_) || (x.global_row_ == y.global_row_ && x.global_col_ < y.global_col_);
}

/**
 * @brief operator determining inequality by {global_row, global_col}
 * @param x the ElemInfo on the left
 * @param y the ElemInfo on the right
 */
inline bool operator!=(const ElemInfo& x, const ElemInfo& y)
{
  return (x.global_row_ != y.global_row_) || (x.global_col_ != y.global_col_);
}

/**
 * @brief this type explicitly stores sign (typically used conveying edge/face orientation) and index values
 *
 * TODO: investigate implementation via bitfield (should have smaller memory footprint, better readability than mfem's
 * {sign, index} int32_t encoding)
 */
struct SignedIndex {
  /// the actual index of some quantity
  uint32_t index_;

  /// whether or not the value associated with this index is positive or negative
  int sign_;

  /// the implicit conversion to int extracts only the index
  operator uint32_t() { return index_; }
};

/**
 * @brief mfem will frequently encode {sign, index} into a single int32_t.
 * This function decodes those values.
 *
 * @param i an integer that mfem has encoded to contain two separate pieces of information
 */
inline SignedIndex decodeSignedIndex(int i)
{
  return SignedIndex{static_cast<uint32_t>((i >= 0) ? i : -1 - i), (i >= 0) ? 1 : -1};
}

/**
 * @brief return whether or not the underlying function space is Hcurl or not
 *
 * @param fes the finite element space in question
 */
inline bool isHcurl(const mfem::ParFiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::TANGENTIAL);
}

/**
 * @brief return whether or not the underlying function space is L2 or not
 *
 * @param fes the finite element space in question
 */
inline bool isL2(const mfem::ParFiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS);
}

/**
 * @brief attempt to characterize which FiniteElementSpaces
 * mfem::FaceRestriction actually works with
 *
 * @param fes the finite element space in question
 */
inline bool compatibleWithFaceRestriction(const mfem::ParFiniteElementSpace& fes)
{
  return !(isHcurl(fes) && fes.GetMesh()->Dimension() == 2) && !(isHcurl(fes) && fes.GetMesh()->Dimension() == 3) &&
         !(isL2(fes)) && fes.GetMesh()->GetNBE() > 0;
}

/**
 * @brief this is a (hopefully) temporary measure to work around the fact that mfem's
 * support for querying information about boundary elements is inconsistent, or entirely
 * unimplemented. If the finite element spaces both work with mfem::FaceRestriction, it will
 * return a 3D array sized to store the boundary element gradient matrices, else the 3D array
 * will have dimensions 0x0x0 to indicate that it is unused.
 *
 * @param trial_fes the trial finite element space
 * @param test_fes the test finite element space
 *
 * known issues: getting dofs/ids for boundary elements in 2D w/ Hcurl spaces
 *               getting dofs/ids for boundary elements in 2D,3D w/ L2 spaces
 */
template <typename T, ExecutionSpace exec>
ExecArray<T, 3, exec> allocateMemoryForBdrElementGradients(const mfem::ParFiniteElementSpace& trial_fes,
                                                           const mfem::ParFiniteElementSpace& test_fes)
{
  auto* test_BE  = test_fes.GetBE(0);
  auto* trial_BE = trial_fes.GetBE(0);
  return {static_cast<size_t>(trial_fes.GetNFbyType(mfem::FaceType::Boundary)),
          static_cast<size_t>(test_BE->GetDof() * test_fes.GetVDim()),
          static_cast<size_t>(trial_BE->GetDof() * trial_fes.GetVDim())};
}

/// @overload
template <typename T, ExecutionSpace exec>
ExecArray<T, 2, exec> allocateMemoryForBdrElementGradients(const mfem::ParFiniteElementSpace& fes)
{
  if (compatibleWithFaceRestriction(fes)) {
    auto* BE = fes.GetBE(0);
    return {static_cast<size_t>(fes.GetNFbyType(mfem::FaceType::Boundary)),
            static_cast<size_t>(BE->GetDof() * fes.GetVDim())};
  } else {
    return {0, 0};
  }
}

/**
 * @brief this object extracts the dofs for each element in a FiniteElementSpace as a 2D array such that
 *   element_dofs_(e, i) will be the `i`th dof of element `e`.
 *
 * Note: due to an internal inconsistency between mfem::FiniteElementSpace and mfem::FaceRestriction,
 *    we choose to use the Restriction operator as the "source of truth", since we are also using its
 *    convention for quadrature point numbering.
 */

struct DofNumbering {
  /**
   * @param fespace the finite element space to extract dof numbers from
   *
   * @brief create lookup tables of which degrees of freedom correspond to
   * each element and boundary element
   */
  DofNumbering(const mfem::ParFiniteElementSpace& fespace)
      : element_dofs_(static_cast<size_t>(fespace.GetNE()),
                      static_cast<size_t>(fespace.GetFE(0)->GetDof() * fespace.GetVDim())),
        bdr_element_dofs_(allocateMemoryForBdrElementGradients<SignedIndex, ExecutionSpace::CPU>(fespace))
  {
    int                  dim          = fespace.GetMesh()->Dimension();
    mfem::Geometry::Type elem_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT, mfem::Geometry::SQUARE,
                                         mfem::Geometry::CUBE};
    ElementRestriction   dofs(&fespace, elem_geom[dim]);
    ElementRestriction   boundary_dofs(&fespace, elem_geom[dim - 1], FaceType::BOUNDARY);

    {
      auto elem_restriction = fespace.GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);

      mfem::Vector iota(elem_restriction->Width());
      mfem::Vector dof_ids(elem_restriction->Height());
      dof_ids = 0.0;
      for (int i = 0; i < iota.Size(); i++) {
        iota[i] = i + 1;  //  note: 1-based index
      }

      // we're using Mult() to reveal the locations nonzero entries
      // in the restriction operator, since that information is not
      // made available through its public interface
      //
      // TODO: investigate refactoring mfem's restriction operators
      // to provide this information in more natural way.
      elem_restriction->Mult(iota, dof_ids);
      const double* dof_ids_h = dof_ids.HostRead();

      int index = 0;
      for (axom::IndexType e = 0; e < element_dofs_.shape()[0]; e++) {
        for (axom::IndexType i = 0; i < element_dofs_.shape()[1]; i++) {
          uint32_t dof_id     = static_cast<uint32_t>(fabs(dof_ids_h[index]));  // note: 1-based index
          int      dof_sign   = dof_ids[index] > 0 ? +1 : -1;
          element_dofs_(e, i) = {dof_id - 1, dof_sign};  // subtract 1 to get back to 0-based index
          index++;
        }
      }
    }

    if (bdr_element_dofs_.size() > 0) {
      auto face_restriction = fespace.GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                                         mfem::FaceType::Boundary, mfem::L2FaceValues::SingleValued);

      mfem::Vector iota(face_restriction->Width());
      mfem::Vector dof_ids(face_restriction->Height());
      for (int i = 0; i < iota.Size(); i++) {
        iota[i] = i + 1;  //  note: 1-based index
      }

      face_restriction->Mult(iota, dof_ids);
      const double* dof_ids_h = dof_ids.HostRead();

      int index = 0;
      for (axom::IndexType e = 0; e < bdr_element_dofs_.shape()[0]; e++) {
        for (axom::IndexType i = 0; i < bdr_element_dofs_.shape()[1]; i++) {
          uint32_t dof_id         = static_cast<uint32_t>(fabs(dof_ids_h[index]));  // note: 1-based index
          int      dof_sign       = dof_ids[index] > 0 ? +1 : -1;
          bdr_element_dofs_(e, i) = {dof_id - 1, dof_sign};  // subtract 1 to get back to 0-based index
          index++;
        }
      }
    }
  }

  /// @brief element_dofs_(e, i) stores the `i`th dof of element `e`.
  CPUArray<SignedIndex, 2> element_dofs_;

  /// @brief bdr_element_dofs_(b, i) stores the `i`th dof of boundary element `b`.
  CPUArray<SignedIndex, 2> bdr_element_dofs_;
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
  /// @brief a type for representing a nonzero entry in a sparse matrix
  struct Entry {
    uint32_t row;     ///< row value for this nonzero Entry
    uint32_t column;  ///< column value for this nonzero Entry

    /// operator< is used when sorting `Entry`. Lexicographical ordering
    bool operator<(const Entry& other) const
    {
      return (row < other.row) || ((row == other.row) && (column < other.column));
    }

    /// operator== is required for use in `std::unordered_map`
    bool operator==(const Entry& other) const { return (row == other.row && column == other.column); }

    /// hash functor required for use in `std::unordered_map`
    struct Hasher {
      /// @brief a hash function implementation for `Entry`
      std::size_t operator()(const Entry& k) const
      {
        std::size_t seed = std::hash<uint32_t>()(k.row);
        seed ^= std::hash<uint32_t>()(k.column) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
      }
    };
  };

  /// dummy default ctor to enable deferred initialization
  GradientAssemblyLookupTables() : initialized{false} {};

  /**
   * @param block_test_dofs object containing information about dofs for the test space
   * @param block_trial_dofs object containing information about dofs for the trial space
   *
   * @brief create lookup tables describing which degrees of freedom
   * correspond to each domain/boundary element
   */
  void init(const serac::BlockElementRestriction& block_test_dofs,
            const serac::BlockElementRestriction& block_trial_dofs)
  {
    // we start by having each element and boundary element emit the (i,j) entry that it
    // touches in the global "stiffness matrix", and also keep track of some metadata about
    // which element and which dof are associated with that particular nonzero entry
    for (const auto& [geometry, trial_dofs] : block_trial_dofs.restrictions) {
      const auto& test_dofs = block_test_dofs.restrictions.at(geometry);

      std::vector<DoF> test_vdofs(test_dofs.nodes_per_elem * test_dofs.components);
      std::vector<DoF> trial_vdofs(trial_dofs.nodes_per_elem * trial_dofs.components);

      auto num_elements = static_cast<uint32_t>(trial_dofs.num_elements);
      for (uint32_t e = 0; e < num_elements; e++) {
        for (uint64_t i = 0; i < uint64_t(test_dofs.dof_info.shape()[1]); i++) {
          auto test_dof = test_dofs.dof_info(e, i);

          for (uint64_t j = 0; j < uint64_t(trial_dofs.dof_info.shape()[1]); j++) {
            auto trial_dof = trial_dofs.dof_info(e, j);

            for (uint64_t k = 0; k < test_dofs.components; k++) {
              uint32_t test_global_id = uint32_t(test_dofs.GetVDof(test_dof, k).index());
              for (uint64_t l = 0; l < trial_dofs.components; l++) {
                uint32_t trial_global_id                  = uint32_t(trial_dofs.GetVDof(trial_dof, l).index());
                nz_LUT[{test_global_id, trial_global_id}] = 0;  // just store the keys initially
              }
            }
          }
        }
      }
    }
    std::vector<Entry> entries(nz_LUT.size());

    uint32_t count = 0;
    for (auto [key, value] : nz_LUT) {
      entries[count++] = key;
    }

    std::sort(entries.begin(), entries.end());

    nnz = static_cast<uint32_t>(nz_LUT.size());
    row_ptr.resize(static_cast<size_t>(block_test_dofs.LSize() + 1));
    col_ind.resize(nnz);

    row_ptr[0] = 0;
    col_ind[0] = int(entries[0].column);

    for (uint32_t i = 1; i < nnz; i++) {
      nz_LUT[entries[i]] = i;
      col_ind[i]         = int(entries[i].column);

      // if the new entry has a different row, then the row_ptr offsets must be set as well
      for (uint32_t j = entries[i - 1].row; j < entries[i].row; j++) {
        row_ptr[j + 1] = int(i);
      }
    }

    row_ptr.back() = static_cast<int>(nnz);

    initialized = true;
  }

  /**
   * @brief return the index (into the nonzero entries) corresponding to entry (i,j)
   * @param i the row
   * @param j the column
   */
  uint32_t operator()(int i, int j) const { return nz_LUT.at({uint32_t(i), uint32_t(j)}); }

  /// @brief how many nonzero entries appear in the sparse matrix
  uint32_t nnz;

  /**
   * @brief array holding the offsets for a given row of the sparse matrix
   * i.e. row r corresponds to the indices [row_ptr[r], row_ptr[r+1])
   */
  std::vector<int> row_ptr;

  /// @brief array holding the column associated with each nonzero entry
  std::vector<int> col_ind;

  /**
   * @brief `nz_LUT` returns the index of the `col_ind` / `value` CSR arrays
   * corresponding to the (i,j) entry
   */
  std::unordered_map<Entry, uint32_t, Entry::Hasher> nz_LUT;

  /// @brief specifies if the table has already been initialized or not
  bool initialized;
};

}  // namespace serac
