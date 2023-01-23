#pragma once

#include "mfem.hpp"

#include "serac/infrastructure/accelerator.hpp"

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
  uint32_t global_row_;   ///< The global row number
  uint32_t global_col_;   ///< The global column number
  uint32_t local_row_;    ///< The local row number
  uint32_t local_col_;    ///< The global column number
  uint32_t element_id_;   ///< The element ID
  int      sign_;         ///< The orientation of the element
  bool     on_boundary_;  ///< True if element is on the boundary
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
SignedIndex decodeSignedIndex(int i)
{
  return SignedIndex{static_cast<uint32_t>((i >= 0) ? i : -1 - i), (i >= 0) ? 1 : -1};
}

/**
 * @brief return whether or not the underlying function space is Hcurl or not
 *
 * @param fes the finite element space in question
 */
bool isHcurl(const mfem::ParFiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::TANGENTIAL);
}

/**
 * @brief return whether or not the underlying function space is L2 or not
 *
 * @param fes the finite element space in question
 */
bool isL2(const mfem::ParFiniteElementSpace& fes)
{
  return (fes.FEColl()->GetContType() == mfem::FiniteElementCollection::DISCONTINUOUS);
}

/**
 * @brief attempt to characterize which FiniteElementSpaces
 * mfem::FaceRestriction actually works with
 *
 * @param fes the finite element space in question
 */
bool compatibleWithFaceRestriction(const mfem::ParFiniteElementSpace& fes)
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
  if (compatibleWithFaceRestriction(test_fes) && compatibleWithFaceRestriction(trial_fes)) {
    auto* test_BE  = test_fes.GetBE(0);
    auto* trial_BE = trial_fes.GetBE(0);
    return {static_cast<size_t>(trial_fes.GetNFbyType(mfem::FaceType::Boundary)),
            static_cast<size_t>(test_BE->GetDof() * test_fes.GetVDim()),
            static_cast<size_t>(trial_BE->GetDof() * trial_fes.GetVDim())};
  } else {
    return {0, 0, 0};
  }
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

    int dim = fespace.GetMesh()->Dimension();
    mfem::Geometry::Type elem_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT, mfem::Geometry::SQUARE, mfem::Geometry::CUBE};
    ElementRestriction dofs(&fespace, elem_geom[dim]);
    ElementRestriction boundary_dofs(&fespace, elem_geom[dim-1], FaceType::BOUNDARY);

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

          std::cout << dof_id - 1 << " " << boundary_dofs.dof_info(e, i).index() << std::endl;
          index++;
        }
        std::cout << std::endl;
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
  /**
   * @param test_fespace the test finite element space to extract dof numbers from
   * @param trial_fespace the trial finite element space to extract dof numbers from
   *
   * @brief create lookup tables of which degrees of freedom correspond to
   * each element and boundary element
   */
  GradientAssemblyLookupTables(const mfem::ParFiniteElementSpace& test_fespace,
                               const mfem::ParFiniteElementSpace& trial_fespace)
      : element_nonzero_LUT(static_cast<size_t>(trial_fespace.GetNE()),
                            static_cast<size_t>(test_fespace.GetFE(0)->GetDof() * test_fespace.GetVDim()),
                            static_cast<size_t>(trial_fespace.GetFE(0)->GetDof() * trial_fespace.GetVDim())),
        bdr_element_nonzero_LUT(
            allocateMemoryForBdrElementGradients<SignedIndex, ExecutionSpace::CPU>(trial_fespace, test_fespace))
  {
    int dim = test_fespace.GetMesh()->Dimension();
    mfem::Geometry::Type elem_geom[4] = {mfem::Geometry::INVALID, mfem::Geometry::SEGMENT, mfem::Geometry::SQUARE, mfem::Geometry::CUBE};
    
    std::vector<ElemInfo> infos;

    // we start by having each element and boundary element emit the (i,j) entry that it
    // touches in the global "stiffness matrix", and also keep track of some metadata about
    // which element and which dof are associated with that particular nonzero entry
    {
      bool on_boundary = false;

      serac::ElementRestriction test_dofs(&test_fespace, elem_geom[dim]);
      serac::ElementRestriction trial_dofs(&trial_fespace, elem_geom[dim]);

      std::vector< DoF > test_vdofs(test_dofs.nodes_per_elem * test_dofs.components);
      std::vector< DoF > trial_vdofs(trial_dofs.nodes_per_elem * trial_dofs.components);

      auto num_elements     = static_cast<uint32_t>(trial_fespace.GetNE());
      for (uint32_t e = 0; e < num_elements; e++) {
        for (uint64_t i = 0; i < test_dofs.dof_info.dim[1]; i++) {
          auto test_dof = test_dofs.dof_info(e, i);

          for (uint64_t j = 0; j < trial_dofs.dof_info.dim[1]; j++) {
            auto trial_dof = trial_dofs.dof_info(e, j);

            for (uint64_t k = 0; k < test_dofs.components; k++) {
              for (uint64_t l = 0; l < trial_dofs.components; l++) {

                uint32_t test_global_id = uint32_t(test_dofs.GetVDof(test_dof, k).index());
                uint32_t trial_global_id = uint32_t(trial_dofs.GetVDof(trial_dof, l).index());
                uint32_t test_local_id = uint32_t(k * test_dofs.nodes_per_elem + i);
                uint32_t trial_local_id = uint32_t(l * trial_dofs.nodes_per_elem + j);

                ElemInfo info {
                  test_global_id,
                  trial_global_id,
                  test_local_id,
                  trial_local_id,
                  e,
                  test_dof.sign() * trial_dof.sign(),
                  on_boundary
                };

                infos.push_back(info);
              }
            }
          }
        }
      }
    }

    {
      bool on_boundary = true;

      serac::ElementRestriction test_boundary_dofs(&test_fespace, elem_geom[dim-1], FaceType::BOUNDARY);
      serac::ElementRestriction trial_boundary_dofs(&trial_fespace, elem_geom[dim-1], FaceType::BOUNDARY);

      std::vector< DoF > test_vdofs(test_boundary_dofs.nodes_per_elem * test_boundary_dofs.components);
      std::vector< DoF > trial_vdofs(trial_boundary_dofs.nodes_per_elem * trial_boundary_dofs.components);

      auto num_bdr_elements = static_cast<uint32_t>(trial_fespace.GetNFbyType(mfem::FaceType::Boundary));
      for (uint32_t e = 0; e < num_bdr_elements; e++) {
        for (uint64_t i = 0; i < test_boundary_dofs.dof_info.dim[1]; i++) {
          auto test_dof = test_boundary_dofs.dof_info(e, i);

          for (uint64_t j = 0; j < trial_boundary_dofs.dof_info.dim[1]; j++) {
            auto trial_dof = trial_boundary_dofs.dof_info(e, j);

            for (uint64_t k = 0; k < test_boundary_dofs.components; k++) {
              for (uint64_t l = 0; l < trial_boundary_dofs.components; l++) {

                uint32_t test_global_id = uint32_t(test_boundary_dofs.GetVDof(test_dof, k).index());
                uint32_t trial_global_id = uint32_t(trial_boundary_dofs.GetVDof(trial_dof, l).index());
                uint32_t test_local_id = uint32_t(test_boundary_dofs.components * i + k);
                uint32_t trial_local_id = uint32_t(trial_boundary_dofs.components * j + l);

                ElemInfo info {
                  test_global_id,
                  trial_global_id,
                  test_local_id,
                  trial_local_id,
                  e,
                  test_dof.sign() * trial_dof.sign(),
                  on_boundary
                };

                infos.push_back(info);

              }
            }
          }
        }
      }

    }

    // sorting the ElemInfos by row and column groups the different contributions
    // to the same location of the global stiffness matrix, and makes it easy to identify
    // the unique entries
    std::sort(infos.begin(), infos.end());

    // the row_ptr array size only depends on the number of rows in the global stiffness matrix,
    // so we already know its size before processing the ElemInfo array
    row_ptr.resize(static_cast<size_t>(test_fespace.GetNDofs() * test_fespace.GetVDim() + 1));

    // the other CSR matrix arrays are formed incrementally by going through the sorted ElemInfo values
    std::vector<SignedIndex> nonzero_ids(infos.size());

    nnz        = 0;
    row_ptr[0] = 0;
    col_ind.push_back(static_cast<int>(infos[0].global_col_));
    nonzero_ids[0] = {0, infos[0].sign_};

    for (size_t i = 1; i < infos.size(); i++) {
      // increment the nonzero count every time we find a new (i,j) entry
      nnz += (infos[i - 1] != infos[i]);

      // record the index, sign, and column of this particular (i,j) entry
      nonzero_ids[i] = SignedIndex{nnz, infos[i].sign_};
      if (infos[i - 1] != infos[i]) {
        col_ind.push_back(static_cast<int>(infos[i].global_col_));
      }

      // if the new entry has a different row, then the row_ptr offsets must be set as well
      for (uint32_t j = infos[i - 1].global_row_; j < infos[i].global_row_; j++) {
        row_ptr[j + 1] = static_cast<int>(nonzero_ids[i]);
      }
    }

    row_ptr.back() = static_cast<int>(++nnz);

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
  uint32_t nnz;

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
