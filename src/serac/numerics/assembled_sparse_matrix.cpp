
#include "mfem/linalg/dtensor.hpp"
#include "assembled_sparse_matrix.hpp"

namespace detail {
/**
 * We'd like to access the protected offset, indices, and gatherMap variables from mfem::ElementRestriction
 * to compute our sparsity patterns in a manner consistent with MFEM.
 * In order to do that, we use the following template tricks.
 * Here's an article that explains the trick being used: https://accu.org/journals/overload/28/156/harrison_2776/
 */
template <auto offsetV, auto indicesV, auto gatherV>
struct forbidden_restriction {
  friend mfem::Array<int>& ElementRestriction_offsets(mfem::ElementRestriction& from) { return from.*offsetV; }
  friend mfem::Array<int>& ElementRestriction_indices(mfem::ElementRestriction& from) { return from.*indicesV; }
  friend mfem::Array<int>& ElementRestriction_gatherMap(mfem::ElementRestriction& from) { return from.*gatherV; }
};

mfem::Array<int>& ElementRestriction_offsets(mfem::ElementRestriction&);
mfem::Array<int>& ElementRestriction_indices(mfem::ElementRestriction&);
mfem::Array<int>& ElementRestriction_gatherMap(mfem::ElementRestriction&);

template struct forbidden_restriction<&mfem::ElementRestriction::offsets, &mfem::ElementRestriction::indices,
                                      &mfem::ElementRestriction::gatherMap>;
/**
 * @brief Account for HCURL-oriented index
 * @param[in] index Index-value. Can be negative for HCURL
 * @return corresponding valid index value in array (must be positive)
 */

template <typename T>
T oriented_index(T index)
{
  return index >= 0 ? index : -1 - index;
}

/**
 * @brief Returns sign of index
 * @param[in] index Index-value.
 * @return sign of index. +1 for 0 or positive, and -1 for negative
 */

template <typename T>
T sign(T index)
{
  return index >= 0 ? 1 : -1;
}

/**
 * @brief Adds HCURL orientation encoding to index
 * @param[in] index positive index value
 * @param[in] orientation Positive orientation or negative
 */
template <typename T>
T orient_index(T index, bool pos)
{
  return pos ? index : -1 - index;
}

}  // namespace detail

namespace serac {
namespace mfem_ext {

AssembledSparseMatrix::AssembledSparseMatrix(
    const mfem::ParFiniteElementSpace& test,   // test_elem_dofs * ne * vdim x vdim * test_ndofs
    const mfem::ParFiniteElementSpace& trial,  // trial_elem_dofs * ne * vdim x vdim * trial_ndofs
    mfem::ElementDofOrdering           elem_order)
    : mfem::SparseMatrix(test.GetNDofs() * test.GetVDim(), trial.GetNDofs() * trial.GetVDim()),
      test_fes_(test),
      trial_fes_(trial),
      test_restriction_(test, elem_order),
      trial_restriction_(trial, elem_order),
      elem_ordering_(elem_order)
{
  GetMemoryI().New(Height() + 1, GetMemoryI().GetMemoryType());

  const int nnz = FillI();
  GetMemoryJ().New(nnz, GetMemoryJ().GetMemoryType());
  GetMemoryData().New(nnz, GetMemoryData().GetMemoryType());
  FillJ();

  // zero initialize the data
  for (int i = 0; i < nnz; i++) {
    A[i] = 0.;
  }
}

int AssembledSparseMatrix::FillI()
{
  // ElementRestriction creates a CSR matrix that maps vdof -> (dof, ne).
  // offsets are the row offsets corresponding to a vdof
  // indices maps a given vdof to the the assembled element matrix vector (dof * ne + d).
  // gatherMap takes an element matrix vector offset (dof, ne) and returns the partition-local vdof (d.o.f. id).
  auto& test_offsets    = detail::ElementRestriction_offsets(test_restriction_);
  auto& test_indices    = detail::ElementRestriction_indices(test_restriction_);
  auto& trial_gatherMap = detail::ElementRestriction_gatherMap(trial_restriction_);

  /**
     We expect mat_ea to be of size (test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne)
     We assume a consistent striding from (elem_dof, vd) within each element
  */
  const int test_elem_dof  = test_fes_.GetFE(0)->GetDof();
  const int trial_elem_dof = trial_fes_.GetFE(0)->GetDof();
  const int test_vdim      = test_fes_.GetVDim();
  const int trial_vdim     = trial_fes_.GetVDim();

  std::fill(&I[0], &I[I.Capacity()], 0);

  for (int test_vdof = 0; test_vdof < test_fes_.GetNDofs(); test_vdof++) {
    // Look through each element corresponding to a test_vdof
    const int test_row_offset = test_offsets[test_vdof];
    const int nrow_elems      = test_offsets[test_vdof + 1] - test_row_offset;

    // Build temporary array to get rid of duplicates
    mfem::Array<int> trial_vdofs(nrow_elems * trial_elem_dof);
    trial_vdofs = -1;
    int nnz_row = 0;
    for (int e_index = 0; e_index < nrow_elems; e_index++) {
      // test_indices can be negative in the case of Hcurl
      const int test_index_v = test_indices[test_row_offset + e_index];
      const int test_index   = test_index_v >= 0 ? test_index_v : -test_index_v - 1;
      const int e            = test_index / test_elem_dof;

      // find corresponding trial_vdofs
      mfem::Array<int> trial_elem_vdofs(trial_elem_dof);
      for (int j = 0; j < trial_elem_dof; j++) {
        // this might be negative
        const auto trial_j_vdof_v = trial_gatherMap[trial_elem_dof * e + j];
        const auto trial_j_vdof   = trial_j_vdof_v >= 0 ? trial_j_vdof_v : -1 - trial_j_vdof_v;
        trial_elem_vdofs[j]       = trial_j_vdof;
        if (trial_vdofs.Find(trial_j_vdof) == -1) {
          // we haven't seen this before
          trial_vdofs[nnz_row] = trial_j_vdof;
          nnz_row++;
        }
      }
    }

    // add entries to I
    for (int vi = 0; vi < test_vdim; vi++) {
      const auto nnz_index_v = test_fes_.DofToVDof(test_vdof, vi);
      const auto nnz_index   = nnz_index_v >= 0 ? nnz_index_v : -1 - nnz_index_v;
      I[nnz_index]           = nnz_row * trial_vdim;
    }
  }

  // Perform exclusive scan
  // Note: Currently gcc8.3.1 doesn't support exclusive_scan
  // Use when possible: std::exclusive_scan(&I[0], &I[I.Capacity()], &I[0], 0);
  int nnz = 0;
  for (int i = 0; i < I.Capacity() - 1; i++) {
    int temp = I[i];
    I[i]     = nnz;
    nnz += temp;
  }
  I[I.Capacity() - 1] = nnz;

  return nnz;
}

void AssembledSparseMatrix::FillJ()
{
  auto& test_offsets    = detail::ElementRestriction_offsets(test_restriction_);
  auto& test_indices    = detail::ElementRestriction_indices(test_restriction_);
  auto& trial_gatherMap = detail::ElementRestriction_gatherMap(trial_restriction_);

  const int test_elem_dof  = test_fes_.GetFE(0)->GetDof();
  const int trial_elem_dof = trial_fes_.GetFE(0)->GetDof();
  const int test_vdim      = test_fes_.GetVDim();
  const int trial_vdim     = trial_fes_.GetVDim();

  const int ne = trial_fes_.GetNE();
  ea_map_.SetSize(test_elem_dof * test_vdim * trial_elem_dof * trial_vdim * ne);
  auto map_ea = mfem::Reshape(ea_map_.ReadWrite(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

  // initialize J
  for (int j = 0; j < this->J.Capacity(); j++) {
    this->J[j] = -1;
  }

  for (int test_vdof = 0; test_vdof < test_fes_.GetNDofs(); test_vdof++) {
    // Look through each element corresponding to a test_vdof
    const int test_row_offset = test_offsets[test_vdof];
    const int nrow_elems      = test_offsets[test_vdof + 1] - test_row_offset;

    // here we assume all the components have the same number of columns
    const int        nnz_row = I[test_fes_.DofToVDof(test_vdof, 0) + 1] - I[test_fes_.DofToVDof(test_vdof, 0)];
    mfem::Array<int> trial_vdofs(nnz_row);
    trial_vdofs      = -1;  // initialize with -1
    int j_vdof_index = 0;

    // Build temporary array for assembled J
    for (int e_index = 0; e_index < nrow_elems; e_index++) {
      // test_indices can be negative in the case of Hcurl
      const int test_index_v = test_indices[test_row_offset + e_index];
      const int test_index   = detail::oriented_index(test_index_v);
      const int e            = test_index / test_elem_dof;
      const int test_i_elem  = test_index % test_elem_dof;

      // find corresponding trial_vdofs
      mfem::Array<int> trial_elem_vdofs(trial_elem_dof);
      for (int j_elem = 0; j_elem < trial_elem_dof; j_elem++) {
        // could be negative.. but trial_elem_vdofs is a temporary array
        const auto trial_j_vdof_v = trial_gatherMap[trial_elem_dof * e + j_elem];
        const auto trial_j_vdof   = detail::oriented_index(trial_j_vdof_v);
        trial_elem_vdofs[j_elem]  = trial_j_vdof;

        // since trial_j_vdof could be negative but there are now two indices that point to the same dof (just oriented
        // differently).. we only want to search for positive oriented indices
        auto find_index = trial_vdofs.Find(trial_j_vdof);
        if (find_index == -1) {
          // we haven't seen this before
          trial_vdofs[j_vdof_index] = trial_j_vdof;

          // we can add this entry to J
          for (int vi = 0; vi < test_vdim; vi++) {
            const auto i_dof_offset = I[test_fes_.DofToVDof(test_vdof, vi)];

            // this access pattern corresnponds to j_vdof_index + vj * nnz_row
            for (int vj = 0; vj < trial_vdim; vj++) {
              const auto column_index = j_vdof_index + vj * nnz_row / trial_vdim;
              const auto j_nnz_index  = i_dof_offset + column_index;
              // this index may be negative, but J needs to be positive
              const auto j_value = trial_fes_.DofToVDof(trial_j_vdof_v, vj);
              J[j_nnz_index]     = detail::oriented_index(j_value);
            }
          }

          // write mapping from ea to csr_nnz_index (can probably optimize this)
          for (int vi = 0; vi < test_vdim; vi++) {
            const auto i_dof_offset = I[test_fes_.DofToVDof(test_vdof, vi)];
            for (int vj = 0; vj < trial_vdim; vj++) {
              const auto column_index       = j_vdof_index + vj * nnz_row / trial_vdim;
              const int  index_val          = i_dof_offset + column_index;
              const int  trial_index        = trial_fes_.DofToVDof(trial_j_vdof_v, vj);
              const int  orientation_factor = detail::sign(test_index_v) * detail::sign(trial_index);
              map_ea(test_i_elem + test_elem_dof * vi, j_elem + trial_elem_dof * vj, e) =
                  detail::orient_index(index_val, orientation_factor > 0);
            }
          }

          j_vdof_index++;
        } else {
          // this is a duplicate entry
          // write mapping from ea to csr_nnz_index (can probably optimize this)
          for (int vi = 0; vi < test_vdim; vi++) {
            const auto i_dof_offset = I[test_fes_.DofToVDof(test_vdof, vi)];
            for (int vj = 0; vj < trial_vdim; vj++) {
              const auto column_index       = find_index + vj * nnz_row / trial_vdim;
              const int  index_val          = i_dof_offset + column_index;
              const int  trial_index        = trial_fes_.DofToVDof(trial_j_vdof_v, vj);
              const int  orientation_factor = detail::sign(test_index_v) * detail::sign(trial_index);
              map_ea(test_i_elem + test_elem_dof * vi, j_elem + trial_elem_dof * vj, e) =
                  detail::orient_index(index_val, orientation_factor > 0);
            }
          }
        }
      }
    }
  }
}
void AssembledSparseMatrix::FillData(const mfem::Vector& ea_data)
{
  auto Data = WriteData();

  const int test_elem_dof  = test_fes_.GetFE(0)->GetDof();
  const int trial_elem_dof = trial_fes_.GetFE(0)->GetDof();
  const int test_vdim      = test_fes_.GetVDim();
  const int trial_vdim     = trial_fes_.GetVDim();

  const int ne = trial_fes_.GetNE();

  auto map_ea = mfem::Reshape(ea_map_.Read(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

  auto mat_ea = mfem::Reshape(ea_data.Read(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

  // Use map_ea to take ea_data directly to CSR entry
  for (int e = 0; e < ne; e++) {
    for (int i_elem = 0; i_elem < test_elem_dof; i_elem++) {
      for (int vi = 0; vi < test_vdim; vi++) {
        for (int j_elem = 0; j_elem < trial_elem_dof; j_elem++) {
          for (int vj = 0; vj < trial_vdim; vj++) {
            const auto map_ea_v     = map_ea(i_elem + vi * test_elem_dof, j_elem + vj * trial_elem_dof, e);
            const auto map_ea_index = detail::oriented_index(map_ea_v);
            Data[map_ea_index] +=
                detail::sign(map_ea_v) * mat_ea(i_elem + vi * test_elem_dof, j_elem + vj * trial_elem_dof, e);
          }
        }
      }
    }
  }
}

}  // namespace mfem_ext

}  // namespace serac
