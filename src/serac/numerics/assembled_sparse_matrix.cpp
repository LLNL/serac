
#include "mfem/linalg/dtensor.hpp"
#include "assembled_sparse_matrix.hpp"

template <auto offsetV, auto indicesV, auto gatherV>
struct forbidden_restriction {
  //  friend result_fes _get_fes(From & from) { return & from.*fesV; }
  friend mfem::Array<int>& __get_offsets(mfem::ElementRestriction& from) { return from.*offsetV; }
  friend mfem::Array<int>& __get_indices(mfem::ElementRestriction& from) { return from.*indicesV; }
  friend mfem::Array<int>& __get_gatherMap(mfem::ElementRestriction& from) { return from.*gatherV; }
};

// const FiniteElementSpace * __get_fes(mfem::ElementRestriction &);
mfem::Array<int>& __get_offsets(mfem::ElementRestriction&);
mfem::Array<int>& __get_indices(mfem::ElementRestriction&);
mfem::Array<int>& __get_gatherMap(mfem::ElementRestriction&);

template struct forbidden_restriction<&mfem::ElementRestriction::offsets, &mfem::ElementRestriction::indices,
                                      &mfem::ElementRestriction::gatherMap>;

namespace serac {
  namespace mfem_ext {

    AssembledSparseMatrix::AssembledSparseMatrix(const mfem::FiniteElementSpace& test,   // test_elem_dofs * ne * vdim x vdim * test_ndofs
						 const mfem::FiniteElementSpace& trial,  // trial_elem_dofs * ne * vdim x vdim * trial_ndofs
						 mfem::ElementDofOrdering        elem_order)
      : mfem::SparseMatrix(test.GetNDofs() * test.GetVDim(), trial.GetNDofs() * trial.GetVDim()),
      test_fes(test),
      trial_fes(trial),
      test_restriction(test, elem_order),
      trial_restriction(trial, elem_order),
      elem_ordering(elem_order)
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
      [[maybe_unused]] auto& test_offsets = __get_offsets(test_restriction);
      [[maybe_unused]] auto& test_indices = __get_indices(test_restriction);    
      [[maybe_unused]] auto& test_gatherMap  = __get_gatherMap(test_restriction);
      [[maybe_unused]] auto& trial_offsets   = __get_offsets(trial_restriction);
      [[maybe_unused]] auto& trial_indices   = __get_indices(trial_restriction);
      [[maybe_unused]] auto& trial_gatherMap = __get_gatherMap(trial_restriction);

      /**
	 We expect mat_ea to be of size (test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne)
	 We assume a consistent striding from (elem_dof, vd) within each element
      */
      const int test_elem_dof  = test_fes.GetFE(0)->GetDof();
      const int trial_elem_dof = trial_fes.GetFE(0)->GetDof();
      const int test_vdim      = test_fes.GetVDim();
      const int trial_vdim     = trial_fes.GetVDim();
      const int test_ndofs     = test_fes.GetNDofs();

      for (int i = 0; i < test_vdim * test_ndofs; i++) {
	I[i] = 0;
      }

      for (int test_vdof = 0; test_vdof < test_fes.GetNDofs(); test_vdof++) {
	// Look through each element corresponding to a test_vdof
	const int test_row_offset = test_offsets[test_vdof];
	const int nrow_elems      = test_offsets[test_vdof + 1] - test_row_offset;

	// Build temporary array to get rid of duplicates
	mfem::Array<int> trial_vdofs(nrow_elems * trial_elem_dof);
	trial_vdofs = -1;
	int nnz_row = 0;
	for (int e_index = 0; e_index < nrow_elems; e_index++) {
	  // test_indices can be negative in the case of Hcurl
	  const int                  test_index_v = test_indices[test_row_offset + e_index];
	  const int                  test_index = test_index_v >= 0 ? test_index_v : -test_index_v - 1;
	  const int                  e           = test_index / test_elem_dof;
	  [[maybe_unused]] const int test_i_elem = test_index % test_elem_dof;

	  // find corresponding trial_vdofs
	  mfem::Array<int> trial_elem_vdofs(trial_elem_dof);
	  for (int j = 0; j < trial_elem_dof; j++) {
	    // this might be negative
	    const auto trial_j_vdof_v = trial_gatherMap[trial_elem_dof * e + j];
	    const auto trial_j_vdof = trial_j_vdof_v >= 0 ? trial_j_vdof_v : -1 -trial_j_vdof_v;
	    trial_elem_vdofs[j]     = trial_j_vdof;
	    if (trial_vdofs.Find(trial_j_vdof) == -1) {
	      // we haven't seen this before
	      trial_vdofs[nnz_row] = trial_j_vdof;
	      nnz_row++;
	    }
	  }
	}

	// add entries to I
	for (int vi = 0; vi < test_vdim; vi++) {
	  const auto nnz_index_v = test_fes.DofToVDof(test_vdof, vi);
	  const auto nnz_index = nnz_index_v >= 0 ? nnz_index_v : -1 -nnz_index_v;
	  I[nnz_index] = nnz_row * trial_vdim;
	}
      }

      // Perform inclusive scan on all entries
      int nnz = 0;
      for (int i = 0; i < test_ndofs * trial_vdim; i++) {
	int temp = I[i];
	I[i]     = nnz;
	nnz += temp;
      }
      I[test_ndofs * trial_vdim] = nnz;

      return nnz;
    }

    void AssembledSparseMatrix::FillJ()
    {

      [[maybe_unused]] auto& test_offsets = __get_offsets(test_restriction);
      [[maybe_unused]] auto& test_indices = __get_indices(test_restriction);    
      [[maybe_unused]] auto& test_gatherMap  = __get_gatherMap(test_restriction);
      [[maybe_unused]] auto& trial_offsets   = __get_offsets(trial_restriction);
      [[maybe_unused]] auto& trial_indices   = __get_indices(trial_restriction);
      [[maybe_unused]] auto& trial_gatherMap = __get_gatherMap(trial_restriction);
  
      const int                  test_elem_dof  = test_fes.GetFE(0)->GetDof();
      const int                  trial_elem_dof = trial_fes.GetFE(0)->GetDof();
      const int                  test_vdim      = test_fes.GetVDim();
      const int                  trial_vdim     = trial_fes.GetVDim();
      [[maybe_unused]] const int test_ndofs     = test_fes.GetNDofs();

      const int ne = trial_fes.GetNE();
      ea_map.SetSize(test_elem_dof * test_vdim * trial_elem_dof * trial_vdim * ne);
      auto map_ea = mfem::Reshape(ea_map.ReadWrite(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

      // initialize J
      for (int j = 0; j < this->J.Capacity(); j++) {
	this->J[j] = -1;
      }

      for (int test_vdof = 0; test_vdof < test_fes.GetNDofs(); test_vdof++) {
	// Look through each element corresponding to a test_vdof
	const int test_row_offset = test_offsets[test_vdof];
	const int nrow_elems      = test_offsets[test_vdof + 1] - test_row_offset;

	// here we assume all the components have the same number of columns
	const int        nnz_row = I[test_fes.DofToVDof(test_vdof, 0) + 1] - I[test_fes.DofToVDof(test_vdof, 0)];
	mfem::Array<int> trial_vdofs(nnz_row);
	trial_vdofs      = -1;
	int j_vdof_index = 0;

	// Build temporary array for assembled J
	for (int e_index = 0; e_index < nrow_elems; e_index++) {
	  // test_indices can be negative in the case of Hcurl
	  const int                  test_index_v = test_indices[test_row_offset + e_index];
	  const int                  test_index = test_index_v >= 0 ? test_index_v : -test_index_v - 1;
	  const int                  e           = test_index / test_elem_dof;
	  [[maybe_unused]] const int test_i_elem = test_index % test_elem_dof;

	  // find corresponding trial_vdofs
	  mfem::Array<int> trial_elem_vdofs(trial_elem_dof);
	  for (int j_elem = 0; j_elem < trial_elem_dof; j_elem++) {
	    // could be negative.. but trial_elem_vdofs is a temporary array
	    const auto trial_j_vdof_v  = trial_gatherMap[trial_elem_dof * e + j_elem];
	    const auto trial_j_vdof = trial_j_vdof_v >= 0 ? trial_j_vdof_v : -1 -trial_j_vdof_v;
	    trial_elem_vdofs[j_elem] = trial_j_vdof;

	    // since trial_j_vdof could be negative but there are now two indices that point to the same dof (just oriented differently).. we only want to search for positive oriented indices
	    auto find_index = trial_vdofs.Find(trial_j_vdof);
	    if (find_index == -1) {
	      // we haven't seen this before
	      trial_vdofs[j_vdof_index] = trial_j_vdof;

	      // we can add this entry to J
	      for (int vi = 0; vi < test_vdim; vi++) {
		const auto i_dof_offset = I[test_fes.DofToVDof(test_vdof, vi)];

		// this access pattern corresnponds to j_vdof_index + vj * nnz_row
		for (int vj = 0; vj < trial_vdim; vj++) {
		  const auto column_index = j_vdof_index + vj * nnz_row / trial_vdim;
		  const auto j_nnz_index  = i_dof_offset + column_index;
		  // this index may be negative, but J needs to be positive
		  const auto j_value = trial_fes.DofToVDof(trial_j_vdof_v, vj);
		  J[j_nnz_index]          = j_value >= 0 ? j_value : -1-j_value;
		}
	      }

	      // write mapping from ea to csr_nnz_index (can probably optimize this)
	      for (int vi = 0; vi < test_vdim; vi++) {
		const auto i_dof_offset = I[test_fes.DofToVDof(test_vdof, vi)];
		for (int vj = 0; vj < trial_vdim; vj++) {
		  const auto column_index = j_vdof_index + vj * nnz_row / trial_vdim;
		  const int index_val = i_dof_offset + column_index;
		  const int trial_index = trial_fes.DofToVDof(trial_j_vdof_v, vj);
		  const int orientation_factor = (test_index_v >= 0 ? 1 : -1) * (trial_index >=0 ? 1 : -1);
		  map_ea(test_i_elem + test_elem_dof * vi, j_elem + trial_elem_dof * vj, e) =
		    orientation_factor > 0 ? index_val : -1-index_val;
		}
	      }

	      j_vdof_index++;
	    } else {
	      // this is a duplicate entry
	      // write mapping from ea to csr_nnz_index (can probably optimize this)
	      for (int vi = 0; vi < test_vdim; vi++) {
		const auto i_dof_offset = I[test_fes.DofToVDof(test_vdof, vi)];
		for (int vj = 0; vj < trial_vdim; vj++) {
		  const auto column_index = find_index + vj * nnz_row / trial_vdim;
		  const int index_val = i_dof_offset + column_index;
		  const int trial_index = trial_fes.DofToVDof(trial_j_vdof_v, vj);
		  const int orientation_factor = (test_index_v >= 0 ? 1 : -1) * (trial_index >=0 ? 1 : -1);      
		  map_ea(test_i_elem + test_elem_dof * vi, j_elem + trial_elem_dof * vj, e) =
		    orientation_factor > 0 ? index_val : -1-index_val;
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

      [[maybe_unused]] auto& test_offsets = __get_offsets(test_restriction);
      [[maybe_unused]] auto& test_indices = __get_indices(test_restriction);    
      [[maybe_unused]] auto& test_gatherMap  = __get_gatherMap(test_restriction);
      [[maybe_unused]] auto& trial_offsets   = __get_offsets(trial_restriction);
      [[maybe_unused]] auto& trial_indices   = __get_indices(trial_restriction);
      [[maybe_unused]] auto& trial_gatherMap = __get_gatherMap(trial_restriction);
  
      const int                  test_elem_dof  = test_fes.GetFE(0)->GetDof();
      const int                  trial_elem_dof = trial_fes.GetFE(0)->GetDof();
      const int                  test_vdim      = test_fes.GetVDim();
      const int                  trial_vdim     = trial_fes.GetVDim();
      [[maybe_unused]] const int test_ndofs     = test_fes.GetNDofs();

      const int ne = trial_fes.GetNE();

      //      auto map_ea = mfem::Reshape(ea_map.Read(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);
      auto map_ea = mfem::Reshape(ea_map.Read(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

      auto mat_ea = mfem::Reshape(ea_data.Read(), test_elem_dof * test_vdim, trial_elem_dof * trial_vdim, ne);

      // Use map_ea to take ea_data directly to CSR entry
      for (int e = 0; e < ne; e++) {
	for (int i_elem = 0; i_elem < test_elem_dof; i_elem++) {
	  for (int vi = 0; vi < test_vdim; vi++) {
	    for (int j_elem = 0; j_elem < trial_elem_dof; j_elem++) {
	      for (int vj = 0; vj < trial_vdim; vj++) {
		const auto map_ea_v = map_ea(i_elem + vi * test_elem_dof, j_elem + vj * trial_elem_dof, e);
		const auto map_ea_index = map_ea_v >= 0 ? map_ea_v : -1 -map_ea_v;
		Data[map_ea_index] += (map_ea_v >= 0 ? 1 : -1) * 
		  mat_ea(i_elem + vi * test_elem_dof, j_elem + vj * trial_elem_dof, e);
	      }
	    }
	  }
	}
      }
    }

  }  // namespace mfem_ext

}  // namespace serac
