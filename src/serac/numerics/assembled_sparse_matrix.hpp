#pragma once

#include <mfem.hpp>

namespace serac {
namespace mfem_ext {

/**
 @brief Creates a CSR sparse matrix from element matrices assembled usign a mfem::ElementDofOrdering
*/
class AssembledSparseMatrix : public mfem::SparseMatrix {
public:
  /**
   * @brief AssembledSparseMatrix creates a SparseMatrix based on finite element spaces and ElementDofOrdering
   *
   * @param[in] test Test finite element space
   * @param[in] trial Trial finite element space
   * @param[in] elem_order ElementDofOrdering chosen for both spaces
   */
  AssembledSparseMatrix(const mfem::FiniteElementSpace& test,   // test_elem_dofs * ne * vdim x vdim * test_ndofs
                        const mfem::FiniteElementSpace& trial,  // trial_elem_dofs * ne * vdim x vdim * trial_ndofs
                        mfem::ElementDofOrdering        elem_order);

  /// Updates SparseMatrix entries based on new element assembled matrices
  virtual void FillData(const mfem::Vector& ea_data);

protected:
  // Test space describing the sparsity pattern
  const mfem::FiniteElementSpace& test_fes;

  // Trial space describing the sparsity pattern
  const mfem::FiniteElementSpace& trial_fes;

  // Test space element restriction
  mfem::ElementRestriction test_restriction;

  // Trial space element restriction
  mfem::ElementRestriction trial_restriction;

  // Consistent element ordering
  mfem::ElementDofOrdering elem_ordering;

  // Maps individual element matrix entries in the K_e vector to the final CSR data offset
  mfem::Array<int> ea_map;

private:
  // Computes the row offsets for the CSR sparsity pattern given by the spaces test(trial)
  int FillI();
  // Computes the column indicies per row for a CSR sparsity pattern given by the spaces test(trial)
  void FillJ();
};
}  // namespace mfem_ext
}  // namespace serac
