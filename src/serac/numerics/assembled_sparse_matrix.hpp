#pragma once

#include <mfem.hpp>

namespace serac {
namespace mfem_ext {

/**
 @brief Creates a CSR sparse matrix from element matrices assembled using a mfem::ElementDofOrdering
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
  AssembledSparseMatrix(const mfem::ParFiniteElementSpace& test,   // test_elem_dofs * ne * vdim x vdim * test_ndofs
                        const mfem::ParFiniteElementSpace& trial,  // trial_elem_dofs * ne * vdim x vdim * trial_ndofs
                        mfem::ElementDofOrdering           elem_order);

  /**
   * @brief Updates SparseMatrix entries based on new element assembled matrices
   * @param[in] ea_data Element-assembled data
   */
  virtual void FillData(const mfem::Vector& ea_data);

  /**
   * @brief Returns the necessary size of element assembled data
   * @return Size of ea_map
   */
  auto GetElementDataSize() { return ea_map_.Size(); }

  /**
   * @brief Assembles a new HypreParMatrix
   * @returns a new HypreParMatrix
   */
  auto ParallelAssemble()
  {
    auto hypre_A =
        std::make_unique<mfem::HypreParMatrix>(trial_fes_.GetComm(), test_fes_.GlobalVSize(), trial_fes_.GlobalVSize(),
                                               test_fes_.GetDofOffsets(), trial_fes_.GetDofOffsets(), this);

    return RAP(test_fes_.Dof_TrueDof_Matrix(), hypre_A.release(), trial_fes_.Dof_TrueDof_Matrix());
  }

protected:
  /// Test space describing the sparsity pattern
  const mfem::ParFiniteElementSpace& test_fes_;

  /// Trial space describing the sparsity pattern
  const mfem::ParFiniteElementSpace& trial_fes_;

  /// Test space element restriction
  mfem::ElementRestriction test_restriction_;

  /// Trial space element restriction
  mfem::ElementRestriction trial_restriction_;

  /// Consistent element ordering
  mfem::ElementDofOrdering elem_ordering_;

  /// Maps individual element matrix entries in the K_e vector to the final CSR data offset
  mfem::Array<int> ea_map_;

private:
  /// Computes the row offsets for the CSR sparsity pattern given by the spaces test(trial)
  int FillI();
  /// Computes the column indices per row for a CSR sparsity pattern given by the spaces test(trial)
  void FillJ();
};
}  // namespace mfem_ext
}  // namespace serac
