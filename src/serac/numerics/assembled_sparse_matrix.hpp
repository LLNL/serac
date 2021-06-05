#pragma once

#include <mfem.hpp>

namespace serac {
namespace mfem_ext {

    /**
     Creates a CSR sparse matrix from element matrices assembled usign a mfem::ElementDofOrdering
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
  const mfem::FiniteElementSpace& test_fes;
  const mfem::FiniteElementSpace& trial_fes;
  // class local ElementRestriction objects
  mfem::ElementRestriction        test_restriction;
  mfem::ElementRestriction        trial_restriction;
  mfem::ElementDofOrdering        elem_ordering;
  mfem::Array<int>                ea_map;
  
private:
  int  FillI();
  void FillJ();

};
} // namespace mfem_ext
} // namespace serac
