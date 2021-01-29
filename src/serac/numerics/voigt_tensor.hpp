// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file voigt_tensor.hpp
 *
 * @brief Utilities for convecting from tensor to Voigt notation
 *
 */

#ifndef VOIGT_TENSOR
#define VOIGT_TENSOR

#include "mfem.hpp"

/**
 * The Serac namespace
 */
namespace serac {

/**
 * @brief Get the shear terms pairs for Voigt tensor notation based on the dimension
 *
 * @param[in] dim Dimension of the problem
 * @param[out] shear_terms Vector of pairs of indicies for the shear terms Voigt tensor notation
 */
inline void getShearTerms(const int dim, std::vector<std::pair<int, int>>& shear_terms)
{
  if (shear_terms.size() == 0) {
    if (dim == 2) {
      shear_terms = {{0, 1}};
    } else {
      shear_terms = {{1, 2}, {0, 2}, {0, 1}};
    }
  }
}

/**
 * @brief Return a second order tensor based on a Voigt notation vector
 *
 * @param[in] shear_terms Vector of pairs of indicies for the shear terms Voigt tensor notation
 * @param[in] vec The input Voigt vector
 * @param[out] mat The output second order tensor (dense matrix)
 */
void getTensorFromVoigtVector(const std::vector<std::pair<int, int>>& shear_terms, const mfem::Vector& vec,
                              mfem::DenseMatrix& mat);

/**
 * @brief Return a Voigt vector based on a second order tensor
 *
 * @param[in] shear_terms Vector of pairs of indicies for the shear terms Voigt tensor notation
 * @param[in] mat The input second order tensor (dense matrix)
 * @param[out] vec The output Voigt notation vector
 */
void getVoigtVectorFromTensor(const std::vector<std::pair<int, int>>& shear_terms, const mfem::DenseMatrix& mat,
                              mfem::Vector& vec);

}  // namespace serac

#endif
