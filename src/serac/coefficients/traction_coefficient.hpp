// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file traction_coefficient.hpp
 *
 * @brief MFEM coefficients for handling traction boundaries
 */

#ifndef TRACTION_COEF
#define TRACTION_COEF

#include "mfem.hpp"

namespace serac::mfem_ext {

/**
 * @brief A vector coefficient with a mutable scalar scaling factor
 *
 */
class VectorScaledConstantCoefficient : public mfem::VectorCoefficient {
public:
  /**
   * @brief Construct a new Vector Scaled Constant Coefficient object
   *
   * @param[in] v The vector to be scaled
   */
  VectorScaledConstantCoefficient(const mfem::Vector& v) : mfem::VectorCoefficient(v.Size()), vec_(v) {}

  /**
   * @brief Set the Scale object
   *
   * @param[in] s The new scale parameter
   */
  void SetScale(double s) { scale_ = s; }

  /**
   * @brief The vector coefficient evaluation function
   *
   * @param[out] V The output scaled coefficient
   */
  virtual void Eval(mfem::Vector& V, mfem::ElementTransformation&, const mfem::IntegrationPoint&)
  {
    V = vec_;
    V *= scale_;
  }

private:
  /**
   * @brief The vector to be scaled
   */
  mfem::Vector vec_;

  /**
   * @brief The mutable scaling factor
   */
  double scale_ = 1.0;
};

}  // namespace serac::mfem_ext

#endif
