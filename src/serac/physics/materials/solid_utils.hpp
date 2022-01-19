// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include "mfem.hpp"

namespace serac::solid_util {

/**
 * @brief Calculate the deformation gradient from the displacement gradient (F = H + I)
 *
 * @param[in] du_dX the displacement gradient (du_dX)
 * @param[out] F the deformation gradient (dx_dX)
 */
void calcDeformationGradient(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& F);

/**
 * @brief Calculate the linearized strain tensor (epsilon = 1/2 * (du_dX + du_dX^T))
 *
 * @param[in] du_dX the displacement gradient (du_dX)
 * @param[out] epsilon the linearized strain tensor epsilon = 1/2 * (du_dX + du_dX^T)
 */
void calcLinearizedStrain(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& epsilon);

/**
 * @brief Calculate the Cauchy stress from the PK1 stress
 *
 * @param[in] F the deformation gradient dx_dX
 * @param[in] P the first Piola-Kirchoff stress tensor
 * @param[out] sigma the Cauchy stress tensor
 */
void calcCauchyStressFromPK1Stress(const mfem::DenseMatrix& F, const mfem::DenseMatrix& P, mfem::DenseMatrix& sigma);
}  // namespace serac::solid_util
