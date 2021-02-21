// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/physics_utils.hpp"

namespace serac::solid_util {

void calcDeformationGradient(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& F)
{
  int dim = du_dX.Size();
  F.SetSize(dim);
  F = du_dX;

  for (int i = 0; i < dim; ++i) {
    F(i, i) += 1.0;
  }
}

void calcLinearizedStrain(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& epsilon)
{
  epsilon.SetSize(du_dX.Size());
  epsilon = du_dX;
  epsilon.Symmetrize();
}

void calcCauchyStressFromPK1Stress(const mfem::DenseMatrix& F, const mfem::DenseMatrix& P, mfem::DenseMatrix& sigma)
{
  sigma.SetSize(F.Size());
  mfem::MultABt(P, F, sigma);
  sigma *= 1.0 / F.Det();
}

}  // namespace serac::solid_util