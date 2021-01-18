// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "voigt_tensor.hpp"

namespace serac {

void getVoigtVectorFromTensor(const std::vector<std::pair<int, int>>& shear_terms, const mfem::DenseMatrix& mat,
                              mfem::Vector& vec)
{
  int dim = mat.Width();
  vec.SetSize(dim + shear_terms.size());
  for (int i = 0; i < dim; ++i) {
    vec(i) = mat(i, i);
  }
  for (int i = 0; i < (int)shear_terms.size(); ++i) {
    vec(dim + i) = mat(shear_terms[i].first, shear_terms[i].second);
  }
}

void getTensorFromVoigtVector(const std::vector<std::pair<int, int>>& shear_terms, const mfem::Vector& vec,
                              mfem::DenseMatrix& mat)
{
  int dim;

  if (shear_terms.size() == 3) {
    dim = 3;
  } else {
    dim = 2;
  }

  mat.SetSize(dim);
  mat = 0.0;

  for (int i = 0; i < dim; ++i) {
    mat(i, i) = vec(i);
  }

  for (int i = 0; i < (int)shear_terms.size(); ++i) {
    mat(shear_terms[i].first, shear_terms[i].second) = vec(i);
    mat(shear_terms[i].second, shear_terms[i].first) = vec(i);
  }
}

}  // namespace serac
