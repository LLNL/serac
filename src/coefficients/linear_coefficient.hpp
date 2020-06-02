// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef LINEAR_COEF
#define LINEAR_COEF

#include <memory>

#include "mfem.hpp"

class LinearTransformationCoefficient : public mfem::Coefficient {
 private:
  std::shared_ptr<mfem::Coefficient> m_coef;
  double                             m_scale;
  double                             m_offset;

 public:
  LinearTransformationCoefficient(const std::shared_ptr<mfem::Coefficient> coef, const double offset,
                                  const double scale)
      : mfem::Coefficient(), m_coef(coef), m_scale(scale), m_offset(offset)
  {
  }
  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
  {
    double val = m_coef->Eval(T, ip);
    return m_scale * val + m_offset;
  };
};

#endif