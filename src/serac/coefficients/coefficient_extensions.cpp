// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "serac/coefficients/coefficient_extensions.hpp"

#include "serac/infrastructure/logger.hpp"

namespace serac::mfem_ext {

TransformedVectorCoefficient::TransformedVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient>                v1,
                                                           std::function<void(const mfem::Vector&, mfem::Vector&)> func)
    : mfem::VectorCoefficient(v1->GetVDim()), v1_(v1), v2_(nullptr), mono_function_(func), bi_function_(nullptr)
{
}

TransformedVectorCoefficient::TransformedVectorCoefficient(
    std::shared_ptr<mfem::VectorCoefficient> v1, std::shared_ptr<mfem::VectorCoefficient> v2,
    std::function<void(const mfem::Vector&, const mfem::Vector&, mfem::Vector&)> func)
    : mfem::VectorCoefficient(v1->GetVDim()), v1_(v1), v2_(v2), mono_function_(nullptr), bi_function_(func)
{
  SLIC_CHECK_MSG(v1_->GetVDim() == v2_->GetVDim(), "v1 and v2 are not the same size");
}

void TransformedVectorCoefficient::Eval(mfem::Vector& V, mfem::ElementTransformation& T,
                                        const mfem::IntegrationPoint& ip)
{
  V.SetSize(v1_->GetVDim());
  mfem::Vector temp(v1_->GetVDim());
  v1_->Eval(temp, T, ip);

  if (mono_function_) {
    mono_function_(temp, V);
  } else {
    mfem::Vector temp2(v1_->GetVDim());
    v2_->Eval(temp2, T, ip);
    bi_function_(temp, temp2, V);
  }
}

TransformedScalarCoefficient::TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient>  s1,
                                                           std::function<double(const double)> func)
    : mfem::Coefficient(), s1_(s1), s2_(nullptr), mono_function_(func), bi_function_(nullptr)
{
}

TransformedScalarCoefficient::TransformedScalarCoefficient(std::shared_ptr<mfem::Coefficient>                s1,
                                                           std::shared_ptr<mfem::Coefficient>                s2,
                                                           std::function<double(const double, const double)> func)
    : mfem::Coefficient(), s1_(s1), s2_(s2), mono_function_(nullptr), bi_function_(func)
{
}

double TransformedScalarCoefficient::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip)
{
  double temp = s1_->Eval(T, ip);

  if (mono_function_) {
    return mono_function_(temp);
  } else {
    double temp2 = s2_->Eval(T, ip);
    return bi_function_(temp, temp2);
  }
}
  
}  // namespace serac::mfem_ext
