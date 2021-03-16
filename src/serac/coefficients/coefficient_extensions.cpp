// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "serac/coefficients/coefficient_extensions.hpp"

#include "serac/infrastructure/logger.hpp"

namespace serac::mfem_ext {

  namespace detail {

    template <>
    typename eval_t<mfem::Coefficient>::type
    eval<mfem::Coefficient>(mfem::Coefficient & c, mfem::ElementTransformation &Tr, const mfem::IntegrationPoint & ip) {
      return c.Eval(Tr, ip);
    }

    template <>
    eval_t<mfem::VectorCoefficient>::type
    eval<mfem::VectorCoefficient> (mfem::VectorCoefficient &v, mfem::ElementTransformation &Tr, const mfem::IntegrationPoint & ip) {
      mfem::Vector temp(v.GetVDim());
      v.Eval(temp, Tr, ip);
      return temp;
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
