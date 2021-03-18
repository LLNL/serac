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
typename eval_t<mfem::Coefficient>::type eval<mfem::Coefficient>(mfem::Coefficient& c, mfem::ElementTransformation& Tr,
                                                                 const mfem::IntegrationPoint& ip)
{
  return c.Eval(Tr, ip);
}

template <>
eval_t<mfem::VectorCoefficient>::type eval<mfem::VectorCoefficient>(mfem::VectorCoefficient&      v,
                                                                    mfem::ElementTransformation&  Tr,
                                                                    const mfem::IntegrationPoint& ip)
{
  mfem::Vector temp(v.GetVDim());
  v.Eval(temp, Tr, ip);
  return temp;
}

}  // namespace detail

}  // namespace serac::mfem_ext
