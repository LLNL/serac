// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "coefficients/coefficient_extensions.hpp"

#include "infrastructure/logger.hpp"

namespace serac {

mfem::Array<int> makeTrueEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c)
{
  mfem::Array<int> ess_tdof_list(0);

  mfem::Array<int> ess_vdof_list = makeEssList(pfes, c);

  for (int i = 0; i < ess_vdof_list.Size(); ++i) {
    int tdof = pfes.GetLocalTDofNumber(ess_vdof_list[i]);
    if (tdof >= 0) {
      ess_tdof_list.Append(tdof);
    }
  }

  return ess_tdof_list;
}

mfem::Array<int> makeEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c)
{
  mfem::Array<int> ess_vdof_list(0);

  mfem::ParGridFunction v_attr(&pfes);
  v_attr.ProjectCoefficient(c);

  for (int vdof = 0; vdof < pfes.GetVSize(); ++vdof) {
    if (v_attr[vdof] > 0.) {
      ess_vdof_list.Append(vdof);
    }
  }

  return ess_vdof_list;
}

mfem::Array<int> makeAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize)
{
  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);
  mfem::Array<int>         attr_list(fes.GetNE());

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (int e = 0; e < fes.GetNE(); e++) {
    attr_list[e] = digitize(elem_attr[e]);
  }

  return attr_list;
}

// Need to use H1_fec because boundary elements don't exist in L2
mfem::Array<int> makeBdrAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize)
{
  mfem::H1_FECollection    h1_fec(1, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &h1_fec);
  mfem::Array<int>         attr_list(fes.GetNBE());
  mfem::Vector             elem_attr(fes.GetNBE());

  for (int e = 0; e < fes.GetNBE(); e++) {
    mfem::Vector dofs(fes.GetBE(e)->GetDof());
    fes.GetBE(e)->Project(c, *fes.GetBdrElementTransformation(e), dofs);
    elem_attr[e] = dofs.Sum() / (dofs.Size() * 1.);
    attr_list[e] = digitize(elem_attr[e]);
  }

  return attr_list;
}

double AttributeModifierCoefficient::Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip)
{
  // Store old attribute and change to new attribute
  double attr  = Tr.Attribute;
  Tr.Attribute = attr_list_[Tr.ElementNo];

  // Evaluate with new attribute
  double result = coef_.Eval(Tr, ip);

  // Set back to original attribute (maybe it's not necessary?.. just to be
  // safe)
  Tr.Attribute = attr;

  return result;
}

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

}  // namespace serac
