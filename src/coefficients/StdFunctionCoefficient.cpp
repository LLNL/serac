// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "StdFunctionCoefficient.hpp"

StdFunctionCoefficient::StdFunctionCoefficient(std::function<double(mfem::Vector &)> func) :
  func_(func)
{ }

double StdFunctionCoefficient::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
  double x[T.GetSpaceDim()];
  Vector transip(x, T.GetSpaceDim());

  T.Transform(ip, transip);
  return func_(transip);
}

StdFunctionVectorCoefficient::StdFunctionVectorCoefficient(int dim,
    std::function<void(mfem::Vector &, mfem::Vector &)> func) :
  VectorCoefficient(dim),
  func_(func)
{ }

void StdFunctionVectorCoefficient::Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
{
  double x[T.GetSpaceDim()];
  Vector transip(x, T.GetSpaceDim());

  T.Transform(ip, transip);
  func_(transip, V);
}

void MakeEssList(Mesh &m, VectorCoefficient &c, Array<int> & ess_vdof_list)
{

  H1_FECollection h1_fec(1, m.SpaceDimension());
  FiniteElementSpace fes(&m, &h1_fec, m.SpaceDimension());

  GridFunction v_attr(&fes);
  v_attr.ProjectCoefficient(c);

  ess_vdof_list.SetSize(0);

  for (int v = 0; v < fes.GetNV(); v++)
    for (int vd = 0; vd < fes.GetVDim(); vd++) {
      if (v_attr[fes.DofToVDof(v,vd)] > 0.) {
        ess_vdof_list.Append(fes.DofToVDof(v,vd));
      }
    }

}

void MakeAttributeList(Mesh &m, Array<int> &attr_list,
                       Coefficient &c, std::function<int(double)> digitize)
{

  L2_FECollection l2_fec(0, m.SpaceDimension());
  FiniteElementSpace fes(&m, &l2_fec);

  attr_list.SetSize(fes.GetNE());

  GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (int e = 0; e < fes.GetNE(); e++) {
    attr_list[e] = digitize(elem_attr[e]);
  }
}

// Need to use H1_fec because boundary elements don't exist in L2
void MakeBdrAttributeList(Mesh &m, Array<int> &attr_list,
                          Coefficient &c, std::function<int(double)> digitize)
{

  H1_FECollection h1_fec(1, m.SpaceDimension());
  FiniteElementSpace fes(&m, &h1_fec);

  attr_list.SetSize(fes.GetNBE());
  Vector elem_attr(fes.GetNBE());

  for (int e = 0; e < fes.GetNBE(); e++) {
    Vector dofs(fes.GetBE(e)->GetDof());
    fes.GetBE(e)->Project(c, *fes.GetBdrElementTransformation(e), dofs);
    elem_attr[e] = dofs.Sum() / (dofs.Size()*1.);
    attr_list[e] = digitize(elem_attr[e]);
  }
}


double AttributeModifierCoefficient::Eval (ElementTransformation &Tr, const IntegrationPoint &ip)
{
  // Store old attribute and change to new attribute
  double attr = Tr.Attribute;
  Tr.Attribute = attr_list_[Tr.ElementNo];

  // Evaluate with new attribute
  double result = C_->Eval(Tr, ip);

  // Set back to original attribute (maybe it's not necessary?.. just to be safe)
  Tr.Attribute = attr;

  return result;
}

TransformedVectorCoefficient::TransformedVectorCoefficient(VectorCoefficient *v1,
    std::function <void (Vector &, Vector &)>  func) :
  VectorCoefficient(v1->GetVDim()),
  mono_function_(func),
  bi_function_(NULL),
  v1_(v1),
  v2_(NULL)
{  }

TransformedVectorCoefficient::TransformedVectorCoefficient(VectorCoefficient *v1, VectorCoefficient *v2,
    std::function <void (Vector &, Vector &, Vector &)>  func) :
  VectorCoefficient(v1->GetVDim()),
  mono_function_(NULL),
  bi_function_(func),
  v1_(v1),
  v2_(v2)
{
  MFEM_VERIFY(v1_->GetVDim() == v2_->GetVDim(), "v1 and v2 are not the same size");
}

void TransformedVectorCoefficient::Eval (Vector & V, ElementTransformation & T, const IntegrationPoint & ip )
{
  V.SetSize(v1_->GetVDim());
  Vector temp (v1_->GetVDim());
  v1_->Eval(temp, T, ip);

  if (mono_function_) {
    mono_function_(temp, V);
  } else {
    Vector temp2 (v1_->GetVDim());
    v2_->Eval(temp2, T, ip);
    bi_function_(temp, temp2, V);
  }
}

