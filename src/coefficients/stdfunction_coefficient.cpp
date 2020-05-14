// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "stdfunction_coefficient.hpp"

StdFunctionCoefficient::StdFunctionCoefficient(std::function<double(mfem::Vector &)> func) : m_func(func) {}

double StdFunctionCoefficient::Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
{
  mfem::Vector transip(T.GetSpaceDim());
  T.Transform(ip, transip);
  return m_func(transip);
}

StdFunctionVectorCoefficient::StdFunctionVectorCoefficient(int                                                 dim,
                                                           std::function<void(mfem::Vector &, mfem::Vector &)> func)
    : mfem::VectorCoefficient(dim), m_func(func)
{
}

void StdFunctionVectorCoefficient::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                                        const mfem::IntegrationPoint &ip)
{
  mfem::Vector transip(T.GetSpaceDim());
  T.Transform(ip, transip);
  m_func(transip, V);
}

void MakeTrueEssList(mfem::ParFiniteElementSpace &pfes, mfem::VectorCoefficient &c, mfem::Array<int> &ess_tdof_list)
{
  mfem::ParGridFunction v_attr(&pfes);
  v_attr.ProjectCoefficient(c);

  ess_tdof_list.SetSize(0);

  for (int vdof = 0; vdof < pfes.GetVSize(); ++vdof) {
    int tdof = pfes.GetLocalTDofNumber(vdof);
    if (v_attr[vdof] > 0. && tdof >= 0) {
      ess_tdof_list.Append(tdof);
    }
  }
}

void MakeEssList(mfem::Mesh &m, mfem::VectorCoefficient &c, mfem::Array<int> &ess_vdof_list)
{
  mfem::H1_FECollection    h1_fec(1, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &h1_fec, m.SpaceDimension());

  mfem::GridFunction v_attr(&fes);
  v_attr.ProjectCoefficient(c);

  ess_vdof_list.SetSize(0);

  for (int v = 0; v < fes.GetNV(); v++)
    for (int vd = 0; vd < fes.GetVDim(); vd++) {
      if (v_attr[fes.DofToVDof(v, vd)] > 0.) {
        ess_vdof_list.Append(fes.DofToVDof(v, vd));
      }
    }
}

void MakeAttributeList(mfem::Mesh &m, mfem::Array<int> &attr_list, mfem::Coefficient &c,
                       std::function<int(double)> digitize)
{
  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);

  attr_list.SetSize(fes.GetNE());

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (int e = 0; e < fes.GetNE(); e++) {
    attr_list[e] = digitize(elem_attr[e]);
  }
}

// Need to use H1_fec because boundary elements don't exist in L2
void MakeBdrAttributeList(mfem::Mesh &m, mfem::Array<int> &attr_list, mfem::Coefficient &c,
                          std::function<int(double)> digitize)
{
  mfem::H1_FECollection    h1_fec(1, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &h1_fec);

  attr_list.SetSize(fes.GetNBE());
  mfem::Vector elem_attr(fes.GetNBE());

  for (int e = 0; e < fes.GetNBE(); e++) {
    mfem::Vector dofs(fes.GetBE(e)->GetDof());
    fes.GetBE(e)->Project(c, *fes.GetBdrElementTransformation(e), dofs);
    elem_attr[e] = dofs.Sum() / (dofs.Size() * 1.);
    attr_list[e] = digitize(elem_attr[e]);
  }
}

double AttributeModifierCoefficient::Eval(mfem::ElementTransformation &Tr, const mfem::IntegrationPoint &ip)
{
  // Store old attribute and change to new attribute
  double attr  = Tr.Attribute;
  Tr.Attribute = m_attr_list[Tr.ElementNo];

  // Evaluate with new attribute
  double result = m_C->Eval(Tr, ip);

  // Set back to original attribute (maybe it's not necessary?.. just to be
  // safe)
  Tr.Attribute = attr;

  return result;
}

TransformedVectorCoefficient::TransformedVectorCoefficient(mfem::VectorCoefficient *                           v1,
                                                           std::function<void(mfem::Vector &, mfem::Vector &)> func)
    : mfem::VectorCoefficient(v1->GetVDim()), m_v1(v1), m_v2(nullptr), m_mono_function(func), m_bi_function(nullptr)
{
}

TransformedVectorCoefficient::TransformedVectorCoefficient(
    mfem::VectorCoefficient *v1, mfem::VectorCoefficient *v2,
    std::function<void(mfem::Vector &, mfem::Vector &, mfem::Vector &)> func)
    : mfem::VectorCoefficient(v1->GetVDim()), m_v1(v1), m_v2(v2), m_mono_function(nullptr), m_bi_function(func)
{
  MFEM_VERIFY(m_v1->GetVDim() == m_v2->GetVDim(), "v1 and v2 are not the same size");
}

void TransformedVectorCoefficient::Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                                        const mfem::IntegrationPoint &ip)
{
  V.SetSize(m_v1->GetVDim());
  mfem::Vector temp(m_v1->GetVDim());
  m_v1->Eval(temp, T, ip);

  if (m_mono_function) {
    m_mono_function(temp, V);
  } else {
    mfem::Vector temp2(m_v1->GetVDim());
    m_v2->Eval(temp2, T, ip);
    m_bi_function(temp, temp2, V);
  }
}
