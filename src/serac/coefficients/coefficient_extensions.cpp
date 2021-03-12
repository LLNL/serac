// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
// # Author: Jonathan Wong @ LLNL.

#include "serac/coefficients/coefficient_extensions.hpp"

#include "serac/infrastructure/logger.hpp"

namespace serac::mfem_ext {

mfem::Array<int> MakeTrueEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c)
{
  mfem::Array<int> ess_tdof_list(0);

  mfem::Array<int> ess_vdof_list = MakeEssList(pfes, c);

  for (int i = 0; i < ess_vdof_list.Size(); ++i) {
    int tdof = pfes.GetLocalTDofNumber(ess_vdof_list[i]);
    if (tdof >= 0) {
      ess_tdof_list.Append(tdof);
    }
  }

  return ess_tdof_list;
}

mfem::Array<int> MakeEssList(mfem::ParFiniteElementSpace& pfes, mfem::VectorCoefficient& c)
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

  template <>
  mfem::Array<int> MakeAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize )
{
  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);
  mfem::Array<int> attr_list(fes.GetNE());

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (int e = 0; e < fes.GetNE(); e++) {
    attr_list[e] = digitize(elem_attr[e]);
  }

  return attr_list;
}

    template <>
  std::vector<int> MakeAttributeList(mfem::Mesh& m, mfem::Coefficient& c, std::function<int(double)> digitize )
{
  mfem::L2_FECollection    l2_fec(0, m.SpaceDimension());
  mfem::FiniteElementSpace fes(&m, &l2_fec);
  std::vector<int> attr_list(static_cast<typename std::vector<int>::size_type>(fes.GetNE()));

  mfem::GridFunction elem_attr(&fes);
  elem_attr.ProjectCoefficient(c);

  for (int e = 0; e < fes.GetNE(); e++) {
    attr_list[static_cast<typename std::vector<int>::size_type>(e)] = digitize(elem_attr[e]);
  }

  return attr_list;
}

  
  void AssignMeshElementAttributes(mfem::Mesh &m, std::variant<mfem::Array<int>, std::vector<int>> && list) {
    // check to make sure the lists are match the number of elements
    if (auto arr = std::get_if<mfem::Array<int>>(&list)) {
      SLIC_ERROR_IF(arr->Size() != m.GetNE(), "list size does not match the number of mesh elements");
    } else if (auto vec = std::get_if<std::vector<int>>(&list)) {
      SLIC_ERROR_IF(static_cast<int>(vec->size()) != m.GetNE(), "list size does not match the number of mesh elements");
    }
    
    for (int e = 0; e < m.GetNE(); e++ ) {
      m.GetElement(e)->SetAttribute(std::visit([=](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, mfem::Array<int>>)
			   return arg[e];
            else 
	      return arg[static_cast<std::vector<int>::size_type>(e)];
	  }, list));
    }
    m.SetAttributes();
  }
  
  void AssignMeshBdrAttributes(mfem::Mesh &m, std::variant<mfem::Array<int>, std::vector<int>> & list) {
    // check to make sure the lists are match the number of elements
    if (auto arr = std::get_if<mfem::Array<int>>(&list)) {
      SLIC_ERROR_IF(arr->Size() != m.GetNBE(), "list size does not match the number of mesh elements");
    } else if (auto vec = std::get_if<std::vector<int>>(&list)) {
      SLIC_ERROR_IF(static_cast<int>(vec->size()) != m.GetNBE(), "list size does not match the number of mesh elements");
    }
    
    for (int e = 0; e < m.GetNBE(); e++ ) {
      m.GetBdrElement(e)->SetAttribute(std::visit([=](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, mfem::Array<int>>)
			   return arg[e];
            else 
	      return arg[static_cast<std::vector<int>::size_type>(e)];
	  }, list));
    }
    m.SetAttributes();
  }

  
double AttributeModifierCoefficient::Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip)
{
  // Store old attribute and change to new attribute
  const int attr = Tr.Attribute;
  Tr.Attribute   = attr_list_[Tr.ElementNo];

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

  namespace digitize {
    int floor(double v) {
      return static_cast<int>(v);
    }
  }
  
}  // namespace serac::mfem_ext
