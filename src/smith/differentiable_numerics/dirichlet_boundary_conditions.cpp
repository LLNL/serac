// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/physics/mesh.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/boundary_conditions/boundary_condition.hpp"
#include "smith/physics/state/finite_element_vector.hpp"

#include <algorithm>
#include <cmath>

namespace smith {

namespace {

constexpr double bc_time_fd_step = 1.0e-4;

class SecondTimeDerivativeScalarCoefficient : public mfem::Coefficient {
 public:
  explicit SecondTimeDerivativeScalarCoefficient(std::shared_ptr<mfem::Coefficient> source) : source_(std::move(source))
  {
  }

  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    const double t0 = GetTime();
    const double h = bc_time_fd_step * std::max(1.0, std::abs(t0));
    source_->SetTime(t0);
    const double f0 = source_->Eval(T, ip);
    source_->SetTime(t0 + h);
    const double f1 = source_->Eval(T, ip);
    source_->SetTime(t0 + 2.0 * h);
    const double f2 = source_->Eval(T, ip);
    source_->SetTime(t0);
    return (f2 - 2.0 * f1 + f0) / (h * h);
  }

 private:
  std::shared_ptr<mfem::Coefficient> source_;
};

class SecondTimeDerivativeVectorCoefficient : public mfem::VectorCoefficient {
 public:
  explicit SecondTimeDerivativeVectorCoefficient(std::shared_ptr<mfem::VectorCoefficient> source)
      : mfem::VectorCoefficient(source->GetVDim()), source_(std::move(source))
  {
  }

  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override
  {
    const double t0 = GetTime();
    const double h = bc_time_fd_step * std::max(1.0, std::abs(t0));
    mfem::Vector v0(vdim), v1(vdim), v2(vdim);
    source_->SetTime(t0);
    source_->Eval(v0, T, ip);
    source_->SetTime(t0 + h);
    source_->Eval(v1, T, ip);
    source_->SetTime(t0 + 2.0 * h);
    source_->Eval(v2, T, ip);
    source_->SetTime(t0);
    V = v2;
    V.Add(-2.0, v1);
    V += v0;
    V /= (h * h);
  }

 private:
  std::shared_ptr<mfem::VectorCoefficient> source_;
};

}  // namespace

DirichletBoundaryConditions::DirichletBoundaryConditions(const mfem::ParMesh& mfem_mesh,
                                                         mfem::ParFiniteElementSpace& space)
    : mfem_mesh_(mfem_mesh), bcs_(mfem_mesh), space_(space)
{
}

DirichletBoundaryConditions::DirichletBoundaryConditions(const Mesh& mesh, mfem::ParFiniteElementSpace& space)
    : DirichletBoundaryConditions(mesh.mfemParMesh(), space)
{
}

void DirichletBoundaryConditions::setCoefficientValueSource()
{
  SLIC_ERROR_IF(value_source_ == ValueSource::Field,
                "Coefficient-valued and field-valued boundary conditions cannot be mixed for one field");
  value_source_ = ValueSource::Coefficient;
}

void DirichletBoundaryConditions::setFieldValueSource(const FieldState& prescribed_field)
{
  SLIC_ERROR_IF(value_source_ == ValueSource::Coefficient,
                "Coefficient-valued and field-valued boundary conditions cannot be mixed for one field");

  const auto& field = *prescribed_field.get();
  const auto& prescribed_space = field.space();
  SLIC_ERROR_IF(!sameFiniteElementSpace(prescribed_space, space_),
                "Prescribed boundary-condition field and constrained field have incompatible spaces");

  bcs_.setPrescribedFieldName(field.name());
  value_source_ = ValueSource::Field;
}

void DirichletBoundaryConditions::setScalarBCs(const Domain& domain, const FieldState& prescribed_field)
{
  setFieldValueSource(prescribed_field);
  SLIC_ERROR_IF(space_.GetVDim() != 1, "Scalar field boundary conditions require a scalar constrained field");

  auto dof_list = domain.dof_list(&space_);
  space_.DofsToVDofs(0, dof_list);
  auto selector_coefficient = std::make_shared<mfem::ConstantCoefficient>(0.0);
  bcs_.addEssential(dof_list, selector_coefficient, space_, 0);
}

void DirichletBoundaryConditions::setVectorBCs(const Domain& domain, std::vector<int> components,
                                               const FieldState& prescribed_field)
{
  setFieldValueSource(prescribed_field);
  const int field_dim = space_.GetVDim();
  auto scalar_dofs = domain.dof_list(&space_);
  for (int component : components) {
    SLIC_ERROR_IF(component < 0 || component >= field_dim,
                  std::format("Trying to set boundary conditions on a field with dim {}, using component {}", field_dim,
                              component));
    auto vector_dofs = scalar_dofs;
    space_.DofsToVDofs(component, vector_dofs);
    auto selector_coefficient = std::make_shared<mfem::ConstantCoefficient>(0.0);
    bcs_.addEssential(vector_dofs, selector_coefficient, space_, component);
  }
}

void DirichletBoundaryConditions::setVectorBCs(const Domain& domain, const FieldState& prescribed_field)
{
  const int field_dim = space_.GetVDim();
  std::vector<int> components(static_cast<size_t>(field_dim));
  for (int component = 0; component < field_dim; ++component) {
    components[static_cast<size_t>(component)] = component;
  }
  setVectorBCs(domain, std::move(components), prescribed_field);
}

const BoundaryConditionManager& DirichletBoundaryConditions::getSecondDerivativeManager() const
{
  second_derivative_manager_.emplace(mfem_mesh_);
  rebuildSecondDerivativeManager(*second_derivative_manager_);
  return *second_derivative_manager_;
}

void DirichletBoundaryConditions::rebuildSecondDerivativeManager(BoundaryConditionManager& target) const
{
  for (const auto& bc : bcs_.essentials()) {
    if (is_vector_valued(bc.coefficient())) {
      auto deriv_coef = std::make_shared<SecondTimeDerivativeVectorCoefficient>(
          get<std::shared_ptr<mfem::VectorCoefficient>>(bc.coefficient()));
      target.addEssentialByTrueDofs(bc.getTrueDofList(), deriv_coef, space_);
    } else {
      auto deriv_coef = std::make_shared<SecondTimeDerivativeScalarCoefficient>(
          get<std::shared_ptr<mfem::Coefficient>>(bc.coefficient()));
      target.addEssential(bc.getLocalDofList(), deriv_coef, space_, bc.component());
    }
  }
}

}  // namespace smith
