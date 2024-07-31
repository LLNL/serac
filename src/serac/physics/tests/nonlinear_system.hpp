// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_system.hpp
 *
 * @brief Specifies parametrized residuals and various linearized evaluations for arbitrary nonlinear systems of equations
 */

#include "serac/physics/common.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

struct Field {
  public:
  FiniteElementState& get() {
    return *field;
  }
  auto space() const { return field->space(); }

  FiniteElementState* field;
  FiniteElementDual* dual;
};

struct FieldDual {
  FiniteElementDual& get() {
    return *dual;
  }
  std::shared_ptr<FiniteElementDual> dual;
  std::shared_ptr<FiniteElementState> dualOfDual;
};


class NonlinearSystem 
{
  // computes residual outputs
  virtual void residual(const std::vector<Field> fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) = 0;

  // seems like this might need to be a matrix of HypreParMatrix
  virtual std::shared_ptr<mfem::HypreParMatrix> fieldJacobian(const std::vector<Field>& fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) = 0;
  virtual std::shared_ptr<mfem::HypreParMatrix> parameterJacobian(const std::vector<Field>& fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) = 0;

  // computes for each residual output: dr/du * fieldsV + dr/dp * parametersV
  virtual void jvp(const std::vector<Field> fields, const std::vector<Field>& parameters, 
                   const std::vector<Field> fieldsV, const std::vector<Field>& parametersV,
                   std::vector<FieldDual>& jacobianVectorProducts) = 0;

  // computes for each input field  (dr/du).T * vResidual
  // computes for each input parameter (dr/dp).T * vResidual
  // can early out if the vectors being requested are sized to 0?
  virtual void vjp(const std::vector<Field> fields, const std::vector<Field>& parameters, 
                   const std::vector<FieldDual> vResiduals,
                   std::vector<FieldDual>& fieldSensitivities,
                   std::vector<FieldDual>& parameterSensitivities) = 0;
};


template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidSystem;

template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidSystem<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>  : public NonlinearSystem
{
  public:

  static constexpr auto NUM_STATE_VARS = 2; // we want to change this to 3 to include velocity

  SolidSystem(const std::string& physics_name, const std::string& mesh_tag) 
  : mesh_tag_(mesh_tag)
  , mesh_(StateManager::mesh(mesh_tag_))
  , shape_displacement_{.field=&StateManager::shapeDisplacement(mesh_tag), .dual=nullptr}
  {
    //auto newT = StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "acceleration"), mesh_tag_))
    mfem::ParFiniteElementSpace shape_space = shape_displacement_.space();
    mfem::ParFiniteElementSpace test_space = getSpace(mesh_, test{});
    residual_ = std::make_unique<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>>(
        shape_space, test_space, test_space);
  }

  // computes residual outputs
  void residual(const std::vector<Field> fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) override
  {

  }

  std::shared_ptr<mfem::HypreParMatrix> fieldJacobian(const std::vector<Field>& fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) override
  {
    return nullptr;
  }

  std::shared_ptr<mfem::HypreParMatrix> parameterJacobian(const std::vector<Field>& fields, const std::vector<Field>& parameters, std::vector<FieldDual>& residuals) override
  {
    return nullptr;
  }

  // computes for each residual output: dr/du * fieldsV + dr/dp * parametersV
  void jvp(const std::vector<Field> fields, const std::vector<Field>& parameters, 
           const std::vector<Field> fieldsV, const std::vector<Field>& parametersV,
           std::vector<FieldDual>& jacobianVectorProducts) override
  {
    return;
  }

  // computes for each input field  (dr/du).T * vResidual
  // computes for each input parameter (dr/dp).T * vResidual
  // can early out if the vectors being requested are sized to 0?
  void vjp(const std::vector<Field> fields, const std::vector<Field>& parameters, 
           const std::vector<FieldDual> vResiduals,
           std::vector<FieldDual>& fieldSensitivities,
           std::vector<FieldDual>& parameterSensitivities) override
  {
    return;
  }

  private:

  using trial = H1<order, dim>;
  using test = H1<order, dim>;
  using shape_trial = H1<SHAPE_ORDER, dim>;

  std::string mesh_tag_;
  mfem::ParMesh& mesh_;

  Field shape_displacement_;

  std::vector<std::unique_ptr<mfem::FECollection>> collections; // seems like I need to hold on to these for no good reason

  std::unique_ptr<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>> residual_;
};


}