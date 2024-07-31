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

#pragma once

#include "field.hpp"
#include "serac/physics/common.hpp"

namespace serac {

class NonlinearSystem // NonlinearResidual
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

  SolidSystem(const std::string& physics_name, const std::string& mesh_tag, const mfem::ParFiniteElementSpace& test_space,
     std::vector<std::string> parameter_names = {})
  : mesh_tag_(mesh_tag)
  , shape_displacement_(Field::create(H1<1,dim>{}, detail::addPrefix(physics_name, "shape_displacement"), mesh_tag))
  {

    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> trial_spaces;
    trial_spaces[0] = &test_space;
    trial_spaces[1] = &test_space;

    SLIC_ERROR_ROOT_IF(
        sizeof...(parameter_space) != parameter_names.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(parameter_space), parameter_names.size()));

    if constexpr (sizeof...(parameter_space) > 0) {
      tuple<parameter_space...> types{};
      for_constexpr<sizeof...(parameter_space)>([&](auto i) {
        parameters_.emplace_back( Field::create(types, detail::addPrefix(physics_name, parameter_names[i])) );
        trial_spaces[i + NUM_STATE_VARS] = &(parameters_[i].space());
      });
    }

    residual_ = std::make_unique<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>>(
        &shape_displacement_.space(), &test_space, trial_spaces);
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
  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename Material>
  struct MaterialStressFunctor {
    /// Constructor for the functor
    MaterialStressFunctor(Material material) : material_(material) {}

    /// Material model
    Material material_;

    /**
     * @brief Material stress response call
     *
     * @tparam X Spatial position type
     * @tparam State state
     * @tparam Displacement displacement
     * @tparam Acceleration acceleration
     * @tparam Params variadic parameters for call
     * @param[in] state state
     * @param[in] displacement displacement
     * @param[in] acceleration acceleration
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename X, typename State, typename Displacement, typename Acceleration, typename... Params>
    auto SERAC_HOST_DEVICE operator()(double, X, State& state, Displacement displacement, Acceleration acceleration,
                                      Params... params) const
    {
      static constexpr auto I     = Identity<dim>();

      auto du_dX   = get<DERIVATIVE>(displacement);
      auto d2u_dt2 = get<VALUE>(acceleration);

      auto stress = material_(state, du_dX, params...);

      auto dx_dX = du_dX + I;

      auto flux = dot(stress, transpose(inv(dx_dX))) * det(dx_dX);

      return serac::tuple{material_.density * d2u_dt2, flux};
    }
  };

  using trial = H1<order, dim>;
  using test = H1<order, dim>;
  using shape_trial = H1<SHAPE_ORDER, dim>;

  std::string mesh_tag_;

  Field shape_displacement_;

  /// @brief A vector of the parameters associated with this physics module
  std::vector<Field> parameters_;


  std::unique_ptr<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>> residual_;
};


}