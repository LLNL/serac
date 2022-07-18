// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_functional.hpp
 *
 * @brief An object containing the solver for total Lagrangian finite deformation solid mechanics
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/solid.hpp"
#include "serac/physics/materials/functional_material_utils.hpp"

namespace serac {

namespace solid_util {
/// A timestep and boundary condition enforcement method for a dynamic solver
struct TimesteppingOptions {
  /// The timestepping method to be applied
  TimestepMethod timestepper;

  /// The essential boundary enforcement method to use
  DirichletEnforcementMethod enforcement_method;
};

/**
 * @brief A configuration variant for the various solves
 * For quasistatic solves, leave the @a dyn_options parameter null. @a T_nonlin_options and @a T_lin_options
 * define the solver parameters for the nonlinear residual and linear stiffness solves. For
 * dynamic problems, @a dyn_options defines the timestepping scheme while @a T_lin_options and @a T_nonlin_options
 * define the nonlinear residual and linear stiffness solve options as before.
 */
struct SolverOptions {
  /// The linear solver options
  LinearSolverOptions H_lin_options;

  /// The nonlinear solver options
  NonlinearSolverOptions H_nonlin_options;

  /**
   * @brief The optional ODE solver parameters
   * @note If this is not defined, a quasi-static solve is performed
   */
  std::optional<TimesteppingOptions> dyn_options = std::nullopt;
};
}  // namespace solid_util

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear total Lagrangian quasi-static and dynamic
 * hyperelastic solver object. This uses Functional to compute the tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... parameter_space>
class SolidFunctional : public BasePhysics {
public:
  /**
   * @brief Construct a new Solid Functional object
   *
   * @param options The options for the linear, nonlinear, and ODE solves
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param keep_deformation Flag to keep the deformation in the underlying mesh post-destruction
   * @param name An optional name for the physics module instance
   * @param parameter_states An array of FiniteElementStates containing the user-specified parameter fields
   */
  SolidFunctional(
      const solid_util::SolverOptions& options, GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On,
      FinalMeshOption keep_deformation = FinalMeshOption::Deformed, const std::string& name = "",
      std::array<std::reference_wrapper<FiniteElementState>, sizeof...(parameter_space)> parameter_states = {})
      : BasePhysics(2, order, name),
        velocity_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "velocity")})),
        displacement_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "displacement")})),
        adjoint_displacement_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "adjoint_displacement")})),
        parameter_states_(parameter_states),
        ode2_(displacement_.space().TrueVSize(), {.c0 = c0_, .c1 = c1_, .u = u_, .du_dt = du_dt_, .d2u_dt2 = previous_},
              nonlin_solver_, bcs_),
        c0_(0.0),
        c1_(0.0),
        geom_nonlin_(geom_nonlin),
        keep_deformation_(keep_deformation)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    state_.push_back(velocity_);
    state_.push_back(displacement_);
    state_.push_back(adjoint_displacement_);

    // Create a pack of the primal field and parameter finite element spaces
    std::array<mfem::ParFiniteElementSpace*, sizeof...(parameter_space) + 1> trial_spaces;
    trial_spaces[0] = &displacement_.space();

    functional_call_args_.emplace_back(displacement_);

    if constexpr (sizeof...(parameter_space) > 0) {
      for (size_t i = 0; i < sizeof...(parameter_space); ++i) {
        trial_spaces[i + 1]         = &(parameter_states_[i].get().space());
        parameter_sensitivities_[i] = std::make_unique<FiniteElementDual>(mesh_, parameter_states_[i].get().space());
        functional_call_args_.emplace_back(parameter_states_[i].get());
      }
    }

    M_functional_ = std::make_unique<Functional<test(trial, parameter_space...)>>(&displacement_.space(), trial_spaces);

    K_functional_ = std::make_unique<Functional<test(trial, parameter_space...)>>(&displacement_.space(), trial_spaces);

    state_.push_back(velocity_);
    state_.push_back(displacement_);

    // Initialize the mesh node pointers
    reference_nodes_ = std::make_unique<mfem::ParGridFunction>(&displacement_.space());
    mesh_.EnsureNodes();
    mesh_.GetNodes(*reference_nodes_);

    deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

    displacement_ = 0.0;
    velocity_     = 0.0;

    const auto& lin_options = options.H_lin_options;
    // If the user wants the AMG preconditioner with a linear solver, set the pfes
    // to be the displacement
    const auto& augmented_options = mfem_ext::AugmentAMGForElasticity(lin_options, displacement_.space());

    nonlin_solver_ = mfem_ext::EquationSolver(mesh_.GetComm(), augmented_options, options.H_nonlin_options);

    // Check for dynamic mode
    if (options.dyn_options) {
      ode2_.SetTimestepper(options.dyn_options->timestepper);
      ode2_.SetEnforcementMethod(options.dyn_options->enforcement_method);
      is_quasistatic_ = false;
    } else {
      is_quasistatic_ = true;
    }

    int true_size = velocity_.space().TrueVSize();

    u_.SetSize(true_size);
    du_dt_.SetSize(true_size);
    previous_.SetSize(true_size);
    previous_ = 0.0;

    zero_.SetSize(true_size);
    zero_ = 0.0;
  }

  /// @brief Destroy the Solid Functional object
  ~SolidFunctional()
  {
    // Update the mesh with the new deformed nodes if requested
    if (keep_deformation_ == FinalMeshOption::Deformed) {
      *reference_nodes_ += displacement_.gridFunction();
    }

    // Build a new grid function to store the mesh nodes post-destruction
    // NOTE: MFEM will manage the memory of these objects

    auto mesh_fe_coll  = new mfem::H1_FECollection(order_, mesh_.Dimension());
    auto mesh_fe_space = new mfem::ParFiniteElementSpace(displacement_.space(), &mesh_, mesh_fe_coll);
    auto mesh_nodes    = new mfem::ParGridFunction(mesh_fe_space);
    mesh_nodes->MakeOwner(mesh_fe_coll);

    *mesh_nodes = *reference_nodes_;

    // Set the mesh to the newly created nodes object and pass ownership
    mesh_.NewNodes(*mesh_nodes, true);
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes on which to enforce a displacement
   * @param[in] disp The prescribed boundary displacement function
   */
  void setDisplacementBCs(const std::set<int>&                                           disp_bdr,
                          std::function<void(const mfem::Vector& x, mfem::Vector& disp)> disp)
  {
    // Project the coefficient onto the grid function
    disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_.addEssential(disp_bdr, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp The vector function containing the set displacement values
   * @param[in] component The component to set the displacment on
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<double(const mfem::Vector& x)> disp,
                          int component)
  {
    // Project the coefficient onto the grid function
    component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(disp);

    bcs_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_.space(), component);
  }

  void setSlideWallBCs(const std::set<int>& slide_bdr)
  {
    SLIC_ERROR_ROOT_IF(slide_bdr.size() > dim,
                       "Number of sliding wall attribute boundaries must be less than the dimension.");

    auto& space = displacement_.space();

    std::unordered_map<int, mfem::Vector> attribute_normals;

    for (int i = 0; i < space.GetNBE(); ++i) {
      int att = space.GetBdrAttribute(i);

      auto found = slide_bdr.find(att);
      if (found != slide_bdr.end()) {
        // compute the boundary element normal
        mfem::ElementTransformation* Tr    = space.GetBdrElementTransformation(i);
        const mfem::FiniteElement*   fe    = space.GetBE(i);
        const mfem::IntegrationRule& nodes = fe->GetNodes();

        mfem::Array<int> dofs;
        space.GetBdrElementDofs(i, dofs);
        SLIC_ERROR_IF(dofs.Size() != nodes.Size(), "Something wrong in finite element space!");

        for (int j = 0; j < dofs.Size(); ++j) {
          Tr->SetIntPoint(&nodes[j]);

          mfem::Vector normal(dim);

          // the normal returned in the next line is scaled by h. We should normalize it to 1
          // to check for consistency along the boundary
          CalcOrtho(Tr->Jacobian(), normal);

          double norm_l2 = normal.Norml2();
          normal /= norm_l2;

          auto found_attr = attribute_normals.find(att);

          if (found_attr == attribute_normals.end()) {
            // if the normal is not already found, add it to the list
            attribute_normals.emplace(att, normal);
          } else {
            // otherwise check that it is consistent with previously found normals
            auto& other_normal = found_attr->second;
            for (int comp = 0; comp < normal.Size(); ++comp) {
              SLIC_ERROR_IF(std::abs(std::abs(other_normal(comp)) - std::abs(normal(comp))) > 1.0e-7,
                            "The normal vectors on an attribute surface are not constant in the sliding wall boundary "
                            "condition.");
            }
          }
        }
      }
    }

    // TODO reduce the normal across MPI ranks

    // TODO implement for more than one sliding wall
    SLIC_ERROR_ROOT_IF(attribute_normals.size() > 1, "Sliding wall not implemented for more than one wall.");

    for (auto& normal : attribute_normals) {
      SLIC_ERROR_ROOT_IF(dim != 2 && dim != 3, "Dimension must be 2 or 3 for sliding wall boundary conditions.");

      if constexpr (dim == 2) {
        double angle = std::acos(normal.second(0));

        coordinate_transform_ = {{{std::cos(angle), std::sin(angle)}, {-1.0 * std::sin(angle), std::cos(angle)}}};

        inv_coordinate_transform_ = transpose(*coordinate_transform_);
      }
      if constexpr (dim == 3) {
        // Use Gram Schmidt orthogonalization to compute a new coordinate set
        tensor<double, dim> normal_1{normal.second(0), normal.second(1), normal.second(2)};

        int min_index = 0;
        for (int i = 1; i < 3; ++i) {
          if (std::abs(normal_1[i]) < std::abs(normal_1[min_index])) {
            min_index = i;
          }
        }

        tensor<double, dim> normal_2{0.0, 0.0, 0.0};
        normal_2[min_index] = 1.0;
        normal_2 -= normal_1[min_index] * normal_1;
        normal_2 = normalize(normal_2);

        // The third normal is the cross product of the first two
        tensor<double, dim> normal_3 = cross(normal_1, normal_2);

        normal_3 = normalize(normal_3);

        // Generate a coordinate transform from the new basis set
        coordinate_transform_ = {{{normal_1[0], normal_2[0], normal_3[0]},
                                  {normal_1[1], normal_2[1], normal_3[1]},
                                  {normal_1[2], normal_2[2], normal_3[2]}}};

        inv_coordinate_transform_ = transpose(*coordinate_transform_);
      }

      // Add the appropriate component-based boundary condition
      // Project the coefficient onto the grid function
      component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>([](const mfem::Vector&) { return 0.0; });

      bcs_.addEssential({normal.first}, component_disp_bdr_coef_, displacement_.space(), 0);
    }
  }

  /// @brief Solve the Quasi-static Newton system
  void quasiStaticSolve() { nonlin_solver_.Mult(zero_, displacement_); }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping
   * schemes
   * @pre SolidFunctional::completeSetup() must be called prior to this call
   */
  void advanceTimestep(double& dt) override
  {
    SLIC_ERROR_ROOT_IF(!residual_, "completeSetup() must be called prior to advanceTimestep(dt) in SolidFunctional.");

    // Set the mesh nodes to the reference configuration
    if (geom_nonlin_ == GeometricNonlinearities::On) {
      mesh_.NewNodes(*reference_nodes_);
    }

    rotate();

    if (is_quasistatic_) {
      // Update the time for housekeeping purposes
      time_ += dt;
      // Project the essential boundary coefficients
      for (auto& bc : bcs_.essentials()) {
        bc.setDofs(displacement_, time_);
      }

      quasiStaticSolve();
    } else {
      // Note that the ODE solver handles the essential boundary condition application itself
      ode2_.Step(displacement_, velocity_, time_, dt);
    }

    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Update the mesh with the new deformed nodes
      deformed_nodes_->Set(1.0, displacement_.gridFunction());
      deformed_nodes_->Add(1.0, *reference_nodes_);

      mesh_.NewNodes(*deformed_nodes_);
    }

    undoRotate();

    cycle_ += 1;
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @param material A material containing density and stress evaluation information
   *
   * @pre MaterialType must have a method density() defining the density
   * @pre MaterialType must have the operator (du_dX) defined as the Kirchoff stress
   */
  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    if constexpr (is_parameterized<MaterialType>::value) {
      static_assert(material.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in solid does not equal the number of parameters in the "
                    "solid material.");
    }

    auto parameterized_material = parameterizeMaterial(material);

    K_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [this, parameterized_material](auto x, auto displacement, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = displacement;

          auto source = zero{};

          auto response = parameterized_material(x, u, du_dX, serac::get<0>(params)...);

          auto flux = response.stress;

          if (geom_nonlin_ == GeometricNonlinearities::On) {
            auto deformation_grad = du_dX + I_;
            flux                  = flux * inv(transpose(deformation_grad));
          }

          return serac::tuple{source, flux};
        },
        mesh_);

    M_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [this, parameterized_material](auto x, auto displacement, auto... params) {
          auto [u, du_dX] = displacement;

          auto response = parameterized_material(x, u, du_dX, serac::get<0>(params)...);

          auto flux = 0.0 * du_dX;

          double geom_factor = (geom_nonlin_ == GeometricNonlinearities::On ? 1.0 : 0.0);

          auto deformation_grad = du_dX + I_;
          auto source           = response.density * u * (1.0 + geom_factor * (det(deformation_grad) - 1.0));

          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed displacement
   *
   * @param disp The function describing the displacement field
   */
  void setDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& disp)> disp)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient disp_coef(dim, disp);
    displacement_.project(disp_coef);
    gf_initialized_[1] = true;
  }

  /**
   * @brief Set the underlying finite element state to a prescribed velocity
   *
   * @param vel The function describing the velocity field
   */
  void setVelocity(std::function<void(const mfem::Vector& x, mfem::Vector& vel)> vel)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient vel_coef(dim, vel);
    velocity_.project(vel_coef);
    gf_initialized_[0] = true;
  }

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @param body_force_function A source function for a prescribed body load
   *
   * @pre BodyForceType must have the operator (x, time, displacement, d displacement_dx) defined as the body force
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force_function)
  {
    if constexpr (is_parameterized<BodyForceType>::value) {
      static_assert(body_force_function.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in solid not equal the number of parameters in the "
                    "body force.");
    }

    auto parameterized_body_force = parameterizeSource(body_force_function);

    K_functional_->AddDomainIntegral(
        Dimension<dim>{},
        [parameterized_body_force, this](auto x, auto displacement, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = displacement;

          auto flux = du_dX * 0.0;

          double geom_factor = (geom_nonlin_ == GeometricNonlinearities::On ? 1.0 : 0.0);

          auto deformation_grad = du_dX + I_;

          auto source = parameterized_body_force(x, time_, u, du_dX, serac::get<0>(params)...) *
                        (1.0 + geom_factor * (det(deformation_grad) - 1.0));
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   * @param compute_on_reference Flag to compute the traction in the reference configuration
   *
   * @pre TractionType must have the operator (x, normal, time) to return the thermal flux value
   */
  template <typename TractionType>
  void setTractionBCs(TractionType traction_function, bool compute_on_reference = true)
  {
    if constexpr (is_parameterized<TractionType>::value) {
      static_assert(traction_function.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in solid does not equal the number of parameters in the "
                    "traction boundary.");
    }

    auto parameterized_traction = parameterizeFlux(traction_function);

    // TODO fix this when we can get gradients from boundary integrals
    SLIC_ERROR_IF(!compute_on_reference, "SolidFunctional cannot compute traction BCs in deformed configuration");

    K_functional_->AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [this, parameterized_traction](auto x, auto n, auto, auto... params) {
          return -1.0 * parameterized_traction(x, n, time_, params...);
        },
        mesh_);
  }

  /**
   * @brief Set the pressure boundary condition
   *
   * @tparam PressureType The type of the pressure load
   * @param pressure_function A function describing the pressure applied to a boundary
   * @param compute_on_reference Flag to compute the pressure in the reference configuration
   *
   * @pre PressureType must have the operator (x, time) to return the thermal flux value
   */
  template <typename PressureType>
  void setPressureBCs(PressureType pressure_function, bool compute_on_reference = true)
  {
    if constexpr (is_parameterized<PressureType>::value) {
      static_assert(pressure_function.numParameters() == sizeof...(parameter_space),
                    "Number of parameters in solid does not equal the number of parameters in the "
                    "pressure boundary.");
    }

    auto parameterized_pressure = parameterizePressure(pressure_function);

    // TODO fix this when we can get gradients from boundary integrals
    SLIC_ERROR_IF(!compute_on_reference, "SolidFunctional cannot compute pressure BCs in deformed configuration");

    K_functional_->AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [this, parameterized_pressure](auto x, auto n, auto, auto... params) {
          return parameterized_pressure(x, time_, params...) * n;
        },
        mesh_);
  }

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const serac::FiniteElementState& displacement() const { return displacement_; };

  /// @overload
  serac::FiniteElementState& displacement() { return displacement_; };

  /**
   * @brief Get the adjoint displacement state
   *
   * @return A reference to the current adjoint displacement finite element state
   */
  const serac::FiniteElementState& adjointDisplacement() const { return adjoint_displacement_; };

  /// @overload
  serac::FiniteElementState& adjointDisplacement() { return adjoint_displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return A reference to the current velocity finite element state
   */
  const serac::FiniteElementState& velocity() const { return velocity_; };

  /// @overload
  serac::FiniteElementState& velocity() { return velocity_; };

  /// @brief Reset the mesh, displacement, and velocity to the reference (stress-free) configuration
  void resetToReferenceConfiguration()
  {
    displacement_ = 0.0;
    velocity_     = 0.0;

    mesh_.NewNodes(*reference_nodes_);
  }

  /// @brief Build the quasi-static operator corresponding to the total Lagrangian formulation
  std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator()
  {
    // the quasistatic case is entirely described by the residual,
    // there is no ordinary differential equation
    auto residual = std::make_unique<mfem_ext::StdFunctionOperator>(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& u, mfem::Vector& r) {
          functional_call_args_[0] = u;

          r = (*K_functional_)(functional_call_args_);
          r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          functional_call_args_[0] = u;

          auto [r, drdu] = (*K_functional_)(functional_call_args_, Index<0>{});
          J_             = assemble(drdu);
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          return *J_;
        });

    return residual;
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    // Build the dof array lookup tables
    displacement_.space().BuildDofToArrays();

    if (is_quasistatic_) {
      residual_ = buildQuasistaticOperator();
    } else {
      // the dynamic case is described by a residual function and a second order
      // ordinary differential equation. Here, we define the residual function in
      // terms of an acceleration.
      residual_ = std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
            functional_call_args_[0] = d2u_dt2;

            auto M_residual = (*M_functional_)(functional_call_args_);

            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, c0_, d2u_dt2, K_arg);
            functional_call_args_[0] = K_arg;

            auto K_residual = (*K_functional_)(functional_call_args_);

            functional_call_args_[0] = u_;

            add(M_residual, K_residual, r);
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
            functional_call_args_[0] = d2u_dt2;

            auto M = serac::get<1>((*M_functional_)(functional_call_args_, Index<0>{}));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J = M + c0 * H(u_predicted)
            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, c0_, d2u_dt2, K_arg);
            functional_call_args_[0] = K_arg;

            auto K = serac::get<1>((*K_functional_)(functional_call_args_, Index<0>{}));

            functional_call_args_[0] = u_;

            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            J_.reset(mfem::Add(1.0, *m_mat, c0_, *k_mat));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

    nonlin_solver_.SetOperator(*residual_);
  }

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current displacement state is valid
   * @note If the essential boundary state is not specified, homogeneous essential boundary conditions are applied
   *
   * @param[in] adjoint_load The dual state that contains the right hand side of the adjoint system (d quantity of
   * interest/d displacement)
   * @param[in] dual_with_essential_boundary A optional finite element dual containing the non-homogenous essential
   * boundary condition data for the adjoint problem
   * @return The computed adjoint finite element state
   */
  virtual const serac::FiniteElementState& solveAdjoint(FiniteElementDual& adjoint_load,
                                                        FiniteElementDual* dual_with_essential_boundary = nullptr)
  {
    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Set the mesh nodes to the reference configuration
      mesh_.NewNodes(*reference_nodes_);
    }

    mfem::HypreParVector adjoint_load_vector(adjoint_load);

    // Add the sign correction to move the term to the RHS
    adjoint_load_vector *= -1.0;

    auto& lin_solver = nonlin_solver_.LinearSolver();

    // By default, use a homogeneous essential boundary condition
    mfem::HypreParVector adjoint_essential(adjoint_load);
    adjoint_essential = 0.0;

    functional_call_args_[0] = displacement_;

    auto [r, drdu] = (*K_functional_)(functional_call_args_, Index<0>{});
    auto jacobian  = assemble(drdu);
    auto J_T       = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    // If we have a non-homogeneous essential boundary condition, extract it from the given state
    if (dual_with_essential_boundary) {
      adjoint_essential = *dual_with_essential_boundary;
    }

    for (const auto& bc : bcs_.essentials()) {
      bc.apply(*J_T, adjoint_load_vector, adjoint_essential);
    }

    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(adjoint_load_vector, adjoint_displacement_);

    // Reset the equation solver to use the full nonlinear residual operator
    nonlin_solver_.SetOperator(*residual_);

    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Update the mesh with the new deformed nodes
      mesh_.NewNodes(*deformed_nodes_);
    }

    return adjoint_displacement_;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the parameter field
   *
   * @tparam parameter_field The index of the parameter to take a derivative with respect to
   * @return The sensitivity with respect to the parameter
   *
   * @pre `solveAdjoint` with an appropriate adjoint load must be called prior to this method.
   */
  template <int parameter_field>
  FiniteElementDual& computeSensitivity()
  {
    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Set the mesh nodes to the reference configuration
      mesh_.NewNodes(*reference_nodes_);
    }

    functional_call_args_[0] = displacement_;

    auto [r, drdparam] = (*K_functional_)(functional_call_args_, Index<parameter_field + 1>{});

    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_displacement_, *parameter_sensitivities_[parameter_field]);

    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Set the mesh nodes back to the reference configuration
      mesh_.NewNodes(*deformed_nodes_);
    }

    return *parameter_sensitivities_[parameter_field];
  }

protected:
  void rotate()
  {
    if (coordinate_transform_) {
      // Rotate the mesh nodes
      auto* nodes = mesh_.GetNodes();

      rotateGridFunction(*coordinate_transform_, *nodes);

      // Get the displacement grid function from the state
      auto& displacement_gf = displacement_.gridFunction();

      // Rotate the displacement
      rotateGridFunction(*coordinate_transform_, displacement_gf);

      // Set the underlying true vector from the rotated grid function
      displacement_.setFromGridFunction(displacement_gf);

      // Get the velocity grid function
      auto& velocity_gf = velocity_.gridFunction();
      rotateGridFunction(*coordinate_transform_, velocity_gf);
      velocity_.setFromGridFunction(velocity_gf);

      auto& adjoint_disp = adjoint_displacement_.gridFunction();
      rotateGridFunction(*coordinate_transform_, adjoint_disp);
      adjoint_displacement_.setFromGridFunction(adjoint_disp);
    }
  }

  void undoRotate()
  {
    if (coordinate_transform_) {
      // Rotate the mesh nodes
      auto* nodes = mesh_.GetNodes();
      rotateGridFunction(*inv_coordinate_transform_, *nodes);

      // Get the displacement grid function from the state
      auto& displacement_gf = displacement_.gridFunction();

      // Rotate the displacement
      rotateGridFunction(*inv_coordinate_transform_, displacement_gf);

      // Set the underlying true vector from the rotated grid function
      displacement_.setFromGridFunction(displacement_gf);

      // Get the velocity grid function
      auto& velocity_gf = velocity_.gridFunction();
      rotateGridFunction(*inv_coordinate_transform_, velocity_gf);
      velocity_.setFromGridFunction(velocity_gf);

      auto& adjoint_disp = adjoint_displacement_.gridFunction();
      rotateGridFunction(*inv_coordinate_transform_, adjoint_disp);
      adjoint_displacement_.setFromGridFunction(adjoint_disp);
    }
  }

  void rotateGridFunction(const tensor<double, dim, dim>& rotation, mfem::GridFunction& grid_function)
  {
    tensor<double, dim> unrotated_vector;
    tensor<double, dim> rotated_vector;

    // For each degree of freedom, determine the appropriate vector degree of freedom indices

    for (int dof = 0; dof < grid_function.FESpace()->GetNDofs(); ++dof) {
      for (int component = 0; component < grid_function.FESpace()->GetVDim(); ++component) {
        int vector_dof_index        = grid_function.FESpace()->DofToVDof(dof, component);
        unrotated_vector[component] = grid_function(vector_dof_index);
      }

      rotated_vector = dot(rotation, unrotated_vector);

      for (int component = 0; component < grid_function.FESpace()->GetVDim(); ++component) {
        int vector_dof_index            = grid_function.FESpace()->DofToVDof(dof, component);
        grid_function(vector_dof_index) = rotated_vector[component];
      }
    }
  }

  /// The compile-time finite element trial space for displacement and velocity (H1 of order p)
  using trial = H1<order, dim>;

  /// The compile-time finite element test space for displacement and velocity (H1 of order p)
  using test = H1<order, dim>;

  /// The velocity finite element state
  FiniteElementState velocity_;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /// The displacement finite element state
  FiniteElementState adjoint_displacement_;

  /// Mass functional object
  std::unique_ptr<Functional<test(trial, parameter_space...)>> M_functional_;

  /// Stiffness functional object
  std::unique_ptr<Functional<test(trial, parameter_space...)>> K_functional_;

  /// The finite element states representing user-defined parameter fields
  std::array<std::reference_wrapper<FiniteElementState>, sizeof...(parameter_space)> parameter_states_;

  /// The sensitivities (dual vectors) with repect to each of the input parameter fields
  std::array<std::unique_ptr<FiniteElementDual>, sizeof...(parameter_space)> parameter_sensitivities_;

  /// The set of input trial space vectors (displacement + parameters) used to call the underlying functional
  std::vector<std::reference_wrapper<const mfem::Vector>> functional_call_args_;

  /**
   * @brief mfem::Operator that describes the nonlinear residual
   * and its gradient with respect to displacement
   */
  std::unique_ptr<mfem_ext::StdFunctionOperator> residual_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the second time derivative of displacement, given
   * the current displacement, velocity, and source terms
   */
  mfem_ext::SecondOrderODE ode2_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  mfem_ext::EquationSolver nonlin_solver_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u_;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector du_dt_;

  /// @brief the previous acceleration, used as a starting guess for newton's method
  mfem::Vector previous_;

  /// @brief Current time step
  double c0_;

  /// @brief Previous time step
  double c1_;

  /// @brief A flag denoting whether to compute geometric nonlinearities in the residual
  GeometricNonlinearities geom_nonlin_;

  /// @brief Pointer to the reference mesh data
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /// @brief Flag to indicate the final mesh node state post-destruction
  FinalMeshOption keep_deformation_;

  /// @brief Pointer to the deformed mesh data
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> component_disp_bdr_coef_;

  /// @brief An auxilliary zero vector
  mfem::Vector zero_;

  /// @brief Auxilliary identity rank 2 tensor
  const isotropic_tensor<double, dim, dim> I_ = Identity<dim>();

  std::optional<tensor<double, dim, dim>> coordinate_transform_;

  std::optional<tensor<double, dim, dim>> inv_coordinate_transform_;
};

}  // namespace serac
