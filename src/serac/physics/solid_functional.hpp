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

template < typename ... T >
struct Parameters{
    static constexpr int n = sizeof ... (T);
};

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
template <int order, int dim, typename parameters = Parameters<>, typename parameter_indices  = std::make_integer_sequence< int, parameters::n > >
class SolidFunctional;

template <int order, int dim, typename ... parameter_space, int ... parameter_indices >
class SolidFunctional< order, dim, Parameters< parameter_space ... >, std::integer_sequence< int, parameter_indices ... > > : public BasePhysics {
public:
  static constexpr int VALUE = 0, DERIVATIVE = 1;
  static constexpr auto I = Identity<dim>();

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
      : BasePhysics(2, order),
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
    mfem::ParFiniteElementSpace* test_space = &displacement_.space();
    std::array<mfem::ParFiniteElementSpace*, sizeof...(parameter_space) + 2> trial_spaces;
    trial_spaces[0] = &displacement_.space();
    trial_spaces[1] = &displacement_.space(); // the accelerations have the same trial space as displacement

    if constexpr (sizeof...(parameter_space) > 0) {
      for (size_t i = 0; i < sizeof...(parameter_space); ++i) {
        trial_spaces[i + 2]         = &(parameter_states_[i].get().space());
        parameter_sensitivities_[i] = std::make_unique<FiniteElementDual>(mesh_, parameter_states_[i].get().space());
        functional_call_args_.emplace_back(parameter_states_[i].get());
      }
    }

    residual_ = std::make_unique<Functional<test(trial, trial, parameter_space...)>>(test_space, trial_spaces);

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

    bcs_.addEssential(disp_bdr, disp_bdr_coef_, displacement_);
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

    bcs_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_, component);
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

    bcs_.setTime(time_);

    if (is_quasistatic_) {
      quasiStaticSolve();
      // Update the time for housekeeping purposes
      time_ += dt;
    } else {
      ode2_.Step(displacement_, velocity_, time_, dt);
    }

    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Update the mesh with the new deformed nodes
      deformed_nodes_->Set(1.0, displacement_.gridFunction());
      deformed_nodes_->Add(1.0, *reference_nodes_);

      mesh_.NewNodes(*deformed_nodes_);
    }

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
  template <typename MaterialType, typename StateType>
  void setMaterial(MaterialType material, const QuadratureData<StateType> & QData)
  {
    auto copy = QData;
    residual_->AddDomainIntegral(
        Dimension<dim>{},
        [this, material](auto /*x*/, auto & state, auto displacement, auto acceleration, auto ... params) {
          auto a = get<VALUE>(acceleration);
          auto du_dX = get<DERIVATIVE>(displacement);
          auto body_force = material.density * a + 0.0 * du_dX[0];
          auto stress = material(state, du_dX, serac::get<0>(params)...);

          if (geom_nonlin_ == GeometricNonlinearities::On) {
            auto F = I + du_dX;
            body_force = body_force * det(F);
            stress = dot(stress, inv(transpose(F)));
          }

          return serac::tuple{body_force, stress};
        },
        mesh_, copy);
  }

  template <typename MaterialType>
  void setMaterial(MaterialType material) {
    static_assert(std::is_same_v< typename MaterialType::State, Empty >, "Error: material requires state information, but none was provided.");
    setMaterial(material, QuadratureData<Empty>{});
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
   * @param body_force A source function for a prescribed body load
   *
   * @pre BodyForceType must have the operator (x, time, displacement, d displacement_dx) defined as the body force
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force)
  {
    residual_->AddDomainIntegral(
        Dimension<dim>{},
        [body_force, this](auto x, auto displacement, auto /* acceleration */, auto... /*params*/) {
          auto du_dX = get<DERIVATIVE>(displacement);
          auto one_dual = (1.0 + 0.0 * du_dX[0][0]);
          auto source = body_force(x, time_) * one_dual;
          if (geom_nonlin_ == GeometricNonlinearities::On) {
            source = source * det(I + du_dX);
          }
          return serac::tuple{source, zero{}}; 
        },
        mesh_);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   *
   * @pre TractionType must have the operator (x, normal, time) to return the thermal flux value
   */
  template <typename TractionType>
  void setPiolaTraction(TractionType traction_function)
  {
    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [this, traction_function](auto x, auto n, auto, auto, auto... params) {
          return -1.0 * traction_function(x, n, time_, params...);
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
    return std::make_unique<mfem_ext::StdFunctionOperator>(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& u, mfem::Vector& r) {
          functional_call_args_[0].get() = u;
          r = (*residual_)(functional_call_args_);
          r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          functional_call_args_[0].get() = u;
          auto [r, drdu] = (*residual_)(functional_call_args_, Index<0>{});
          J_             = assemble(drdu);
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          return *J_;
        });
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

    // Project the essential boundary coefficients
    for (auto& bc : bcs_.essentials()) {
      bc.projectBdr(displacement_, time_);
    }

    if (is_quasistatic_) {
      residual_with_bcs_ = buildQuasistaticOperator();
    } else {
      // the dynamic case is described by a residual function and a second order
      // ordinary differential equation. Here, we define the residual function in
      // terms of an acceleration.
      residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
            functional_call_args_[1].get() = d2u_dt2;
            add(1.0, u_, c0_, d2u_dt2, functional_call_args_[0].get());
            r = (*residual_)(functional_call_args_);
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
            functional_call_args_[1].get() = d2u_dt2;
            add(1.0, u_, c0_, d2u_dt2, functional_call_args_[0].get());

            // K := dR/du
            auto K = serac::get<1>((*residual_)(functional_call_args_, Index<0>{}));
            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            // M := dR/da
            auto M = serac::get<1>((*residual_)(functional_call_args_, Index<1>{}));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J = M + c0 * K
            J_.reset(mfem::Add(1.0, *m_mat, c0_, *k_mat));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

    nonlin_solver_.SetOperator(*residual_with_bcs_);
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

    auto [r, drdu] = (*residual_)(functional_call_args_, Index<0>{});
    auto jacobian  = assemble(drdu);
    auto J_T       = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    // If we have a non-homogeneous essential boundary condition, extract it from the given state
    if (dual_with_essential_boundary) {
      adjoint_essential = *dual_with_essential_boundary;
    }

    for (const auto& bc : bcs_.essentials()) {
      bc.eliminateFromMatrix(*J_T);
      bc.eliminateToRHS(*J_T, adjoint_essential, adjoint_load_vector);
    }

    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(adjoint_load_vector, adjoint_displacement_);

    // Reset the equation solver to use the full nonlinear residual operator
    nonlin_solver_.SetOperator(*residual_with_bcs_);

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

    auto [r, drdparam] = (*residual_)(functional_call_args_, Index<parameter_field + 1>{});

    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_displacement_, *parameter_sensitivities_[parameter_field]);

    if (geom_nonlin_ == GeometricNonlinearities::On) {
      // Set the mesh nodes back to the reference configuration
      mesh_.NewNodes(*deformed_nodes_);
    }

    return *parameter_sensitivities_[parameter_field];
  }

protected:
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

  std::unique_ptr<Functional<test(trial, trial, parameter_space...)>> residual_;

  /// @brief mfem::Operator that calculates the residual after applying essential boundary conditions 
  std::unique_ptr<mfem_ext::StdFunctionOperator> residual_with_bcs_;

  /// The finite element states representing user-defined parameter fields
  std::array<std::reference_wrapper<FiniteElementState>, sizeof...(parameter_space)> parameter_states_;

  /// The sensitivities (dual vectors) with repect to each of the input parameter fields
  std::array<std::unique_ptr<FiniteElementDual>, sizeof...(parameter_space)> parameter_sensitivities_;

  /// The set of input trial space vectors (displacement + parameters) used to call the underlying functional
  std::vector<std::reference_wrapper<mfem::Vector>> functional_call_args_;

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

};

}  // namespace serac
