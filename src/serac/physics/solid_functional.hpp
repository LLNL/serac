// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file thermal_conduction.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

/**
 * @brief Enum describing the generic solid boundary conditions
 *
 */
enum class SolidBoundaryCondition
{
  ReferencePressure, /**< Pressure applied in the reference configuration */
  ReferenceTraction, /**< Traction applied in the reference configuration */
  DeformedPressure,  /**< Pressure applied in the deformed (current) configuration */
  DeformedTraction   /**< Traction applied in the deformed (current) configuration */
};

/**
 * @brief Enum to save the deformation after the Solid module is destructed
 *
 */
enum class FinalMeshOption
{
  Deformed, /**< Keep the mesh in the deformed state post-destruction */
  Reference /**< Revert the mesh to the reference state post-destruction */
};

/**
 * @brief Enum to denote the previous solve completed in the Solid module
 *
 */
enum class PreviousSolve
{
  Forward, /**< Previous solve was a forward analysis */
  Adjoint, /**< Previous solve was an adjoint analysis */
  None     /**< No solves have been completed */
};

enum class GeometricNonlinearities
{
  On, /**< Include geometric nonlinearities */
  Off /**< Do not include geometric nonlinearities */
};

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear hyperelastic quasi-static and dynamic
 * hyperelastic solver object. It is derived from MFEM
 * example 10p.
 */
template <int order, int dim>
class SolidFunctional : public BasePhysics {
public:
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

  SolidFunctional(const SolverOptions& options, GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On,
                  FinalMeshOption keep_deformation = FinalMeshOption::Deformed, const std::string& name = "")
      : BasePhysics(2, order),
        velocity_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "velocity")})),
        displacement_(StateManager::newState(FiniteElementState::Options{
            .order = order, .vector_dim = mesh_.Dimension(), .name = detail::addPrefix(name, "displacement")})),
        M_functional_(&displacement_.space(), &displacement_.space()),
        K_functional_(&displacement_.space(), &displacement_.space()),
        ode2_(displacement_.space().TrueVSize(), {.c0 = c0_, .c1 = c1_, .u = u_, .du_dt = du_dt_, .d2u_dt2 = previous_},
              nonlin_solver_, bcs_),
        geom_nonlin_(geom_nonlin),
        keep_deformation_(keep_deformation)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension and runtime mesh dimension mismatch"));

    state_.push_back(velocity_);
    state_.push_back(displacement_);

    // Initialize the mesh node pointers
    reference_nodes_ = displacement_.createOnSpace<mfem::ParGridFunction>();
    mesh_.EnsureNodes();
    mesh_.GetNodes(*reference_nodes_);

    reference_nodes_->GetTrueDofs(x_);
    deformed_nodes_ = std::make_unique<mfem::ParGridFunction>(*reference_nodes_);

    displacement_.trueVec() = 0.0;
    velocity_.trueVec()     = 0.0;

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

  ~SolidFunctional()
  {
    // Update the mesh with the new deformed nodes if requested
    if (keep_deformation_ == FinalMeshOption::Deformed) {
      *reference_nodes_ += displacement_.gridFunc();
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
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp The prescribed boundary temperature function
   */
  void setDisplacementBCs(const std::set<int>&                                           disp_bdr,
                          std::function<void(const mfem::Vector& x, mfem::Vector& disp)> disp)
  {
    // Project the coefficient onto the grid function
    disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_.addEssential(disp_bdr, disp_bdr_coef_, displacement_);
  }

  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<double(const mfem::Vector& x)> disp,
                          int component)
  {
    // Project the coefficient onto the grid function
    component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(disp);

    bcs_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_, component);
  }

  // Solve the Quasi-static Newton system
  void quasiStaticSolve() { nonlin_solver_.Mult(zero_, displacement_.trueVec()); }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  void advanceTimestep(double& dt) override
  {
    // Initialize the true vector
    velocity_.initializeTrueVec();
    displacement_.initializeTrueVec();

    // Set the mesh nodes to the reference configuration
    mesh_.NewNodes(*reference_nodes_);

    bcs_.setTime(time_);

    if (is_quasistatic_) {
      quasiStaticSolve();
      // Update the time for housekeeping purposes
      time_ += dt;
    } else {
      ode2_.Step(displacement_.trueVec(), velocity_.trueVec(), time_, dt);
    }

    // Distribute the shared DOFs
    velocity_.distributeSharedDofs();
    displacement_.distributeSharedDofs();

    // Update the mesh with the new deformed nodes
    deformed_nodes_->Set(1.0, displacement_.gridFunc());
    deformed_nodes_->Add(1.0, *reference_nodes_);

    mesh_.NewNodes(*deformed_nodes_);

    cycle_ += 1;
  }

  /**
   * @brief Set the thermal flux and mass properties for the physics module
   *
   * @tparam MaterialType The thermal material type
   * @param material A material containing density, specific heat, and thermal flux evaluation information
   *
   * @pre MaterialType must have a method specificHeatCapacity() defining the specific heat
   * @pre MaterialType must have a method density() defining the density
   * @pre MaterialType must have the operator (temperature, d temperature_dx) defined as the thermal flux
   */
  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [this, material](auto, auto displacement) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX]    = displacement;
          double geom_factor = (geom_nonlin_ == GeometricNonlinearities::On ? 1.0 : 0.0);

          auto deformation_grad = du_dX + I_;
          auto flux             = material(du_dX) * (1.0 + geom_factor * (det(deformation_grad) - 1.0));

          auto source = u * 0.0;

          // Return the source and the flux as a tuple
          return serac::tuple{source, flux};
        },
        mesh_);

    M_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [this, material](auto x, auto displacement) {
          auto [u, du_dX] = displacement;

          auto flux = 0.0 * du_dX;

          double geom_factor = (geom_nonlin_ == GeometricNonlinearities::On ? 1.0 : 0.0);

          auto deformation_grad = du_dX + I_;
          auto source           = material.density(x) * u * (1.0 + geom_factor * (det(deformation_grad) - 1.0));

          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temp The function describing the temperature field
   */
  void setDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& disp)> disp)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient disp_coef(dim, disp);
    displacement_.project(disp_coef);
    gf_initialized_[1] = true;
  }

  void setVelocity(std::function<void(const mfem::Vector& x, mfem::Vector& vel)> vel)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient vel_coef(dim, vel);
    velocity_.project(vel_coef);
    gf_initialized_[0] = true;
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   *
   * @pre SourceType must have the operator (x, time, temperature, d temperature_dx) defined as the thermal source
   */
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force_function)
  {
    K_functional_.AddDomainIntegral(
        Dimension<dim>{},
        [body_force_function, this](auto x, auto displacement) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = displacement;

          auto flux = du_dX * 0.0;

          double geom_factor = (geom_nonlin_ == GeometricNonlinearities::On ? 1.0 : 0.0);

          auto deformation_grad = du_dX + I_;

          auto source = body_force_function(x, time_, u, du_dX) * (1.0 + geom_factor * (det(deformation_grad) - 1.0));
          return serac::tuple{source, flux};
        },
        mesh_);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the flux function
   * @param flux_function A function describing the thermal flux applied to a boundary
   *
   * @pre FluxType must have the operator (x, normal, temperature) to return the thermal flux value
   */

  template <typename TractionType>
  void setTractionBCs(TractionType traction_function, bool compute_on_reference = true)
  {
    // TODO fix this when we can get gradients from boundary integrals
    SLIC_ERROR_IF(!compute_on_reference, "SolidFunctional cannot compute traction BCs in deformed configuration");

    K_functional_.AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [this, traction_function](auto x, auto n, auto u) { return -1.0 * traction_function(x, n, time_) + 0.0 * u; },
        mesh_);
  }

  template <typename PressureType>
  void setPressureBCs(PressureType pressure_function, bool compute_on_reference = true)
  {
    // TODO fix this when we can get gradients from boundary integrals
    SLIC_ERROR_IF(!compute_on_reference, "SolidFunctional cannot compute pressure BCs in deformed configuration");

    K_functional_.AddBoundaryIntegral(
        Dimension<dim - 1>{},
        [this, pressure_function](auto x, auto n, auto u) { return pressure_function(x, time_) * n + 0.0 * u; }, mesh_);
  }

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& displacement() const { return displacement_; };

  /// @overload
  serac::FiniteElementState& displacement() { return displacement_; };

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& velocity() const { return velocity_; };

  /// @overload
  serac::FiniteElementState& velocity() { return velocity_; };

  void resetToReferenceConfiguration()
  {
    displacement_.gridFunc() = 0.0;
    velocity_.gridFunc()     = 0.0;

    velocity_.initializeTrueVec();
    displacement_.initializeTrueVec();

    mesh_.NewNodes(*reference_nodes_);
  }

  std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator()
  {
    // the quasistatic case is entirely described by the residual,
    // there is no ordinary differential equation
    auto residual = std::make_unique<mfem_ext::StdFunctionOperator>(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& u, mfem::Vector& r) {
          r = K_functional_(u);
          r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          K_functional_(u);
          J_.reset(grad(K_functional_));
          bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          return *J_;
        });

    return residual;
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    // Build the dof array lookup tables
    displacement_.space().BuildDofToArrays();

    // Project the essential boundary coefficients
    for (auto& bc : bcs_.essentials()) {
      bc.projectBdr(displacement_, time_);
    }

    // Initialize the true vector
    displacement_.initializeTrueVec();

    if (is_quasistatic_) {
      residual_ = buildQuasistaticOperator();
    } else {
      // the dynamic case is described by a residual function and a second order
      // ordinary differential equation. Here, we define the residual function in
      // terms of an acceleration.
      residual_ = std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, c0_, d2u_dt2, K_arg);

            add(M_functional_(d2u_dt2), K_functional_(K_arg), r);

            r.SetSubVector(bcs_.allEssentialDofs(), 0.0);
          },

          [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
            // J = M + c0 * H(u_predicted)
            mfem::Vector K_arg(u_.Size());
            add(1.0, u_, c0_, d2u_dt2, K_arg);

            M_functional_(u_);
            std::unique_ptr<mfem::HypreParMatrix> m_mat(grad(M_functional_));

            K_functional_(K_arg);
            std::unique_ptr<mfem::HypreParMatrix> k_mat(grad(K_functional_));

            J_.reset(mfem::Add(1.0, *m_mat, c0_, *k_mat));
            bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

    nonlin_solver_.SetOperator(*residual_);
  }

protected:
  /// The compile-time finite element trial space for thermal conduction (H1 of order p)
  using trial = H1<order, dim>;

  /// The compile-time finite element test space for thermal conduction (H1 of order p)
  using test = H1<order, dim>;

  /// The temperature finite element state
  FiniteElementState velocity_;
  FiniteElementState displacement_;

  /// Mass functional object \f$\mathbf{M} = \int_\Omega c_p \, \rho \, \phi_i \phi_j\, dx \f$
  Functional<test(trial)> M_functional_;

  /// Stiffness functional object \f$\mathbf{K} = \int_\Omega \theta \cdot \nabla \phi_i  + f \phi_i \, dx \f$
  Functional<test(trial)> K_functional_;

  /**
   * @brief mfem::Operator that describes the weight residual
   * and its gradient with respect to temperature
   */
  std::unique_ptr<mfem_ext::StdFunctionOperator> residual_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the time derivative of temperature, given
   * the current temperature and source terms
   */
  mfem_ext::SecondOrderODE ode2_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  mfem_ext::EquationSolver nonlin_solver_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /**
   * @brief alias for the reference mesh coordinates
   *
   *   this is used to correct for the fact that mfem's hyperelastic
   *   material model is based on the nodal positions rather than nodal displacements
   */
  mfem::Vector x_;

  /**
   * @brief used to communicate the ODE solver's predicted displacement to the residual operator
   */
  mfem::Vector u_;

  /**
   * @brief used to communicate the ODE solver's predicted velocity to the residual operator
   */
  mfem::Vector du_dt_;

  /**
   * @brief the previous acceleration, used as a starting guess for newton's method
   */
  mfem::Vector previous_;

  /**
   * @brief Current time step
   */
  double c0_;

  /**
   * @brief Previous time step
   */
  double c1_;

  GeometricNonlinearities geom_nonlin_;
  /**
   * @brief Pointer to the reference mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /**
   * @brief Flag to indicate the final mesh node state post-destruction
   */
  FinalMeshOption keep_deformation_;

  /**
   * @brief Pointer to the deformed mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> component_disp_bdr_coef_;

  /// An auxilliary zero vector
  mfem::Vector zero_;

  /// Auxilliary identity rank 2 tensor
  const tensor<double, dim, dim> I_ = Identity<dim>();
};

}  // namespace serac
