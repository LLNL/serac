// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file heat_transfer.hpp
 *
 * @brief An object containing the solver for a thermal conduction PDE
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/common.hpp"
#include "serac/physics/heat_transfer_input.hpp"
#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

namespace heat_transfer {

/**
 * @brief Reasonable defaults for most thermal linear solver options
 */
const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                    .preconditioner = Preconditioner::HypreL1Jacobi,
                                                    .relative_tol   = 1.0e-6,
                                                    .absolute_tol   = 1.0e-12,
                                                    .max_iterations = 200};

/// the default direct solver option for solving the linear stiffness equations
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU};

/**
 * @brief Reasonable defaults for most thermal nonlinear solver options
 */
const NonlinearSolverOptions default_nonlinear_options = {.nonlin_solver  = NonlinearSolver::Newton,
                                                          .relative_tol   = 1.0e-4,
                                                          .absolute_tol   = 1.0e-8,
                                                          .max_iterations = 500,
                                                          .print_level    = 1};

/**
 * @brief Reasonable defaults for dynamic thermal conduction simulations
 */
const TimesteppingOptions default_timestepping_options = {TimestepMethod::BackwardEuler,
                                                          DirichletEnforcementMethod::RateControl};

/**
 * @brief Reasonable defaults for static thermal conduction simulations
 */
const TimesteppingOptions default_static_options = {TimestepMethod::QuasiStatic};

}  // namespace heat_transfer

/**
 * @brief An object containing the solver for a thermal conduction PDE
 *
 * This is a generic linear thermal diffusion oeprator of the form
 *
 * \f[
 * \mathbf{M} \frac{\partial \mathbf{u}}{\partial t} = -\kappa \mathbf{K} \mathbf{u} + \mathbf{f}
 * \f]
 *
 *  where \f$\mathbf{M}\f$ is a mass matrix, \f$\mathbf{K}\f$ is a stiffness matrix, \f$\mathbf{u}\f$ is the
 *  temperature degree of freedom vector, and \f$\mathbf{f}\f$ is a thermal load vector.
 */
template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class HeatTransfer;

/// @overload
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class HeatTransfer<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public BasePhysics {
public:
  //! @cond Doxygen_Suppress
  static constexpr int  VALUE = 0, DERIVATIVE = 1;
  static constexpr int  SHAPE = 2;
  static constexpr auto I     = Identity<dim>();
  //! @endcond

  /// @brief The total number of non-parameter state variables (temperature, dtemp_dt, shape) passed to the FEM
  /// integrators
  static constexpr auto NUM_STATE_VARS = 3;

  /**
   * @brief Construct a new heat transfer object
   *
   * @param[in] nonlinear_opts The nonlinear solver options for solving the nonlinear residual equations
   * @param[in] lin_opts The linear solver options for solving the linearized Jacobian equations
   * @param[in] timestepping_opts The timestepping options for the heat transfer ordinary differential equations
   * @param[in] name An optional name for the physics module instance
   * used by an underlying material model or load
   * @param[in] pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  HeatTransfer(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
               const serac::TimesteppingOptions timestepping_opts, const std::string& name = "",
               mfem::ParMesh* pmesh = nullptr)
      : HeatTransfer(std::make_unique<EquationSolver>(nonlinear_opts, lin_opts,
                                                      StateManager::mesh(StateManager::collectionID(pmesh)).GetComm()),
                     timestepping_opts, name, pmesh)
  {
  }

  /**
   * @brief Construct a new heat transfer object
   *
   * @param[in] solver The nonlinear equation solver for the heat transfer equations
   * @param[in] timestepping_opts The timestepping options for the heat transfer ordinary differential equations
   * @param[in] name An optional name for the physics module instance
   * used by an underlying material model or load
   * @param[in] pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  HeatTransfer(std::unique_ptr<serac::EquationSolver> solver, const serac::TimesteppingOptions timestepping_opts,
               const std::string& name = "", mfem::ParMesh* pmesh = nullptr)
      : BasePhysics(NUM_STATE_VARS, order, name, pmesh),
        temperature_(StateManager::newState(
            FiniteElementState::Options{
                .order = order, .vector_dim = 1, .name = detail::addPrefix(name, "temperature")},
            sidre_datacoll_id_)),
        adjoint_temperature_(StateManager::newState(
            FiniteElementState::Options{
                .order = order, .vector_dim = 1, .name = detail::addPrefix(name, "adjoint_temperature")},
            sidre_datacoll_id_)),
        residual_with_bcs_(temperature_.space().TrueVSize()),
        nonlin_solver_(std::move(solver)),
        ode_(temperature_.space().TrueVSize(),
             {.time = ode_time_point_, .u = u_, .dt = dt_, .du_dt = previous_, .previous_dt = previous_dt_},
             *nonlin_solver_, bcs_)
  {
    SLIC_ERROR_ROOT_IF(
        mesh_.Dimension() != dim,
        axom::fmt::format("Compile time class dimension template parameter and runtime mesh dimension do not match"));

    SLIC_ERROR_ROOT_IF(
        !nonlin_solver_,
        "EquationSolver argument is nullptr in HeatTransfer constructor. It is possible that it was previously moved.");

    states_.push_back(&temperature_);
    states_.push_back(&adjoint_temperature_);

    parameters_.resize(sizeof...(parameter_space));

    // Create a pack of the primal field and parameter finite element spaces
    mfem::ParFiniteElementSpace* test_space = &temperature_.space();

    std::array<const mfem::ParFiniteElementSpace*, sizeof...(parameter_space) + NUM_STATE_VARS> trial_spaces;
    trial_spaces[0] = &temperature_.space();
    trial_spaces[1] = &temperature_.space();
    trial_spaces[2] = &shape_displacement_.space();

    if constexpr (sizeof...(parameter_space) > 0) {
      tuple<parameter_space...> types{};
      for_constexpr<sizeof...(parameter_space)>([&](auto i) {
        auto [fes, fec] =
            generateParFiniteElementSpace<typename std::remove_reference<decltype(get<i>(types))>::type>(&mesh_);
        parameters_[i].trial_space       = std::move(fes);
        parameters_[i].trial_collection  = std::move(fec);
        trial_spaces[i + NUM_STATE_VARS] = parameters_[i].trial_space.get();
      });
    }

    residual_ = std::make_unique<Functional<test(scalar_trial, scalar_trial, shape_trial, parameter_space...)>>(
        test_space, trial_spaces);

    nonlin_solver_->setOperator(residual_with_bcs_);

    // Check for dynamic mode
    if (timestepping_opts.timestepper != TimestepMethod::QuasiStatic) {
      ode_.SetTimestepper(timestepping_opts.timestepper);
      ode_.SetEnforcementMethod(timestepping_opts.enforcement_method);
      is_quasistatic_ = false;
    } else {
      is_quasistatic_ = true;
    }

    dt_          = 0.0;
    previous_dt_ = -1.0;

    int true_size = temperature_.space().TrueVSize();
    u_.SetSize(true_size);
    u_predicted_.SetSize(true_size);

    previous_.SetSize(true_size);
    previous_ = 0.0;

    zero_.SetSize(true_size);
    zero_ = 0.0;

    shape_displacement_  = 0.0;
    temperature_         = 0.0;
    adjoint_temperature_ = 0.0;
  }

  /**
   * @brief Construct a new Nonlinear HeatTransfer Solver object
   *
   * @param[in] options The solver information parsed from the input file
   * @param[in] name An optional name for the physics module instance. Note that this is NOT the mesh tag.
   */
  HeatTransfer(const HeatTransferInputOptions& options, const std::string& name = "")
      : HeatTransfer(options.nonlin_solver_options, options.lin_solver_options, options.timestepping_options, name)
  {
    if (options.initial_temperature) {
      auto temp = options.initial_temperature->constructScalar();
      temperature_.project(*temp);
    }

    if (options.source_coef) {
      // TODO: Not implemented yet in input files
      // NOTE: cannot use std::functions that use mfem::vector
      SLIC_ERROR("'source' is not implemented yet in input files.");
    }

    // Process the BCs in sorted order for correct behavior with repeated attributes
    std::map<std::string, input::BoundaryConditionInputOptions> sorted_bcs(options.boundary_conditions.begin(),
                                                                           options.boundary_conditions.end());
    for (const auto& [bc_name, bc] : sorted_bcs) {
      // FIXME: Better naming for boundary conditions?
      if (bc_name.find("temperature") != std::string::npos) {
        std::shared_ptr<mfem::Coefficient> temp_coef(bc.coef_opts.constructScalar());
        bcs_.addEssential(bc.attrs, temp_coef, temperature_.space(), *bc.coef_opts.component);
      } else if (bc_name.find("flux") != std::string::npos) {
        // TODO: Not implemented yet in input files
        // NOTE: cannot use std::functions that use mfem::vector
        SLIC_ERROR("'flux' is not implemented yet in input files.");
      } else {
        SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << name);
      }
    }
  }

  /**
   * @brief Set essential temperature boundary conditions (strongly enforced)
   *
   * @param[in] temp_bdr The boundary attributes on which to enforce a temperature
   * @param[in] temp The prescribed boundary temperature function
   */
  void setTemperatureBCs(const std::set<int>& temp_bdr, std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    temp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(temp);

    bcs_.addEssential(temp_bdr, temp_bdr_coef_, temperature_.space());
  }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to advance. For adaptive time integration methods, the actual timestep is returned.
   */
  void advanceTimestep(double& dt) override
  {
    if (is_quasistatic_) {
      time_ += dt;
      // Project the essential boundary coefficients
      for (auto& bc : bcs_.essentials()) {
        bc.setDofs(temperature_, time_);
      }
      nonlin_solver_->solve(temperature_);
    } else {
      SLIC_ASSERT_MSG(gf_initialized_[0], "Thermal state not initialized!");

      // Step the time integrator
      // Note that the ODE solver handles the essential boundary condition application itself
      ode_.Step(temperature_, time_, dt);
    }
    cycle_ += 1;
  }

  /**
   * @brief Set the thermal material model for the physics solver
   *
   * @tparam MaterialType The thermal material type
   * @param material A material containing heat capacity and thermal flux evaluation information
   *
   * @pre material must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial position of the material evaluation call
   *    2. `T temperature` the current temperature at the quadrature point
   *    3. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    4. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @pre MaterialType must return a serac::tuple of volumetric heat capacity and thermal flux when operator() is called
   * with the arguments listed above.
   */
  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, MaterialType material)
  {
    residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
        [material](auto x, auto temperature, auto dtemp_dt, auto shape, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = temperature;
          auto [p, dp_dX] = shape;
          auto du_dt      = get<VALUE>(dtemp_dt);

          auto I_plus_dp_dX     = I + dp_dX;
          auto inv_I_plus_dp_dX = inv(I_plus_dp_dX);
          auto det_I_plus_dp_dX = det(I_plus_dp_dX);

          // Note that the current configuration x = X + p, where X is the original reference
          // configuration and p is the shape displacement. We need the gradient with
          // respect to the perturbed reference configuration x = X + p for the material model. Therefore, we calculate
          // du/dx = du/dX * dX/dx = du/dX * (dx/dX)^-1 = du/dX * (I + dp/dX)^-1

          auto du_dx = dot(du_dX, inv_I_plus_dp_dX);

          auto [heat_capacity, heat_flux] = material(x + p, u, du_dx, params...);

          // Note that the return is integrated in the perturbed reference
          // configuration, hence the det(I + dp_dx) = det(dx/dX)
          return serac::tuple{heat_capacity * du_dt * det_I_plus_dp_dX,
                              -1.0 * dot(inv_I_plus_dp_dX, heat_flux) * det_I_plus_dp_dX};
        },
        mesh_);
  }

  /// @overload
  template <typename MaterialType>
  void setMaterial(MaterialType material)
  {
    setMaterial(DependsOn<>{}, material);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed temperature
   *
   * @param temp The function describing the temperature field
   */
  void setTemperature(std::function<double(const mfem::Vector& x, double t)> temp)
  {
    // Project the coefficient onto the grid function
    mfem::FunctionCoefficient temp_coef(temp);

    temp_coef.SetTime(time_);
    temperature_.project(temp_coef);
    gf_initialized_[0] = true;
  }

  /**
   * @brief Set the thermal source function
   *
   * @tparam SourceType The type of the source function
   * @param source_function A source function for a prescribed thermal load
   *
   * @pre source_function must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `T temperature` the current temperature at the quadrature point
   *    4. `tensor<T,dim>` the spatial gradient of the temperature at the quadrature point
   *    5. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   */
  template <int... active_parameters, typename SourceType>
  void setSource(DependsOn<active_parameters...>, SourceType source_function)
  {
    residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
        [source_function, this](auto x, auto temperature, auto /* dtemp_dt */, auto shape, auto... params) {
          // Get the value and the gradient from the input tuple
          auto [u, du_dX] = temperature;
          auto [p, dp_dX] = shape;

          auto I_plus_dp_dX = I + dp_dX;

          // Note that the current configuration x = X + p, where X is the original reference
          // configuration and p is the shape displacement. We need the gradient with
          // respect to the perturbed reference configuration x = X + p for the material model. Therefore, we calculate
          // du/dx = du/dX * dX/dx = du/dX * (dx/dX)^-1 = du/dX * (I + dp/dX)^-1

          auto du_dx = dot(du_dX, inv(I_plus_dp_dX));

          auto source = source_function(x + p, ode_time_point_, u, du_dx, params...);

          // Return the source and the flux as a tuple
          return serac::tuple{-1.0 * source * det(I_plus_dp_dX), serac::zero{}};
        },
        mesh_);
  }

  /// @overload
  template <typename SourceType>
  void setSource(SourceType source_function)
  {
    setSource(DependsOn<>{}, source_function);
  }

  /**
   * @brief Set the thermal flux boundary condition
   *
   * @tparam FluxType The type of the thermal flux object
   * @param flux_function A function describing the flux applied to a boundary
   *
   * @pre FluxType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    3. `double t` the time (note: time will be handled differently in the future)
   *    4. `T temperature` the current temperature at the quadrature point
   *    4. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note: until mfem::GetFaceGeometricFactors implements their JACOBIANS option,
   * (or we implement a replacement kernel ourselves) we are not able to compute
   * shape sensitivities for boundary integrals.
   */
  template <int... active_parameters, typename FluxType>
  void setFluxBCs(DependsOn<active_parameters...>, FluxType flux_function)
  {
    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
        [this, flux_function](auto X, auto u, auto /* dtemp_dt */, auto shape, auto... params) {
          auto temp = get<VALUE>(u);
          auto x    = X + shape;
          auto n    = cross(get<DERIVATIVE>(x));

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + dp_dxi)), so we multiply by the ratio w_new / w_old
          // to get
          //   q * area_correction * w_old
          // = q * (w_new / w_old) * w_old
          // = q * w_new
          auto area_correction = norm(n) / norm(cross(get<DERIVATIVE>(X)));
          return flux_function(x, normalize(n), ode_time_point_, temp, params...) * area_correction;
        },
        mesh_);
  }

  /// @overload
  template <typename FluxType>
  void setFluxBCs(FluxType flux_function)
  {
    setFluxBCs(DependsOn<>{}, flux_function);
  }

  /**
   * @brief Set the underlying finite element state to a prescribed shape displacement
   *
   * This field will perturb the mesh nodes as required by shape optimization workflows.
   *
   * @param shape_disp The function describing the shape displacement field
   */
  void setShapeDisplacement(std::function<void(const mfem::Vector& x, mfem::Vector& shape_disp)> shape_disp)
  {
    // Project the coefficient onto the grid function
    mfem::VectorFunctionCoefficient shape_disp_coef(dim, shape_disp);
    shape_displacement_.project(shape_disp_coef);
  }

  /// @overload
  void setShapeDisplacement(FiniteElementState& shape_disp) { shape_displacement_ = shape_disp; }

  /**
   * @brief Get the temperature state
   *
   * @return A reference to the current temperature finite element state
   */
  const serac::FiniteElementState& temperature() const { return temperature_; };

  /// @overload
  serac::FiniteElementState& temperature() { return temperature_; };

  /**
   * @brief Get the adjoint temperature state
   *
   * @return A reference to the current adjoint temperature finite element state
   */
  const serac::FiniteElementState& adjointTemperature() const { return adjoint_temperature_; };

  /// @overload
  serac::FiniteElementState& adjointTemperature() { return adjoint_temperature_; };

  /**
   * @brief Accessor for getting named finite element state fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  const FiniteElementState& state(const std::string& state_name) override
  {
    if (state_name == "temperature") {
      return temperature_;
    } else if (state_name == "adjoint_temperature") {
      return adjoint_temperature_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requestion from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return temperature_;
  }

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> stateNames() override
  {
    return std::vector<std::string>{{"temperature"}, {"adjoint_temperature"}};
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    // Build the dof array lookup tables
    temperature_.space().BuildDofToArrays();

    if (is_quasistatic_) {
      residual_with_bcs_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),

          [this](const mfem::Vector& u, mfem::Vector& r) {
            const mfem::Vector res =
                (*residual_)(u, zero_, shape_displacement_, *parameters_[parameter_indices].state...);

            // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
            // tracking strategy
            // See https://github.com/mfem/mfem/issues/3531
            r = res;
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& u) -> mfem::Operator& {
            auto [r, drdu] = (*residual_)(differentiate_wrt(u), zero_, shape_displacement_,
                                          *parameters_[parameter_indices].state...);
            J_             = assemble(drdu);
            J_e_           = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            return *J_;
          });

    } else {
      residual_with_bcs_ = mfem_ext::StdFunctionOperator(
          temperature_.space().TrueVSize(),

          [this](const mfem::Vector& du_dt, mfem::Vector& r) {
            add(1.0, u_, dt_, du_dt, u_predicted_);
            const mfem::Vector res =
                (*residual_)(u_predicted_, du_dt, shape_displacement_, *parameters_[parameter_indices].state...);

            // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
            // tracking strategy
            // See https://github.com/mfem/mfem/issues/3531
            r = res;
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& du_dt) -> mfem::Operator& {
            add(1.0, u_, dt_, du_dt, u_predicted_);

            // K := dR/du
            auto K = serac::get<DERIVATIVE>((*residual_)(differentiate_wrt(u_predicted_), du_dt, shape_displacement_,
                                                         *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            // M := dR/du_dot
            auto M = serac::get<DERIVATIVE>((*residual_)(u_predicted_, differentiate_wrt(du_dt), shape_displacement_,
                                                         *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J := M + dt K
            J_.reset(mfem::Add(1.0, *m_mat, dt_, *k_mat));
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }
  }

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current temperature state is valid
   * @pre The adjoint load maps are expected to contain a single entry named "temperature"
   * @note If the essential boundary dual is not specified, homogeneous essential boundary conditions are applied to
   * the adjoint system
   *
   * @param adjoint_loads An unordered map containing finite element duals representing the RHS of the adjoint equations
   * indexed by their name
   * @param adjoint_with_essential_boundary An unordered map containing finite element states representing the
   * non-homogeneous essential boundary condition data for the adjoint problem indexed by their name
   * @return An unordered map of the adjoint solutions indexed by their name. It has a single entry named
   * "adjoint_temperature"
   */
  const std::unordered_map<std::string, const serac::FiniteElementState&> solveAdjoint(
      std::unordered_map<std::string, const serac::FiniteElementDual&>  adjoint_loads,
      std::unordered_map<std::string, const serac::FiniteElementState&> adjoint_with_essential_boundary = {}) override
  {
    SLIC_ERROR_ROOT_IF(adjoint_loads.size() != 1,
                       "Adjoint load container is not the expected size of 1 in the heat transfer module.");

    auto temp_adjoint_load = adjoint_loads.find("temperature");

    SLIC_ERROR_ROOT_IF(temp_adjoint_load == adjoint_loads.end(), "Adjoint load for \"temperature\" not found.");

    mfem::HypreParVector adjoint_load_vector(temp_adjoint_load->second);

    // Add the sign correction to move the term to the RHS
    adjoint_load_vector *= -1.0;

    auto& lin_solver = nonlin_solver_->linearSolver();

    // By default, use a homogeneous essential boundary condition
    mfem::HypreParVector adjoint_essential(temp_adjoint_load->second);
    adjoint_essential = 0.0;

    auto [r, drdu] = (*residual_)(differentiate_wrt(temperature_), zero_, shape_displacement_,
                                  *parameters_[parameter_indices].state...);
    auto jacobian  = assemble(drdu);
    auto J_T       = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    // If we have a non-homogeneous essential boundary condition, extract it from the given state
    auto essential_adjoint_temp = adjoint_with_essential_boundary.find("temperature");

    if (essential_adjoint_temp != adjoint_with_essential_boundary.end()) {
      adjoint_essential = essential_adjoint_temp->second;
    } else {
      // If the essential adjoint load container does not have a temperature dual but it has a non-zero size, the
      // user has supplied an incorrectly-named dual vector.
      SLIC_ERROR_IF(adjoint_with_essential_boundary.size() != 0,
                    "Essential adjoint boundary condition given for an unexpected primal field. Expected adjoint "
                    "boundary condition named \"temperature\".");
    }

    for (const auto& bc : bcs_.essentials()) {
      bc.apply(*J_T, adjoint_load_vector, adjoint_essential);
    }

    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(adjoint_load_vector, adjoint_temperature_);

    // Reset the equation solver to use the full nonlinear residual operator
    nonlin_solver_->setOperator(residual_with_bcs_);

    return {{"adjoint_temperature", adjoint_temperature_}};
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
  FiniteElementDual& computeSensitivity(size_t parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' reqested for sensitivity."));

    auto drdparam     = serac::get<1>(d_residual_d_[parameter_field]());
    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_temperature_, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the shape displacement field
   *
   * @return The sensitivity with respect to the shape displacement
   *
   * @pre `solveAdjoint` with an appropriate adjoint load must be called prior to this method.
   */
  FiniteElementDual& computeShapeSensitivity() override
  {
    auto drdshape = serac::get<DERIVATIVE>((*residual_)(DifferentiateWRT<SHAPE>{}, temperature_, zero_,
                                                        shape_displacement_, *parameters_[parameter_indices].state...));

    auto drdshape_mat = assemble(drdshape);

    drdshape_mat->MultTranspose(adjoint_temperature_, shape_displacement_sensitivity_);

    return shape_displacement_sensitivity_;
  }

  /// Destroy the Thermal Solver object
  virtual ~HeatTransfer() = default;

protected:
  /// The compile-time finite element trial space for thermal conduction (H1 of order p)
  using scalar_trial = H1<order>;

  /// The compile-time finite element trial space for shape displacement (vector H1 of order 1)
  using shape_trial = H1<1, dim>;

  /// The compile-time finite element test space for thermal conduction (H1 of order p)
  using test = H1<order>;

  /// The temperature finite element state
  serac::FiniteElementState temperature_;

  /// The adjoint temperature finite element state
  serac::FiniteElementState adjoint_temperature_;

  /// serac::Functional that is used to calculate the residual and its derivatives
  std::unique_ptr<Functional<test(scalar_trial, scalar_trial, shape_trial, parameter_space...)>> residual_;

  /// Assembled mass matrix
  std::unique_ptr<mfem::HypreParMatrix> M_;

  /// Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> temp_bdr_coef_;

  /**
   * @brief mfem::Operator that describes the weight residual
   * and its gradient with respect to temperature
   */
  mfem_ext::StdFunctionOperator residual_with_bcs_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  std::unique_ptr<EquationSolver> nonlin_solver_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the time derivative of temperature, given
   * the current temperature and source terms
   */
  mfem_ext::FirstOrderODE ode_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// rows and columns of J_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_;

  /// The current timestep
  double dt_;

  /// The previous timestep
  double previous_dt_;

  /// An auxilliary zero vector
  mfem::Vector zero_;

  /// Predicted temperature true dofs
  mfem::Vector u_;

  /// Predicted temperature true dofs
  mfem::Vector u_predicted_;

  /// Previous value of du_dt used to prime the pump for the nonlinear solver
  mfem::Vector previous_;

  /// @brief Array functions computing the derivative of the residual with respect to each given parameter
  /// @note This is needed so the user can ask for a specific sensitivity at runtime as opposed to it being a
  /// template parameter.
  std::array<std::function<decltype((*residual_)(DifferentiateWRT<0>{}, temperature_, zero_, shape_displacement_,
                                                 *parameters_[parameter_indices].state...))()>,
             sizeof...(parameter_indices)>
      d_residual_d_ = {[&]() {
        return (*residual_)(DifferentiateWRT<NUM_STATE_VARS + parameter_indices>{}, temperature_, zero_,
                            shape_displacement_, *parameters_[parameter_indices].state...);
      }...};
};

}  // namespace serac
