// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics.hpp
 *
 * @brief An object containing the solver for total Lagrangian finite deformation solid mechanics
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/common.hpp"
#include "serac/physics/solid_mechanics_input.hpp"
#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"

namespace serac {

namespace solid_mechanics {

/**
 * @brief default method and tolerances for solving the
 * systems of linear equations that show up in implicit
 * solid mechanics simulations
 */
const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                    .preconditioner = Preconditioner::HypreAMG,
                                                    .relative_tol   = 1.0e-6,
                                                    .absolute_tol   = 1.0e-16,
                                                    .max_iterations = 500,
                                                    .print_level    = 0};

/// the default direct solver option for solving the linear stiffness equations
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU, .print_level = 0};

/**
 * @brief default iteration limits, tolerances and verbosity for solving the
 * systems of nonlinear equations that show up in implicit
 * solid mechanics simulations
 */
const NonlinearSolverOptions default_nonlinear_options = {.nonlin_solver  = NonlinearSolver::Newton,
                                                          .relative_tol   = 1.0e-4,
                                                          .absolute_tol   = 1.0e-8,
                                                          .max_iterations = 10,
                                                          .print_level    = 1};

/// default quasistatic timestepping options for solid mechanics
const TimesteppingOptions default_quasistatic_options = {TimestepMethod::QuasiStatic};

/// default implicit dynamic timestepping options for solid mechanics
const TimesteppingOptions default_dynamic_options = {TimestepMethod::Newmark, DirichletEnforcementMethod::RateControl};

}  // namespace solid_mechanics

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidMechanics;

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
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public BasePhysics {
public:
  //! @cond Doxygen_Suppress
  static constexpr int  VALUE = 0, DERIVATIVE = 1;
  static constexpr auto I = Identity<dim>();
  //! @endcond

  /// @brief The total number of non-parameter state variables (displacement, velocity, shape) passed to the FEM
  /// integrators
  static constexpr auto NUM_STATE_VARS = 3;

  /**
   * @brief a list of the currently supported element geometries, by dimension
   * @note: this is hardcoded for now, since we currently
   * only support tensor product elements (1 element type per spatial dimension)
   */
  static constexpr Geometry geom = supported_geometries[dim];

  /**
   * @brief Construct a new SolidMechanics Functional object
   *
   * @param solver The nonlinear equation solver for the implicit solid mechanics equations
   * @param dynamic_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param name An optional name for the physics module instance
   * @param pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  SolidMechanics(const serac::mfem_ext::EquationSolver& solver, const serac::TimesteppingOptions dynamic_opts,
                 const GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On, const std::string& name = "",
                 mfem::ParMesh* pmesh = nullptr)
      : BasePhysics(2, order, name, pmesh),
        velocity_(StateManager::newState(
            FiniteElementState::Options{.order = order, .vector_dim = dim, .name = detail::addPrefix(name, "velocity")},
            sidre_datacoll_id_)),
        displacement_(StateManager::newState(
            FiniteElementState::Options{
                .order = order, .vector_dim = dim, .name = detail::addPrefix(name, "displacement")},
            sidre_datacoll_id_)),
        adjoint_displacement_(StateManager::newState(
            FiniteElementState::Options{
                .order = order, .vector_dim = dim, .name = detail::addPrefix(name, "adjoint_displacement")},
            sidre_datacoll_id_)),
        reactions_(StateManager::newDual(displacement_.space(), detail::addPrefix(name, "reactions"))),
        nonlin_solver_(solver),
        ode2_(displacement_.space().TrueVSize(),
              {.time = ode_time_point_, .c0 = c0_, .c1 = c1_, .u = u_, .du_dt = du_dt_, .d2u_dt2 = previous_},
              nonlin_solver_, bcs_),
        c0_(0.0),
        c1_(0.0),
        geom_nonlin_(geom_nonlin)
  {
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension, {0}, and runtime mesh dimension, {1}, mismatch", dim,
                                         mesh_.Dimension()));

    states_.push_back(&velocity_);
    states_.push_back(&displacement_);
    states_.push_back(&adjoint_displacement_);

    duals_.push_back(&reactions_);

    parameters_.resize(sizeof...(parameter_space));

    // Create a pack of the primal field and parameter finite element spaces
    mfem::ParFiniteElementSpace* test_space = &displacement_.space();

    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> trial_spaces;
    trial_spaces[0] = &displacement_.space();
    trial_spaces[1] = &displacement_.space();
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

    residual_ =
        std::make_unique<Functional<test(trial, trial, shape_trial, parameter_space...)>>(test_space, trial_spaces);

    displacement_         = 0.0;
    velocity_             = 0.0;
    shape_displacement_   = 0.0;
    adjoint_displacement_ = 0.0;

    // If the user wants the AMG preconditioner with a linear solver, set the pfes
    // to be the displacement
    auto* amg_prec = dynamic_cast<mfem::HypreBoomerAMG*>(nonlin_solver_.Preconditioner());
    if (amg_prec) {
      amg_prec->SetElasticityOptions(&displacement_.space());
    }

    // Check for dynamic mode
    if (dynamic_opts.timestepper != TimestepMethod::QuasiStatic) {
      ode2_.SetTimestepper(dynamic_opts.timestepper);
      ode2_.SetEnforcementMethod(dynamic_opts.enforcement_method);
      is_quasistatic_ = false;
    } else {
      is_quasistatic_ = true;
    }

    int true_size = velocity_.space().TrueVSize();

    u_.SetSize(true_size);
    du_dt_.SetSize(true_size);
    previous_.SetSize(true_size);
    previous_ = 0.0;

    du_.SetSize(true_size);
    du_ = 0.0;

    dr_.SetSize(true_size);
    dr_ = 0.0;

    predicted_displacement_.SetSize(true_size);
    predicted_displacement_ = 0.0;

    zero_.SetSize(true_size);
    zero_ = 0.0;
  }

  /**
   * @brief Construct a new Nonlinear SolidMechanics Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] name An optional name for the physics module instance. Note that this is NOT the mesh tag.
   */
  SolidMechanics(const SolidMechanicsInputOptions& input_options, const std::string& name = "")
      : SolidMechanics(
            mfem_ext::buildEquationSolver(input_options.nonlin_solver_options, input_options.lin_solver_options),
            input_options.timestepping_options, input_options.geom_nonlin, name)
  {
    // This is the only other options stored in the input file that we can use
    // in the initialization stage
    // TODO: move these material parameters out of the SolidMechanicsInputOptions
    if (input_options.material_nonlin) {
      solid_mechanics::NeoHookean mat{input_options.initial_mass_density, input_options.K, input_options.mu};
      setMaterial(mat);
    } else {
      solid_mechanics::LinearIsotropic mat{input_options.initial_mass_density, input_options.K, input_options.mu};
      setMaterial(mat);
    }

    if (input_options.initial_displacement) {
      displacement_.project(input_options.initial_displacement->constructVector(dim));
    }

    if (input_options.initial_velocity) {
      velocity_.project(input_options.initial_velocity->constructVector(dim));
    }

    for (const auto& [bc_name, bc] : input_options.boundary_conditions) {
      // FIXME: Better naming for boundary conditions?
      if (bc_name.find("displacement") != std::string::npos) {
        if (bc.coef_opts.isVector()) {
          std::shared_ptr<mfem::VectorCoefficient> disp_coef(bc.coef_opts.constructVector(dim));
          bcs_.addEssential(bc.attrs, disp_coef, displacement_.space());
        } else {
          SLIC_ERROR_ROOT_IF(
              !bc.coef_opts.component,
              "Component not specified with scalar coefficient when setting the displacement condition.");
          std::shared_ptr<mfem::Coefficient> disp_coef(bc.coef_opts.constructScalar());
          bcs_.addEssential(bc.attrs, disp_coef, displacement_.space(), *bc.coef_opts.component);
        }
      } else if (bc_name.find("traction") != std::string::npos) {
        // TODO: Not implemented yet in input files
        SLIC_ERROR("'traction' is not implemented yet in input files.");
      } else if (bc_name.find("traction_ref") != std::string::npos) {
        // TODO: Not implemented yet in input files
        SLIC_ERROR("'traction_ref' is not implemented yet in input files.");
      } else if (bc_name.find("pressure") != std::string::npos) {
        // TODO: Not implemented yet in input files
        SLIC_ERROR("'pressure' is not implemented yet in input files.");
      } else if (bc_name.find("pressure_ref") != std::string::npos) {
        // TODO: Not implemented yet in input files
        SLIC_ERROR("'pressure_ref' is not implemented yet in input files.");
      } else {
        SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << name);
      }
    }
  }

  /// @brief Destroy the SolidMechanics Functional object
  ~SolidMechanics() {}

  /**
   * @brief Create a shared ptr to a quadrature data buffer for the given material type
   *
   * @tparam T the type to be created at each quadrature point
   * @param initial_state the value to be broadcast to each quadrature point
   * @return std::shared_ptr< QuadratureData<T> >
   */
  template <typename T>
  std::shared_ptr<QuadratureData<T>> createQuadratureDataBuffer(T initial_state)
  {
    constexpr auto Q = order + 1;

    size_t num_elements        = size_t(mesh_.GetNE());
    size_t qpoints_per_element = GaussQuadratureRule<geom, Q>().size();

    auto  qdata     = std::make_shared<QuadratureData<T>>(num_elements, qpoints_per_element);
    auto& container = *qdata;
    for (size_t e = 0; e < num_elements; e++) {
      for (size_t q = 0; q < qpoints_per_element; q++) {
        container(e, q) = initial_state;
      }
    }

    return qdata;
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
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes on which to enforce a displacement
   * @param[in] disp The time-dependent prescribed boundary displacement function
   */
  void setDisplacementBCs(const std::set<int>&                                            disp_bdr,
                          std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
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

  /**
   * @brief Accessor for getting named finite element state fields from the physics modules
   *
   * @param state_name The name of the Finite Element State to retrieve
   * @return The named Finite Element State
   */
  const FiniteElementState& state(const std::string& state_name) override
  {
    if (state_name == "displacement") {
      return displacement_;
    } else if (state_name == "velocity") {
      return velocity_;
    } else if (state_name == "adjoint_displacement") {
      return adjoint_displacement_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requestion from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return displacement_;
  }

  /**
   * @brief Get a vector of the finite element state solution variable names
   *
   * @return The solution variable names
   */
  virtual std::vector<std::string> stateNames() override
  {
    return std::vector<std::string>{{"displacement"}, {"velocity"}, {"adjoint_displacement"}};
  }

  /**
   * @brief register a custom domain integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @tparam StateType the type that contains the internal variables (if any) for q-function
   * @param qfunction a callable that returns a tuple of body-force and stress
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * ~~~ {.cpp}
   *
   *  double lambda = 500.0;
   *  double mu = 500.0;
   *  solid_mechanics.addCustomDomainIntegral(DependsOn<>{}, [=](auto x, auto displacement, auto acceleration, auto
   * shape_displacement){ auto du_dx = serac::get<1>(displacement);
   *
   *    auto I       = Identity<dim>();
   *    auto epsilon = 0.5 * (transpose(du_dx) + du_dx);
   *    auto stress = lambda * tr(epsilon) * I + 2.0 * mu * epsilon;
   *
   *    auto d2u_dt2 = serac::get<0>(acceleration);
   *    double rho = 1.0 + x[0]; // spatially-varying density
   *
   *    return serac::tuple{rho * d2u_dt2, stress};
   *  });
   *
   * ~~~
   */
  template <int... active_parameters, typename callable, typename StateType = Nothing>
  void addCustomDomainIntegral(DependsOn<active_parameters...>, callable qfunction,
                               std::shared_ptr<QuadratureData<StateType>> qdata = NoQData)
  {
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
                                 qfunction, mesh_, qdata);
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param material A material that provides a function to evaluate stress
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public member variable `density`
   * @pre MaterialType must define operator() that returns the Cauchy stress
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setMaterial(DependsOn<active_parameters...>, MaterialType material,
                   std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    residual_->AddDomainIntegral(
        Dimension<dim>{},
        DependsOn<0, 1, 2,
                  active_parameters + NUM_STATE_VARS...>{},  // the magic number "+ NUM_STATE_VARS" accounts for the
                                                             // fact that the displacement, acceleration, and shape
                                                             // fields are always-on and come first, so the `n`th
                                                             // parameter will actually be argument `n + NUM_STATE_VARS`
        [this, material](auto /*x*/, auto& state, auto displacement, auto acceleration, auto shape, auto... params) {
          auto du_dX   = get<DERIVATIVE>(displacement);
          auto d2u_dt2 = get<VALUE>(acceleration);
          auto dp_dX   = get<DERIVATIVE>(shape);

          // Compute the displacement gradient with respect to the shape-adjusted coordinate.

          // Note that the current configuration x = X + u + p, where X is the original reference
          // configuration, u is the displacement, and p is the shape displacement. We need the gradient with
          // respect to the perturbed reference configuration X' = X + p for the material model. Therefore, we calculate
          // du/dX' = du/dX * dX/dX' = du/dX * (dX'/dX)^-1 = du/dX * (I + dp/dX)^-1

          auto du_dX_prime = dot(du_dX, inv(I + dp_dX));

          auto stress = material(state, du_dX_prime, params...);

          // dx_dX is the volumetric transform to get us back to the original
          // reference configuration (dx/dX = I + du/dX + dp/dX). If we are not including geometric
          // nonlinearities, we ignore the du/dX factor.

          auto dx_dX = 0.0 * du_dX + dp_dX + I;

          if (geom_nonlin_ == GeometricNonlinearities::On) {
            dx_dX += du_dX;
          }

          auto flux = dot(stress, transpose(inv(dx_dX))) * det(dx_dX);

          // This transpose on the stress in the following line is a
          // hack to fix a bug in the residual operator. The stress
          // should be transposed in the contraction of the Piola
          // stress with the shape function gradients.
          //
          // Note that the mass part of the return is integrated in the perturbed reference
          // configuration, hence the det(I + dp_dx) = det(dX'/dX)
          return serac::tuple{material.density * d2u_dt2 * det(I + dp_dX), flux};
        },
        mesh_, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(MaterialType material, std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
  {
    setMaterial(DependsOn<>{}, material, qdata);
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
   * @pre body_force must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...>, BodyForceType body_force)
  {
    residual_->AddDomainIntegral(
        Dimension<dim>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
        [body_force, this](auto x, auto /* displacement */, auto /* acceleration */, auto shape, auto... params) {
          // note: this assumes that the body force function is defined
          // per unit volume in the reference configuration
          auto p     = get<VALUE>(shape);
          auto dp_dX = get<DERIVATIVE>(shape);
          return serac::tuple{-1.0 * body_force(x + p, ode_time_point_, params...) * det(dp_dX + I), zero{}};
        },
        mesh_);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force)
  {
    addBodyForce(DependsOn<>{}, body_force);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   *
   * @pre TractionType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    2. `tensor<T,dim> n` the outward-facing unit normal for the quadrature point
   *    3. `double t` the time (note: time will be handled differently in the future)
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
  template <int... active_parameters, typename TractionType>
  void setPiolaTraction(DependsOn<active_parameters...>, TractionType traction_function)
  {
    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, 2, active_parameters + NUM_STATE_VARS...>{},
        [this, traction_function](auto x, auto n, auto, auto, auto shape, auto... params) {
          auto p = get<VALUE>(shape);
          return -1.0 * traction_function(x + p, n, ode_time_point_, params...);
        },
        mesh_);
  }

  /// @overload
  template <typename TractionType>
  void setPiolaTraction(TractionType traction_function)
  {
    setPiolaTraction(DependsOn<>{}, traction_function);
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
          const mfem::Vector res =
              (*residual_)(u, zero_, shape_displacement_, *parameters_[parameter_indices].state...);

          // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
          // tracking strategy
          // See https://github.com/mfem/mfem/issues/3531
          r = res;
          r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          auto [r, drdu] =
              (*residual_)(differentiate_wrt(u), zero_, shape_displacement_, *parameters_[parameter_indices].state...);
          J_   = assemble(drdu);
          J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
          return *J_;
        });
  }

  /**
   * @brief Return the assembled stiffness matrix
   *
   * This method returns a pair {K, K_e} representing the last computed linearized stiffness matrix.
   * The K matrix has the essential degree of freedom rows and columns zeroed with a
   * 1 on the diagonal and K_e contains the zeroed rows and columns, e.g. K_total = K + K_e.
   *
   * @warning This interface is not stable and may change in the future.
   *
   * @return A pair of the eliminated stiffness matrix and a matrix containing the eliminated rows and cols
   */
  std::pair<const mfem::HypreParMatrix&, const mfem::HypreParMatrix&> stiffnessMatrix() const
  {
    SLIC_ERROR_ROOT_IF(!J_ || !J_e_, "Stiffness matrix has not yet been assembled.");

    return {*J_, *J_e_};
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    if constexpr (sizeof...(parameter_space) > 0) {
      for (size_t i = 0; i < sizeof...(parameter_space); i++) {
        SLIC_ERROR_ROOT_IF(!parameters_[i].state,
                           "all parameters fields must be initialized before calling completeSetup()");
      }
    }

    // Build the dof array lookup tables
    displacement_.space().BuildDofToArrays();

    if (is_quasistatic_) {
      residual_with_bcs_ = buildQuasistaticOperator();

      // the residual calculation uses the old stiffness matrix
      // to help apply essential boundary conditions, so we
      // compute J here to prime the pump for the first solve
      residual_with_bcs_->GetGradient(displacement_);

    } else {
      // the dynamic case is described by a residual function and a second order
      // ordinary differential equation. Here, we define the residual function in
      // terms of an acceleration.
      residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
            add(1.0, u_, c0_, d2u_dt2, predicted_displacement_);
            const mfem::Vector res = (*residual_)(predicted_displacement_, d2u_dt2, shape_displacement_,
                                                  *parameters_[parameter_indices].state...);

            // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
            // tracking strategy
            // See https://github.com/mfem/mfem/issues/3531
            r = res;
            r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
          },

          [this](const mfem::Vector& d2u_dt2) -> mfem::Operator& {
            add(1.0, u_, c0_, d2u_dt2, predicted_displacement_);

            // K := dR/du
            auto K =
                serac::get<DERIVATIVE>((*residual_)(differentiate_wrt(predicted_displacement_), d2u_dt2,
                                                    shape_displacement_, *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            // M := dR/da
            auto M =
                serac::get<DERIVATIVE>((*residual_)(predicted_displacement_, differentiate_wrt(d2u_dt2),
                                                    shape_displacement_, *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J = M + c0 * K
            J_.reset(mfem::Add(1.0, *m_mat, c0_, *k_mat));
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

    nonlin_solver_.SetOperator(*residual_with_bcs_);
  }

  /// @brief Solve the Quasi-static Newton system
  void quasiStaticSolve(double dt)
  {
    time_ += dt;

    auto& lin_solver = nonlin_solver_.LinearSolver();

    // the ~20 lines of code below are essentially equivalent to the 1-liner
    // u += dot(inv(J), dot(J_elim[:, dofs], (U(t + dt) - u)[dofs]));

    du_ = 0.0;

    // Set the essential boundary conditions for the current time
    for (auto& bc : bcs_.essentials()) {
      bc.setDofs(du_, time_);
    }

    // Compute the change in essential boundary condition dofs and store it in du_
    auto& constrained_dofs = bcs_.allEssentialTrueDofs();
    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j = constrained_dofs[i];
      du_[j] -= displacement_(j);
    }

    // Compute the reaction forces associated with this change in essential boundary condition
    dr_ = 0.0;
    mfem::EliminateBC(*J_, *J_e_, constrained_dofs, du_, dr_);

    lin_solver.SetOperator(*J_);

    // Solve for the updated displacement due to these reaction forces
    lin_solver.Mult(dr_, du_);

    // Modify our initial guess with the displacement due to the reaction forces
    displacement_ += du_;

    nonlin_solver_.Mult(zero_, displacement_);
  }

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping
   * schemes
   * @pre SolidMechanics::completeSetup() must be called prior to this call
   */
  void advanceTimestep(double& dt) override
  {
    SLIC_ERROR_ROOT_IF(!residual_, "completeSetup() must be called prior to advanceTimestep(dt) in SolidMechanics.");

    // bcs_.setTime(time_);

    if (is_quasistatic_) {
      quasiStaticSolve(dt);
    } else {
      ode2_.Step(displacement_, velocity_, time_, dt);
    }

    {
      // after finding displacements that satisfy equilibrium,
      // compute the residual one more time, this time enabling
      // the material state buffers to be updated
      residual_->update_qdata = true;

      // this seems like the wrong way to be doing this assignment, but
      // reactions_ = residual(displacement, ...);
      // isn't currently supported
      reactions_.Vector::operator=(
          (*residual_)(displacement_, zero_, shape_displacement_, *parameters_[parameter_indices].state...));
      // TODO (talamini1): Fix above reactions for dynamics. Setting the accelerations to zero
      // works for quasi-statics, but we need to account for the accelerations in
      // dynamics. We need to figure out how to get the updated accelerations out of the
      // ODE solver.

      residual_->update_qdata = false;
    }

    cycle_ += 1;
  }

  /**
   * @brief Solve the adjoint problem
   * @pre It is expected that the forward analysis is complete and the current displacement state is valid
   * @pre The adjoint load maps are expected to contain a single entry named "displacement"
   * @note If the essential boundary dual is not specified, homogeneous essential boundary conditions are applied to
   * the adjoint system
   *
   * @param adjoint_loads An unordered map containing finite element duals representing the RHS of the adjoint equations
   * indexed by their name
   * @param adjoint_with_essential_boundary A unordered map containing finite element states representing the
   * non-homogeneous essential boundary condition data for the adjoint problem indexed their name
   * @return An unordered map of the adjoint solutions indexed by their name. It has a single entry named
   * "adjoint_displacement"
   */
  const std::unordered_map<std::string, const serac::FiniteElementState&> solveAdjoint(
      std::unordered_map<std::string, const serac::FiniteElementDual&>  adjoint_loads,
      std::unordered_map<std::string, const serac::FiniteElementState&> adjoint_with_essential_boundary = {}) override
  {
    SLIC_ERROR_ROOT_IF(adjoint_loads.size() != 1,
                       "Adjoint load container is not the expected size of 1 in the solid mechanics module.");

    auto disp_adjoint_load = adjoint_loads.find("displacement");

    SLIC_ERROR_ROOT_IF(disp_adjoint_load == adjoint_loads.end(), "Adjoint load for \"displacement\" not found.");
    mfem::HypreParVector adjoint_load_vector(disp_adjoint_load->second);

    // Add the sign correction to move the term to the RHS
    adjoint_load_vector *= -1.0;

    auto& lin_solver = nonlin_solver_.LinearSolver();

    // By default, use a homogeneous essential boundary condition
    mfem::HypreParVector adjoint_essential(disp_adjoint_load->second);
    adjoint_essential = 0.0;

    // sam: is this the right thing to be doing for dynamics simulations,
    // or are we implicitly assuming this should only be used in quasistatic analyses?
    auto drdu     = serac::get<DERIVATIVE>((*residual_)(differentiate_wrt(displacement_), zero_, shape_displacement_,
                                                    *parameters_[parameter_indices].state...));
    auto jacobian = assemble(drdu);
    auto J_T      = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    // If we have a non-homogeneous essential boundary condition, extract it from the given state
    auto essential_adjoint_disp = adjoint_with_essential_boundary.find("displacement");

    if (essential_adjoint_disp != adjoint_with_essential_boundary.end()) {
      adjoint_essential = essential_adjoint_disp->second;
    } else {
      // If the essential adjoint load container does not have a displacement dual but it has a non-zero size, the
      // user has supplied an incorrectly-named dual vector.
      SLIC_ERROR_IF(adjoint_with_essential_boundary.size() != 0,
                    "Essential adjoint boundary condition given for an unexpected primal field. Expected adjoint "
                    "boundary condition named \"displacement\"");
    }

    for (const auto& bc : bcs_.essentials()) {
      bc.apply(*J_T, adjoint_load_vector, adjoint_essential);
    }

    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(adjoint_load_vector, adjoint_displacement_);

    return {{"adjoint_displacement", adjoint_displacement_}};
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the parameter field
   *
   * @param parameter_field The index of the parameter to take a derivative with respect to
   * @return The sensitivity with respect to the parameter
   *
   * @pre `solveAdjoint` with an appropriate adjoint load must be called prior to this method.
   */
  FiniteElementDual& computeSensitivity(size_t parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' requested for sensitivity."));

    auto drdparam = serac::get<DERIVATIVE>(d_residual_d_[parameter_field]());

    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_displacement_, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /**
   * @brief Compute the implicit sensitivity of the quantity of interest used in defining the load for the adjoint
   * problem with respect to the shape displacement field
   *
   * @return The sensitivity with respect to the shape displacement
   */
  FiniteElementDual& computeShapeSensitivity() override
  {
    auto drdshape = serac::get<DERIVATIVE>((*residual_)(DifferentiateWRT<2>{}, displacement_, zero_,
                                                        shape_displacement_, *parameters_[parameter_indices].state...));

    auto drdshape_mat = assemble(drdshape);

    drdshape_mat->MultTranspose(adjoint_displacement_, shape_displacement_sensitivity_);

    return shape_displacement_sensitivity_;
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

  /// @brief getter for nodal forces (before zeroing-out essential dofs)
  const serac::FiniteElementDual& reactions() { return reactions_; };

protected:
  /// The compile-time finite element trial space for displacement and velocity (H1 of order p)
  using trial = H1<order, dim>;

  /// The compile-time finite element test space for displacement and velocity (H1 of order p)
  using test = H1<order, dim>;

  /// The compile-time finite element trial space for shape displacement (H1 of order 1, nodal displacements)
  /// The choice of polynomial order for the shape sensitivity is determined in the StateManager
  using shape_trial = H1<SHAPE_ORDER, dim>;

  /// The velocity finite element state
  FiniteElementState velocity_;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /// The displacement finite element state
  FiniteElementState adjoint_displacement_;

  /// nodal forces
  FiniteElementDual reactions_;

  /// serac::Functional that is used to calculate the residual and its derivatives
  std::unique_ptr<Functional<test(trial, trial, shape_trial, parameter_space...)>> residual_;

  /// mfem::Operator that calculates the residual after applying essential boundary conditions
  std::unique_ptr<mfem_ext::StdFunctionOperator> residual_with_bcs_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  mfem_ext::EquationSolver nonlin_solver_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the second time derivative of displacement, given
   * the current displacement, velocity, and source terms
   */
  mfem_ext::SecondOrderODE ode2_;

  /// Assembled sparse matrix for the Jacobian
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// rows and columns of J_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_;

  /// an intermediate variable used to store the predicted end-step displacement
  mfem::Vector predicted_displacement_;

  /// vector used to store the change in essential bcs between timesteps
  mfem::Vector du_;

  /// vector used to store forces arising from du_ when applying time-dependent bcs
  mfem::Vector dr_;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u_;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector du_dt_;

  /// @brief the previous acceleration, used as a starting guess for newton's method
  mfem::Vector previous_;

  /// coefficient used to calculate predicted displacement: u_p := u + c0 * d2u_dt2
  double c0_;

  /// coefficient used to calculate predicted velocity: dudt_p := dudt + c1 * d2u_dt2
  double c1_;

  /// @brief A flag denoting whether to compute geometric nonlinearities in the residual
  GeometricNonlinearities geom_nonlin_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> component_disp_bdr_coef_;

  /// @brief An auxilliary zero vector
  mfem::Vector zero_;

  /// @brief Array functions computing the derivative of the residual with respect to each given parameter
  /// @note This is needed so the user can ask for a specific sensitivity at runtime as opposed to it being a
  /// template parameter.
  std::array<std::function<decltype((*residual_)(DifferentiateWRT<0>{}, displacement_, zero_, shape_displacement_,
                                                 *parameters_[parameter_indices].state...))()>,
             sizeof...(parameter_indices)>
      d_residual_d_ = {[&]() {
        return (*residual_)(DifferentiateWRT<NUM_STATE_VARS + parameter_indices>{}, displacement_, zero_,
                            shape_displacement_, *parameters_[parameter_indices].state...);
      }...};
};

}  // namespace serac
