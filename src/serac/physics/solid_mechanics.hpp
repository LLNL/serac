// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

#include "serac/serac_config.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/physics/common.hpp"
#include "serac/physics/solid_mechanics_input.hpp"
#include "serac/physics/base_physics.hpp"
#include "serac/numerics/odes.hpp"
#include "serac/numerics/stdfunction_operator.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"

namespace serac {

namespace solid_mechanics {

namespace detail {
/**
 * @brief integrates part of the adjoint equations backward in time
 */
void adjoint_integrate(double dt_n, double dt_np1, mfem::HypreParMatrix* m_mat, mfem::HypreParMatrix* k_mat,
                       mfem::HypreParVector& disp_adjoint_load_vector, mfem::HypreParVector& velo_adjoint_load_vector,
                       mfem::HypreParVector& accel_adjoint_load_vector, mfem::HypreParVector& adjoint_displacement_,
                       mfem::HypreParVector&     implicit_sensitivity_displacement_start_of_step_,
                       mfem::HypreParVector&     implicit_sensitivity_velocity_start_of_step_,
                       BoundaryConditionManager& bcs_, mfem::Solver& lin_solver);
}  // namespace detail

/**
 * @brief default method and tolerances for solving the
 * systems of linear equations that show up in implicit
 * solid mechanics simulations
 */
const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                    .preconditioner = serac::ordering == mfem::Ordering::byVDIM
                                                                          ? Preconditioner::HypreAMG
                                                                          : Preconditioner::HypreJacobi,
                                                    .relative_tol   = 1.0e-6,
                                                    .absolute_tol   = 1.0e-16,
                                                    .max_iterations = 500,
                                                    .print_level    = 0};

/// the default direct solver option for solving the linear stiffness equations
#ifdef MFEM_USE_STRUMPACK
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::Strumpack, .print_level = 0};
#else
const LinearSolverOptions direct_linear_options = {.linear_solver = LinearSolver::SuperLU, .print_level = 0};
#endif

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
const TimesteppingOptions default_timestepping_options = {TimestepMethod::Newmark,
                                                          DirichletEnforcementMethod::RateControl};

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
  static constexpr int  SHAPE = 0;
  static constexpr auto I     = Identity<dim>();
  //! @endcond

  /// @brief The total number of non-parameter state variables (displacement, acceleration) passed to the FEM
  /// integrators
  static constexpr auto NUM_STATE_VARS = 2;

  /// @brief a container holding quadrature point data of the specified type
  /// @tparam T the type of data to store at each quadrature point
  template <typename T>
  using qdata_type = std::shared_ptr<QuadratureData<T>>;

  /**
   * @brief Construct a new SolidMechanics object
   *
   * @param nonlinear_opts The nonlinear solver options for solving the nonlinear residual equations
   * @param lin_opts The linear solver options for solving the linearized Jacobian equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param physics_name A name for the physics module instance
   * @param mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param parameter_names A vector of the names of the requested parameter fields
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   * @param checkpoint_to_disk Flag to save the transient states on disk instead of memory for transient adjoint solver
   * @param use_warm_start Flag to turn on or off the displacement warm start predictor which helps robustness for
   * large deformation problems
   *
   * @note On parallel file systems (e.g. lustre), significant slowdowns and occasional errors were observed when
   *       writing and reading the needed trainsient states to disk for adjoint solves
   */
  SolidMechanics(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
                 const serac::TimesteppingOptions timestepping_opts, const GeometricNonlinearities geom_nonlin,
                 const std::string& physics_name, std::string mesh_tag, std::vector<std::string> parameter_names = {},
                 int cycle = 0, double time = 0.0, bool checkpoint_to_disk = false, bool use_warm_start = true)
      : SolidMechanics(
            std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, StateManager::mesh(mesh_tag).GetComm()),
            timestepping_opts, geom_nonlin, physics_name, mesh_tag, parameter_names, cycle, time, checkpoint_to_disk,
            use_warm_start)
  {
  }

  /**
   * @brief Construct a new SolidMechanics object
   *
   * @param solver The nonlinear equation solver for the implicit solid mechanics equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param physics_name A name for the physics module instance
   * @param mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param parameter_names A vector of the names of the requested parameter fields
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   * @param checkpoint_to_disk Flag to save the transient states on disk instead of memory for transient adjoint solves
   * @param use_warm_start A flag to turn on or off the displacement warm start predictor which helps robustness for
   * large deformation problems
   *
   * @note On parallel file systems (e.g. lustre), significant slowdowns and occasional errors were observed when
   *       writing and reading the needed trainsient states to disk for adjoint solves
   */
  SolidMechanics(std::unique_ptr<serac::EquationSolver> solver, const serac::TimesteppingOptions timestepping_opts,
                 const GeometricNonlinearities geom_nonlin, const std::string& physics_name, std::string mesh_tag,
                 std::vector<std::string> parameter_names = {}, int cycle = 0, double time = 0.0,
                 bool checkpoint_to_disk = false, bool use_warm_start = true)
      : BasePhysics(physics_name, mesh_tag, cycle, time, checkpoint_to_disk),
        displacement_(
            StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "displacement"), mesh_tag_)),
        velocity_(StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "velocity"), mesh_tag_)),
        acceleration_(
            StateManager::newState(H1<order, dim>{}, detail::addPrefix(physics_name, "acceleration"), mesh_tag_)),
        adjoint_displacement_(StateManager::newState(
            H1<order, dim>{}, detail::addPrefix(physics_name, "adjoint_displacement"), mesh_tag_)),
        displacement_adjoint_load_(displacement_.space(), detail::addPrefix(physics_name, "displacement_adjoint_load")),
        velocity_adjoint_load_(displacement_.space(), detail::addPrefix(physics_name, "velocity_adjoint_load")),
        acceleration_adjoint_load_(displacement_.space(), detail::addPrefix(physics_name, "acceleration_adjoint_load")),
        implicit_sensitivity_displacement_start_of_step_(displacement_.space(), "total_deriv_wrt_displacement."),
        implicit_sensitivity_velocity_start_of_step_(displacement_.space(), "total_deriv_wrt_velocity."),
        reactions_(StateManager::newDual(H1<order, dim>{}, detail::addPrefix(physics_name, "reactions"), mesh_tag_)),
        reactions_adjoint_load_(reactions_.space(), "reactions_shape_sensitivity"),
        nonlin_solver_(std::move(solver)),
        ode2_(displacement_.space().TrueVSize(),
              {.time = time_, .c0 = c0_, .c1 = c1_, .u = u_, .du_dt = v_, .d2u_dt2 = acceleration_}, *nonlin_solver_,
              bcs_),
        geom_nonlin_(geom_nonlin),
        use_warm_start_(use_warm_start)
  {
    SERAC_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(mesh_.Dimension() != dim,
                       axom::fmt::format("Compile time dimension, {0}, and runtime mesh dimension, {1}, mismatch", dim,
                                         mesh_.Dimension()));

    SLIC_ERROR_ROOT_IF(!nonlin_solver_,
                       "EquationSolver argument is nullptr in SolidMechanics constructor. It is possible that it was "
                       "previously moved.");

    // Check for dynamic mode
    if (timestepping_opts.timestepper != TimestepMethod::QuasiStatic) {
      ode2_.SetTimestepper(timestepping_opts.timestepper);
      ode2_.SetEnforcementMethod(timestepping_opts.enforcement_method);
      is_quasistatic_ = false;
    } else {
      is_quasistatic_ = true;
    }

    states_.push_back(&displacement_);
    if (!is_quasistatic_) {
      states_.push_back(&velocity_);
      states_.push_back(&acceleration_);
    }

    adjoints_.push_back(&adjoint_displacement_);
    duals_.push_back(&reactions_);
    dual_adjoints_.push_back(&reactions_adjoint_load_);

    // Create a pack of the primal field and parameter finite element spaces
    mfem::ParFiniteElementSpace* test_space  = &displacement_.space();
    mfem::ParFiniteElementSpace* shape_space = &shape_displacement_.space();

    std::array<const mfem::ParFiniteElementSpace*, NUM_STATE_VARS + sizeof...(parameter_space)> trial_spaces;
    trial_spaces[0] = &displacement_.space();
    trial_spaces[1] = &displacement_.space();

    SLIC_ERROR_ROOT_IF(
        sizeof...(parameter_space) != parameter_names.size(),
        axom::fmt::format("{} parameter spaces given in the template argument but {} parameter names were supplied.",
                          sizeof...(parameter_space), parameter_names.size()));

    if constexpr (sizeof...(parameter_space) > 0) {
      tuple<parameter_space...> types{};
      for_constexpr<sizeof...(parameter_space)>([&](auto i) {
        parameters_.emplace_back(mesh_, get<i>(types), detail::addPrefix(name_, parameter_names[i]));

        trial_spaces[i + NUM_STATE_VARS] = &(parameters_[i].state->space());
      });
    }

    residual_ = std::make_unique<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>>(
        shape_space, test_space, trial_spaces);

    // If the user wants the AMG preconditioner with a linear solver, set the pfes
    // to be the displacement
    auto* amg_prec = dynamic_cast<mfem::HypreBoomerAMG*>(&nonlin_solver_->preconditioner());
    if (amg_prec) {
      // ZRA - Iterative refinement tends to be more expensive than it is worth
      // We should add a flag allowing users to enable it

      // bool iterative_refinement = false;
      // amg_prec->SetElasticityOptions(&displacement_.space(), iterative_refinement);

      // SetElasticityOptions only works with byVDIM ordering, some evidence that it is not often optimal
      amg_prec->SetSystemsOptions(displacement_.space().GetVDim(), serac::ordering == mfem::Ordering::byNODES);
    }

    int true_size = velocity_.space().TrueVSize();

    u_.SetSize(true_size);
    v_.SetSize(true_size);
    du_.SetSize(true_size);
    predicted_displacement_.SetSize(true_size);

    shape_displacement_ = 0.0;
    initializeSolidMechanicsStates();
  }

  /**
   * @brief Construct a new Nonlinear SolidMechanics Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] physics_name A name for the physics module instance
   * @param[in] mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  SolidMechanics(const SolidMechanicsInputOptions& input_options, const std::string& physics_name, std::string mesh_tag,
                 int cycle = 0, double time = 0.0)
      : SolidMechanics(input_options.nonlin_solver_options, input_options.lin_solver_options,
                       input_options.timestepping_options, input_options.geom_nonlin, physics_name, mesh_tag, {}, cycle,
                       time)
  {
    for (auto& mat : input_options.materials) {
      if (std::holds_alternative<serac::solid_mechanics::NeoHookean>(mat)) {
        setMaterial(std::get<serac::solid_mechanics::NeoHookean>(mat));
      } else if (std::holds_alternative<serac::solid_mechanics::LinearIsotropic>(mat)) {
        setMaterial(std::get<serac::solid_mechanics::LinearIsotropic>(mat));
      } else if (std::holds_alternative<serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::LinearHardening>>(
                     mat)) {
        if constexpr (dim == 3) {
          serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::LinearHardening>::State initial_state{};
          setMaterial(std::get<serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::LinearHardening>>(mat),
                      createQuadratureDataBuffer(initial_state));
        } else {
          SLIC_ERROR_ROOT("J2 materials only work for 3D simulations");
        }
      } else if (std::holds_alternative<
                     serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::PowerLawHardening>>(mat)) {
        if constexpr (dim == 3) {
          serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::PowerLawHardening>::State initial_state{};
          setMaterial(std::get<serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::PowerLawHardening>>(mat),
                      createQuadratureDataBuffer(initial_state));
        } else {
          SLIC_ERROR_ROOT("J2 materials only work for 3D simulations");
        }
      } else if (std::holds_alternative<serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::VoceHardening>>(
                     mat)) {
        if constexpr (dim == 3) {
          serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::VoceHardening>::State initial_state{};
          setMaterial(std::get<serac::solid_mechanics::J2SmallStrain<serac::solid_mechanics::VoceHardening>>(mat),
                      createQuadratureDataBuffer(initial_state));
        } else {
          SLIC_ERROR_ROOT("J2 materials only work for 3D simulations");
        }
      } else {
        SLIC_ERROR("Invalid material type.");
      }
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
        SLIC_WARNING_ROOT("Ignoring boundary condition with unknown name: " << bc_name);
      }
    }
  }

  /// @brief Destroy the SolidMechanics Functional object
  virtual ~SolidMechanics() {}

  /**
   * @brief Non virtual method to reset thermal states to zero.  This does not reset design parameters or shape.
   *
   */
  void initializeSolidMechanicsStates()
  {
    c0_ = 0.0;
    c1_ = 0.0;

    time_end_step_ = 0.0;

    displacement_ = 0.0;
    velocity_     = 0.0;
    acceleration_ = 0.0;

    adjoint_displacement_      = 0.0;
    displacement_adjoint_load_ = 0.0;
    velocity_adjoint_load_     = 0.0;
    acceleration_adjoint_load_ = 0.0;

    implicit_sensitivity_displacement_start_of_step_ = 0.0;
    implicit_sensitivity_velocity_start_of_step_     = 0.0;

    reactions_ = 0.0;

    u_                      = 0.0;
    v_                      = 0.0;
    du_                     = 0.0;
    predicted_displacement_ = 0.0;

    if (checkpoint_to_disk_) {
      outputStateToDisk();
    } else {
      checkpoint_states_.clear();
      auto state_names = stateNames();
      for (const auto& state_name : state_names) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }
  }

  /// @overload
  void resetStates(int cycle = 0, double time = 0.0) override
  {
    BasePhysics::initializeBasePhysicsStates(cycle, time);
    initializeSolidMechanicsStates();
  }

  /**
   * @brief Create a shared ptr to a quadrature data buffer for the given material type
   *
   * @tparam T the type to be created at each quadrature point
   * @param initial_state the value to be broadcast to each quadrature point
   * @return std::shared_ptr< QuadratureData<T> >
   */
  template <typename T>
  qdata_type<T> createQuadratureDataBuffer(T initial_state)
  {
    return StateManager::newQuadratureDataBuffer(mesh_tag_, order, dim, initial_state);
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes from the mesh on which to enforce a displacement
   * @param[in] disp The prescribed boundary displacement function
   *
   * @note This method must be called prior to completeSetup()
   *
   * For the displacement function, the first argument is the input position and the second argument is the output
   * prescribed displacement.
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
  {
    // Project the coefficient onto the grid function
    disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_.addEssential(disp_bdr, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set essential displacement boundary conditions (strongly enforced)
   *
   * @param[in] disp_bdr The boundary attributes from the mesh on which to enforce a displacement
   * @param[in] disp The prescribed boundary displacement function
   *
   * For the displacement function, the first argument is the input position, the second argument is the time, and the
   * third argument is the output prescribed displacement.
   *
   * @note This method must be called prior to completeSetup()
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
   *
   * For the displacement function, the argument is the input position and the output is the value of the component of
   * the displacement.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::function<double(const mfem::Vector& x)> disp,
                          int component)
  {
    // Project the coefficient onto the grid function
    component_disp_bdr_coef_ = std::make_shared<mfem::FunctionCoefficient>(disp);

    bcs_.addEssential(disp_bdr, component_disp_bdr_coef_, displacement_.space(), component);
  }

  /**
   * @brief Set the displacement essential boundary conditions on a set of true degrees of freedom
   *
   * @param true_dofs A set of true degrees of freedom to set the displacement on
   * @param disp The vector function containing the prescribed displacement values
   *
   * The @a true_dofs list can be determined using functions from the @a mfem::ParFiniteElementSpace related to the
   * displacement @a serac::FiniteElementState .
   *
   * For the displacement function, the first argument is the input position, the second argument is time,
   * and the third argument is the prescribed output displacement vector.
   *
   * @note The displacement function is required to be vector-valued. However, only the dofs specified in the @a
   * true_dofs array will be set. This means that if the @a true_dofs array only contains dofs for a specific vector
   * component in a vector-valued finite element space, only that component will be set.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCsByDofList(const mfem::Array<int>                                          true_dofs,
                                   std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
  {
    disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_.addEssential(true_dofs, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set the displacement essential boundary conditions on a set of true degrees of freedom
   *
   * @param true_dofs A set of true degrees of freedom to set the displacement on
   * @param disp The vector function containing the prescribed displacement values
   *
   * The @a true_dofs list can be determined using functions from the @a mfem::ParFiniteElementSpace class.
   *
   * @note The coefficient is required to be vector-valued. However, only the dofs specified in the @a true_dofs
   * array will be set. This means that if the @a true_dofs array only contains dofs for a specific vector component in
   * a vector-valued finite element space, only that component will be set.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCsByDofList(const mfem::Array<int>                                  true_dofs,
                                   std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
  {
    disp_bdr_coef_ = std::make_shared<mfem::VectorFunctionCoefficient>(dim, disp);

    bcs_.addEssential(true_dofs, disp_bdr_coef_, displacement_.space());
  }

  /**
   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area
   *
   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
   * be constrained by this boundary condition
   * @param disp The vector function containing the prescribed displacement values
   *
   * The displacement function takes a spatial position as the first argument and time as the second argument. It
   * computes the desired displacement and fills the third argument with these displacement values.
   *
   * @note This method searches over the entire mesh, not just the boundary nodes.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>                        is_node_constrained,
                          std::function<void(const mfem::Vector&, double, mfem::Vector&)> disp)
  {
    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained);

    setDisplacementBCsByDofList(constrained_dofs, disp);
  }

  /**
   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area
   *
   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
   * be constrained by this boundary condition
   * @param disp The vector function containing the prescribed displacement values
   *
   * The displacement function takes a spatial position as the first argument. It computes the desired displacement
   * and fills the second argument with these displacement values.
   *
   * @note This method searches over the entire mesh, not just the boundary nodes.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>                is_node_constrained,
                          std::function<void(const mfem::Vector&, mfem::Vector&)> disp)
  {
    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained);

    setDisplacementBCsByDofList(constrained_dofs, disp);
  }

  /**
   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area for a single
   * displacement vector component
   *
   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
   * be constrained by this boundary condition
   * @param disp The scalar function containing the prescribed component displacement values
   * @param component The component of the displacement vector that should be set by this boundary condition. The other
   * components of displacement are unconstrained.
   *
   * The displacement function takes a spatial position as the first argument and current time as the second argument.
   * It computes the desired displacement scalar for the given component and returns that value.
   *
   * @note This method searches over the entire mesh, not just the boundary nodes.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(std::function<bool(const mfem::Vector&)>           is_node_constrained,
                          std::function<double(const mfem::Vector&, double)> disp, int component)
  {
    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained, component);

    auto vector_function = [disp, component](const mfem::Vector& x, double time, mfem::Vector& displacement) {
      displacement            = 0.0;
      displacement(component) = disp(x, time);
    };

    setDisplacementBCsByDofList(constrained_dofs, vector_function);
  }

  /**
   * @brief Set the displacement boundary conditions on a set of nodes within a spatially-defined area for a single
   * displacement vector component
   *
   * @param is_node_constrained A callback function that returns true if displacement nodes at a certain position should
   * be constrained by this boundary condition
   * @param disp The scalar function containing the prescribed component displacement values
   * @param component The component of the displacement vector that should be set by this boundary condition. The other
   * components of displacement are unconstrained.
   *
   * The displacement function takes a spatial position as an argument. It computes the desired displacement scalar for
   * the given component and returns that value.
   *
   * @note This method searches over the entire mesh, not just the boundary nodes.
   *
   * @note This method must be called prior to completeSetup()
   */
  void setDisplacementBCs(std::function<bool(const mfem::Vector& x)>   is_node_constrained,
                          std::function<double(const mfem::Vector& x)> disp, int component)
  {
    auto constrained_dofs = calculateConstrainedDofs(is_node_constrained, component);

    auto vector_function = [disp, component](const mfem::Vector& x, mfem::Vector& displacement) {
      displacement            = 0.0;
      displacement(component) = disp(x);
    };

    setDisplacementBCsByDofList(constrained_dofs, vector_function);
  }

  /// @overload
  const FiniteElementState& state(const std::string& state_name) const override
  {
    if (state_name == "displacement") {
      return displacement_;
    } else if (state_name == "velocity") {
      return velocity_;
    } else if (state_name == "acceleration") {
      return acceleration_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("State '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return displacement_;
  }

  /**
   * @brief Set the primal solution field (displacement, velocity) for the underlying solid mechanics solver
   *
   * @param state_name The name of the field to initialize ("displacement", or "velocity")
   * @param state The finite element state vector containing the values for either the displacement or velocity fields
   *
   * It is expected that @a state has the same underlying finite element space and mesh as the selected primal solution
   * field.
   */
  void setState(const std::string& state_name, const FiniteElementState& state) override
  {
    if (state_name == "displacement") {
      displacement_ = state;
      if (!checkpoint_to_disk_) {
        checkpoint_states_["displacement"][static_cast<size_t>(cycle_)] = displacement_;
      }
      return;
    } else if (state_name == "velocity") {
      velocity_ = state;
      if (!checkpoint_to_disk_) {
        checkpoint_states_["velocity"][static_cast<size_t>(cycle_)] = velocity_;
      }
      return;
    }

    SLIC_ERROR_ROOT(axom::fmt::format(
        "setState for state named '{}' requested from solid mechanics module '{}', but it doesn't exist", state_name,
        name_));
  }

  /// @overload
  std::vector<std::string> stateNames() const override
  {
    if (is_quasistatic_) {
      return std::vector<std::string>{"displacement"};
    } else {
      return std::vector<std::string>{"displacement", "velocity", "acceleration"};
    }
  }

  /**
   * @brief register a custom boundary integral calculation as part of the residual
   *
   * @tparam active_parameters a list of indices, describing which parameters to pass to the q-function
   * @param qfunction a callable that returns the traction on a boundary surface
   * @param optional_domain The domain over which the boundary integral is evaluated. If nothing is supplied the entire
   * boundary is used.
   * ~~~ {.cpp}
   *
   *  solid_mechanics.addCustomBoundaryIntegral(DependsOn<>{}, [](double t, auto position, auto displacement, auto
   * acceleration, auto shape){ auto [X, dX_dxi] = position;
   *
   *     auto [u, du_dxi] = displacement;
   *     auto f           = u * 3.0 (X[0] < 0.01);
   *     return f;  // define a displacement-proportional traction at a given support
   *  });
   *
   * ~~~
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename callable>
  void addCustomBoundaryIntegral(DependsOn<active_parameters...>, callable qfunction,
                                 const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
                                   qfunction, domain);
  }

  /// @overload
  std::vector<std::string> adjointNames() const override { return std::vector<std::string>{{"displacement"}}; }

  /// @overload
  const FiniteElementState& adjoint(const std::string& state_name) const override
  {
    if (state_name == "displacement") {
      return adjoint_displacement_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("adjoint '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      state_name, name_));
    return adjoint_displacement_;
  }

  /// @overload
  std::vector<std::string> dualNames() const override { return std::vector<std::string>{{"reactions"}}; }

  /// @overload
  const FiniteElementDual& dual(const std::string& dual_name) const override
  {
    if (dual_name == "reactions") {
      return reactions_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format("dual '{}' requested from solid mechanics module '{}', but it doesn't exist",
                                      dual_name, name_));
    return reactions_;
  }

  /// @overload
  const FiniteElementDual& dualAdjoint(const std::string& dual_name) const override
  {
    if (dual_name == "reactions") {
      return reactions_adjoint_load_;
    }

    SLIC_ERROR_ROOT(axom::fmt::format(
        "dualAdjoint '{}' requested from solid mechanics module '{}', but it doesn't exist", dual_name, name_));
    return reactions_adjoint_load_;
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
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename callable, typename StateType = Nothing>
  void addCustomDomainIntegral(DependsOn<active_parameters...>, callable qfunction,
                               qdata_type<StateType> qdata = NoQData)
  {
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{}, qfunction,
                                 mesh_, qdata);
  }

  /**
   * @brief Functor representing a material stress.  A functor is used here instead of an
   * extended, generic lambda for compatibility with NVCC.
   */
  template <typename Material>
  struct MaterialStressFunctor {
    /// @brief Constructor for the functor
    MaterialStressFunctor(Material material, GeometricNonlinearities gn) : material_(material), geom_nonlin_(gn) {}

    /// @brief Material model
    Material material_;

    /// @brief Enum value for geometric nonlinearities
    GeometricNonlinearities geom_nonlin_;

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
      auto du_dX   = get<DERIVATIVE>(displacement);
      auto d2u_dt2 = get<VALUE>(acceleration);

      auto stress = material_(state, du_dX, params...);

      auto dx_dX = 0.0 * du_dX + I;

      if (geom_nonlin_ == GeometricNonlinearities::On) {
        dx_dX += du_dX;
      }

      auto flux = dot(stress, transpose(inv(dx_dX))) * det(dx_dX);

      return serac::tuple{material_.density * d2u_dt2, flux};
    }
  };

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
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename MaterialType, typename StateType = Empty>
  void setMaterial(DependsOn<active_parameters...>, const MaterialType& material,
                   qdata_type<StateType> qdata = EmptyQData)
  {
    static_assert(std::is_same_v<StateType, Empty> || std::is_same_v<StateType, typename MaterialType::State>,
                  "invalid quadrature data provided in setMaterial()");
    MaterialStressFunctor<MaterialType> material_functor(material, geom_nonlin_);
    residual_->AddDomainIntegral(
        Dimension<dim>{},
        DependsOn<0, 1,
                  active_parameters + NUM_STATE_VARS...>{},  // the magic number "+ NUM_STATE_VARS" accounts for the
                                                             // fact that the displacement, acceleration, and shape
                                                             // fields are always-on and come first, so the `n`th
                                                             // parameter will actually be argument `n + NUM_STATE_VARS`
        std::move(material_functor), mesh_, qdata);
  }

  /// @overload
  template <typename MaterialType, typename StateType = Empty>
  void setMaterial(const MaterialType& material, std::shared_ptr<QuadratureData<StateType>> qdata = EmptyQData)
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
  }

  /// @overload
  void setDisplacement(const FiniteElementState& temp) { displacement_ = temp; }

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
  }

  /// @overload
  void setVelocity(const FiniteElementState& temp) { velocity_ = temp; }

  /**
   * @brief Functor representing a body force integrand.  A functor is necessary instead
   * of an extended, generic lambda for compatibility with NVCC.
   */
  template <typename BodyForceType>
  struct BodyForceIntegrand {
    /// @brief Body force model
    BodyForceType body_force_;
    /// @brief Constructor for the functor
    BodyForceIntegrand(BodyForceType body_force) : body_force_(body_force) {}

    /**
     * @brief Body force call
     *
     * @tparam T temperature
     * @tparam Position Spatial position type
     * @tparam Displacement displacement
     * @tparam Acceleration acceleration
     * @tparam Params variadic parameters for call
     * @param[in] t temperature
     * @param[in] position position
     * @param[in] params parameter pack
     * @return The calculated material response (tuple of volumetric heat capacity and thermal flux) for a linear
     * isotropic material
     */
    template <typename T, typename Position, typename Displacement, typename Acceleration, typename... Params>
    auto SERAC_HOST_DEVICE operator()(T t, Position position, Displacement, Acceleration, Params... params) const
    {
      return serac::tuple{-1.0 * body_force_(get<VALUE>(position), t, params...), zero{}};
    }
  };

  /**
   * @brief Set the body forcefunction
   *
   * @tparam BodyForceType The type of the body force load
   * @param body_force A function describing the body force applied
   * @param optional_domain The domain over which the body force is applied. If nothing is supplied the entire domain is
   * used.
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
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename BodyForceType>
  void addBodyForce(DependsOn<active_parameters...>, BodyForceType body_force,
                    const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireDomain(mesh_);
    residual_->AddDomainIntegral(Dimension<dim>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
                                 BodyForceIntegrand<BodyForceType>(body_force), domain);
  }

  /// @overload
  template <typename BodyForceType>
  void addBodyForce(BodyForceType body_force, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    addBodyForce(DependsOn<>{}, body_force, optional_domain);
  }

  /**
   * @brief Set the traction boundary condition
   *
   * @tparam TractionType The type of the traction load
   * @param traction_function A function describing the traction applied to a boundary
   * @param optional_domain The domain over which the traction is applied. If nothing is supplied the entire boundary is
   * used.
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
   * @note This traction is applied in the reference (undeformed) configuration.
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename TractionType>
  void setTraction(DependsOn<active_parameters...>, TractionType traction_function,
                   const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
        [traction_function](double t, auto X, auto /* displacement */, auto /* acceleration */, auto... params) {
          auto n = cross(get<DERIVATIVE>(X));

          return -1.0 * traction_function(get<VALUE>(X), normalize(n), t, params...);
        },
        domain);
  }

  /// @overload
  template <typename TractionType>
  void setTraction(TractionType traction_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setTraction(DependsOn<>{}, traction_function, optional_domain);
  }

  /**
   * @brief Set the pressure boundary condition
   *
   * @tparam PressureType The type of the pressure load
   * @param pressure_function A function describing the pressure applied to a boundary
   * @param optional_domain The domain over which the pressure is applied. If nothing is supplied the entire boundary is
   * used.
   * @pre PressureType must be a object that can be called with the following arguments:
   *    1. `tensor<T,dim> x` the reference configuration spatial coordinates for the quadrature point
   *    2. `double t` the time (note: time will be handled differently in the future)
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @note This pressure is applied in the deformed (current) configuration if GeometricNonlinearities are on.
   *
   * @note This method must be called prior to completeSetup()
   */
  template <int... active_parameters, typename PressureType>
  void setPressure(DependsOn<active_parameters...>, PressureType pressure_function,
                   const std::optional<Domain>& optional_domain = std::nullopt)
  {
    Domain domain = (optional_domain) ? *optional_domain : EntireBoundary(mesh_);

    residual_->AddBoundaryIntegral(
        Dimension<dim - 1>{}, DependsOn<0, 1, active_parameters + NUM_STATE_VARS...>{},
        [pressure_function, geom_nonlin = geom_nonlin_](double t, auto X, auto displacement, auto /* acceleration */,
                                                        auto... params) {
          // Calculate the position and normal in the shape perturbed deformed configuration
          auto x = X + 0.0 * displacement;

          if (geom_nonlin == GeometricNonlinearities::On) {
            x = x + displacement;
          }

          auto n = cross(get<DERIVATIVE>(x));

          // serac::Functional's boundary integrals multiply the q-function output by
          // norm(cross(dX_dxi)) at that quadrature point, but if we impose a shape displacement
          // then that weight needs to be corrected. The new weight should be
          // norm(cross(dX_dxi + du_dxi + dp_dxi)) where u is displacement and p is shape displacement. This implies:
          //
          //   pressure * normalize(normal_new) * w_new
          // = pressure * normalize(normal_new) * (w_new / w_old) * w_old
          // = pressure * normalize(normal_new) * (norm(normal_new) / norm(normal_old)) * w_old
          // = pressure * (normal_new / norm(normal_new)) * (norm(normal_new) / norm(normal_old)) * w_old
          // = pressure * (normal_new / norm(normal_old)) * w_old

          // We always query the pressure function in the undeformed configuration
          return pressure_function(get<VALUE>(X), t, params...) * (n / norm(cross(get<DERIVATIVE>(X))));
        },
        domain);
  }

  /// @overload
  template <typename PressureType>
  void setPressure(PressureType pressure_function, const std::optional<Domain>& optional_domain = std::nullopt)
  {
    setPressure(DependsOn<>{}, pressure_function, optional_domain);
  }

  /// @brief Build the quasi-static operator corresponding to the total Lagrangian formulation
  virtual std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator()
  {
    // the quasistatic case is entirely described by the residual,
    // there is no ordinary differential equation
    return std::make_unique<mfem_ext::StdFunctionOperator>(
        displacement_.space().TrueVSize(),

        // residual function
        [this](const mfem::Vector& u, mfem::Vector& r) {
          SERAC_MARK_FUNCTION;
          const mfem::Vector res =
              (*residual_)(time_, shape_displacement_, u, acceleration_, *parameters_[parameter_indices].state...);

          // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
          // tracking strategy
          // See https://github.com/mfem/mfem/issues/3531
          r = res;
          r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
        },

        // gradient of residual function
        [this](const mfem::Vector& u) -> mfem::Operator& {
          SERAC_MARK_FUNCTION;
          auto [r, drdu] = (*residual_)(time_, shape_displacement_, differentiate_wrt(u), acceleration_,
                                        *parameters_[parameter_indices].state...);
          J_             = assemble(drdu);
          J_e_           = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
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

  /// @overload
  void completeSetup() override
  {
    // Build the dof array lookup tables
    displacement_.space().BuildDofToArrays();

    u_ = displacement_;
    v_ = velocity_;

    if (is_quasistatic_) {
      residual_with_bcs_ = buildQuasistaticOperator();
    } else {
      // the dynamic case is described by a residual function and a second order
      // ordinary differential equation. Here, we define the residual function in
      // terms of an acceleration.
      residual_with_bcs_ = std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          [this](const mfem::Vector& d2u_dt2, mfem::Vector& r) {
            add(1.0, u_, c0_, d2u_dt2, predicted_displacement_);
            const mfem::Vector res = (*residual_)(time_, shape_displacement_, predicted_displacement_, d2u_dt2,
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
            auto                                  K = serac::get<DERIVATIVE>((*residual_)(time_, shape_displacement_,
                                                         differentiate_wrt(predicted_displacement_), d2u_dt2,
                                                         *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

            // M := dR/da
            auto M = serac::get<DERIVATIVE>((*residual_)(time_, shape_displacement_, predicted_displacement_,
                                                         differentiate_wrt(d2u_dt2),
                                                         *parameters_[parameter_indices].state...));
            std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

            // J = M + c0 * K
            J_.reset(mfem::Add(1.0, *m_mat, c0_, *k_mat));
            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

            return *J_;
          });
    }

#ifdef SERAC_USE_PETSC
    auto* space_dep_pc =
        dynamic_cast<serac::mfem_ext::PetscPreconditionerSpaceDependent*>(&nonlin_solver_->preconditioner());
    if (space_dep_pc) {
      // This call sets the displacement ParFiniteElementSpace used to get the spatial coordinates and to
      // generate the near null space for the PCGAMG preconditioner
      space_dep_pc->SetFESpace(&displacement_.space());
    }
#endif

    nonlin_solver_->setOperator(*residual_with_bcs_);

    if (checkpoint_to_disk_) {
      outputStateToDisk();
    } else {
      checkpoint_states_.clear();
      auto state_names = stateNames();
      for (const auto& state_name : state_names) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }
  }

  /// @brief Set field to zero wherever their are essential boundary conditions applies
  void zeroEssentials(FiniteElementVector& field) const
  {
    for (const auto& essential : bcs_.essentials()) {
      field.SetSubVector(essential.getTrueDofList(), 0.0);
    }
  }

  /// @overload
  void advanceTimestep(double dt) override
  {
    SERAC_MARK_FUNCTION;
    SLIC_ERROR_ROOT_IF(!residual_, "completeSetup() must be called prior to advanceTimestep(dt) in SolidMechanics.");

    // If this is the first call, initialize the previous parameter values as the initial values
    if (cycle_ == 0) {
      for (auto& parameter : parameters_) {
        *parameter.previous_state = *parameter.state;
      }
    }

    if (is_quasistatic_) {
      quasiStaticSolve(dt);
    } else {
      // The current ode interface tracks 2 times, one internally which we have a handle to via time_,
      // and one here via the step interface.
      // We are ignoring this one, and just using the internal version for now.
      // This may need to be revisited when more complex time integrators are required,
      // but at the moment, the double times creates a lot of confusion, so
      // we short circuit the extra time here by passing a dummy time and ignoring it.
      double time_tmp = time_;
      ode2_.Step(displacement_, velocity_, time_tmp, dt);
    }

    cycle_ += 1;

    if (checkpoint_to_disk_) {
      outputStateToDisk();
    } else {
      for (const auto& state_name : stateNames()) {
        checkpoint_states_[state_name].push_back(state(state_name));
      }
    }

    {
      // after finding displacements that satisfy equilibrium,
      // compute the residual one more time, this time enabling
      // the material state buffers to be updated
      residual_->updateQdata(true);

      reactions_ = (*residual_)(time_, shape_displacement_, displacement_, acceleration_,
                                *parameters_[parameter_indices].state...);

      residual_->updateQdata(false);
    }

    if (cycle_ > max_cycle_) {
      timesteps_.push_back(dt);
      max_cycle_ = cycle_;
      max_time_  = time_;
    }
  }

  /**
   * @brief Set the loads for the adjoint reverse timestep solve
   *
   * @param loads The loads (e.g. right hand sides) for the adjoint problem
   *
   * @pre The adjoint load map is expected to contain an entry named "displacement"
   * @pre The adjoint load map may contain an entry named "velocity"
   * @pre The adjoint load map may contain an entry named "acceleration"
   *
   * These loads are typically defined as derivatives of a downstream quantity of intrest with respect
   * to a primal solution field (in this case, displacement). For this physics module, the unordered
   * map is expected to have one entry with the keys "displacement".
   *
   */
  virtual void setAdjointLoad(std::unordered_map<std::string, const serac::FiniteElementDual&> loads) override
  {
    SLIC_ERROR_ROOT_IF(loads.size() == 0, "Adjoint load container size must be greater than 0 in the solid mechanics.");

    auto disp_adjoint_load = loads.find("displacement");

    SLIC_ERROR_ROOT_IF(disp_adjoint_load == loads.end(), "Adjoint load for \"displacement\" not found.");

    displacement_adjoint_load_ = disp_adjoint_load->second;
    // Add the sign correction to move the term to the RHS
    displacement_adjoint_load_ *= -1.0;

    auto velo_adjoint_load = loads.find("velocity");

    if (velo_adjoint_load != loads.end()) {
      velocity_adjoint_load_ = velo_adjoint_load->second;
      // Add the sign correction to move the term to the RHS
      velocity_adjoint_load_ *= -1.0;
    }

    auto accel_adjoint_load = loads.find("acceleration");

    if (accel_adjoint_load != loads.end()) {
      acceleration_adjoint_load_ = accel_adjoint_load->second;
      // Add the sign correction to move the term to the RHS
      acceleration_adjoint_load_ *= -1.0;
    }
  }

  /// @overload
  void reverseAdjointTimestep() override
  {
    SLIC_ERROR_ROOT_IF(cycle_ <= min_cycle_,
                       "Maximum number of adjoint timesteps exceeded! The number of adjoint timesteps must equal the "
                       "number of forward timesteps");

    cycle_--;  // cycle is now at n \in [0,N-1]

    double       dt_np1_to_np2     = getCheckpointedTimestep(cycle_ + 1);
    const double dt_n_to_np1       = getCheckpointedTimestep(cycle_);
    auto         end_step_solution = getCheckpointedStates(cycle_ + 1);

    displacement_ = end_step_solution.at("displacement");

    if (is_quasistatic_) {
      quasiStaticAdjointSolve(dt_n_to_np1);
    } else {
      SLIC_ERROR_ROOT_IF(ode2_.GetTimestepper() != TimestepMethod::Newmark,
                         "Only Newmark implemented for transient adjoint solid mechanics.");

      // Load the end of step velo, accel from the previous cycle

      velocity_     = end_step_solution.at("velocity");
      acceleration_ = end_step_solution.at("acceleration");

      // K := dR/du
      auto K = serac::get<DERIVATIVE>((*residual_)(time_, shape_displacement_, differentiate_wrt(displacement_),
                                                   acceleration_, *parameters_[parameter_indices].state...));
      std::unique_ptr<mfem::HypreParMatrix> k_mat(assemble(K));

      // M := dR/da
      auto M = serac::get<DERIVATIVE>((*residual_)(time_, shape_displacement_, displacement_,
                                                   differentiate_wrt(acceleration_),
                                                   *parameters_[parameter_indices].state...));
      std::unique_ptr<mfem::HypreParMatrix> m_mat(assemble(M));

      auto& lin_solver = nonlin_solver_->linearSolver();
      solid_mechanics::detail::adjoint_integrate(
          dt_n_to_np1, dt_np1_to_np2, m_mat.get(), k_mat.get(), displacement_adjoint_load_, velocity_adjoint_load_,
          acceleration_adjoint_load_, adjoint_displacement_, implicit_sensitivity_displacement_start_of_step_,
          implicit_sensitivity_velocity_start_of_step_, bcs_, lin_solver);
    }

    time_end_step_ = time_;
    time_ -= dt_n_to_np1;
  }

  /// @overload
  FiniteElementDual& computeTimestepSensitivity(size_t parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' requested for sensitivity."));

    auto drdparam     = serac::get<DERIVATIVE>(d_residual_d_[parameter_field](time_end_step_));
    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(adjoint_displacement_, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  }

  /// @overload
  FiniteElementDual& computeTimestepShapeSensitivity() override
  {
    auto drdshape =
        serac::get<DERIVATIVE>((*residual_)(time_end_step_, differentiate_wrt(shape_displacement_), displacement_,
                                            acceleration_, *parameters_[parameter_indices].state...));

    auto drdshape_mat = assemble(drdshape);

    drdshape_mat->MultTranspose(adjoint_displacement_, *shape_displacement_sensitivity_);

    return *shape_displacement_sensitivity_;
  }

  /// @overload
  const std::unordered_map<std::string, const serac::FiniteElementDual&> computeInitialConditionSensitivity() override
  {
    return {{"displacement", implicit_sensitivity_displacement_start_of_step_},
            {"velocity", implicit_sensitivity_velocity_start_of_step_}};
  }

  /**
   * @brief Get the displacement state
   *
   * @return A reference to the current displacement finite element state
   */
  const serac::FiniteElementState& displacement() const { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return A reference to the current velocity finite element state
   */
  const serac::FiniteElementState& velocity() const { return velocity_; };

  /**
   * @brief Get the acceleration state
   *
   * @return A reference to the current acceleration finite element state
   */
  const serac::FiniteElementState& acceleration() const { return acceleration_; };

  /// @brief getter for nodal forces (before zeroing-out essential dofs)
  const serac::FiniteElementDual& reactions() const { return reactions_; };

  /// @overload
  void computeDualAdjointLoad(const std::string&               dual_name,
                              const serac::FiniteElementState& reaction_direction) override
  {
    SLIC_ERROR_ROOT_IF(dual_name != "reactions", "Solid mechanics has reactions as its only dual");

    auto [_, drdu] = (*residual_)(time_, shape_displacement_, differentiate_wrt(displacement_), acceleration_,
                                  *parameters_[parameter_indices].state...);
    std::unique_ptr<mfem::HypreParMatrix> jacobian = assemble(drdu);
    reactions_adjoint_load_                        = 0.0;
    jacobian->MultTranspose(reaction_direction, reactions_adjoint_load_);
    setAdjointLoad({{"displacement", reactions_adjoint_load_}});
  }

  /// @overload
  const serac::FiniteElementDual& computeDualSensitivity(const serac::FiniteElementState& reaction_direction,
                                                         size_t                           parameter_field) override
  {
    SLIC_ASSERT_MSG(parameter_field < sizeof...(parameter_indices),
                    axom::fmt::format("Invalid parameter index '{}' requested for reaction sensitivity."));

    auto drdparam     = serac::get<DERIVATIVE>(d_residual_d_[parameter_field](time_end_step_));
    auto drdparam_mat = assemble(drdparam);

    drdparam_mat->MultTranspose(reaction_direction, *parameters_[parameter_field].sensitivity);

    return *parameters_[parameter_field].sensitivity;
  };

  /// @overload
  const serac::FiniteElementDual& computeDualShapeSensitivity(
      const serac::FiniteElementState& reaction_direction) override
  {
    auto drdshape =
        serac::get<DERIVATIVE>((*residual_)(time_end_step_, differentiate_wrt(shape_displacement_), displacement_,
                                            acceleration_, *parameters_[parameter_indices].state...));
    auto drdshape_mat = assemble(drdshape);
    drdshape_mat->MultTranspose(reaction_direction, *shape_displacement_sensitivity_);
    return *shape_displacement_sensitivity_;
  };

protected:
  /// The compile-time finite element trial space for displacement and velocity (H1 of order p)
  using trial = H1<order, dim>;

  /// The compile-time finite element test space for displacement and velocity (H1 of order p)
  using test = H1<order, dim>;

  /// The compile-time finite element trial space for shape displacement (H1 of order 1, nodal displacements)
  /// The choice of polynomial order for the shape sensitivity is determined in the StateManager
  using shape_trial = H1<SHAPE_ORDER, dim>;

  /// The displacement finite element state
  FiniteElementState displacement_;

  /// The velocity finite element state
  FiniteElementState velocity_;

  /// The acceleration finite element state
  FiniteElementState acceleration_;

  // In the case of transient dynamics, this is more like an adjoint_acceleration
  /// The displacement finite element adjoint state
  FiniteElementState adjoint_displacement_;

  /// The adjoint load (RHS) for the displacement adjoint system solve (downstream -dQOI/d displacement)
  FiniteElementDual displacement_adjoint_load_;

  /// The adjoint load (RHS) for the velocity adjoint system solve (downstream -dQOI/d velocity)
  FiniteElementDual velocity_adjoint_load_;

  /// The adjoint load (RHS) for the adjoint system solve (downstream -dQOI/d acceleration)
  FiniteElementDual acceleration_adjoint_load_;

  /// The total/implicit sensitivity of the qoi with respect to the start of the previous timestep's displacement
  FiniteElementDual implicit_sensitivity_displacement_start_of_step_;

  /// The total/implicit sensitivity of the qoi with respect to the start of the previous timestep's velocity
  FiniteElementDual implicit_sensitivity_velocity_start_of_step_;

  /// nodal reaction forces
  FiniteElementDual reactions_;

  /// sensitivity of qoi with respect to reaction forces
  FiniteElementDual reactions_adjoint_load_;

  /// serac::Functional that is used to calculate the residual and its derivatives
  std::unique_ptr<ShapeAwareFunctional<shape_trial, test(trial, trial, parameter_space...)>> residual_;

  /// mfem::Operator that calculates the residual after applying essential boundary conditions
  std::unique_ptr<mfem_ext::StdFunctionOperator> residual_with_bcs_;

  /// the specific methods and tolerances specified to solve the nonlinear residual equations
  std::unique_ptr<EquationSolver> nonlin_solver_;

  /**
   * @brief the ordinary differential equation that describes
   * how to solve for the second time derivative of displacement, given
   * the current displacement, velocity, and source terms
   */
  mfem_ext::SecondOrderODE ode2_;

  /// Assembled sparse matrix for the Jacobian df/du (11 block if using Lagrange multiplier contact)
  std::unique_ptr<mfem::HypreParMatrix> J_;

  /// rows and columns of J_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_;

  /// an intermediate variable used to store the predicted end-step displacement
  mfem::Vector predicted_displacement_;

  /// vector used to store the change in essential bcs between timesteps
  mfem::Vector du_;

  /// @brief used to communicate the ODE solver's predicted displacement to the residual operator
  mfem::Vector u_;

  /// @brief used to communicate the ODE solver's predicted velocity to the residual operator
  mfem::Vector v_;

  /// coefficient used to calculate predicted displacement: u_p := u + c0 * d2u_dt2
  double c0_;

  /// coefficient used to calculate predicted velocity: dudt_p := dudt + c1 * d2u_dt2
  double c1_;

  /// @brief End of step time used in reverse mode so that the time can be decremented on reverse steps
  /// @note This time is important to save to evaluate various parameter sensitivities after each reverse step
  double time_end_step_;

  /// @brief A flag denoting whether to compute geometric nonlinearities in the residual
  GeometricNonlinearities geom_nonlin_;

  /// @brief A flag denoting whether to compute the warm start for improved robustness
  bool use_warm_start_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef_;

  /// @brief Coefficient containing the essential boundary values
  std::shared_ptr<mfem::Coefficient> component_disp_bdr_coef_;

  /// @brief Array functions computing the derivative of the residual with respect to each given parameter
  /// @note This is needed so the user can ask for a specific sensitivity at runtime as opposed to it being a
  /// template parameter.
  std::array<std::function<decltype((*residual_)(DifferentiateWRT<1>{}, 0.0, shape_displacement_, displacement_,
                                                 acceleration_, *parameters_[parameter_indices].state...))(double)>,
             sizeof...(parameter_indices)>
      d_residual_d_ = {[&](double _t) {
        return (*residual_)(DifferentiateWRT<NUM_STATE_VARS + 1 + parameter_indices>{}, _t, shape_displacement_,
                            displacement_, acceleration_, *parameters_[parameter_indices].state...);
      }...};

  /// @brief Solve the Quasi-static Newton system
  virtual void quasiStaticSolve(double dt)
  {
    // warm start must be called prior to the time update so that the previous Jacobians can be used consistently
    // throughout.
    warmStartDisplacement(dt);
    time_ += dt;

    // this method is essentially equivalent to the 1-liner
    // u += dot(inv(J), dot(J_elim[:, dofs], (U(t + dt) - u)[dofs]));
    nonlin_solver_->solve(displacement_);
  }

  /// @brief Solve the Quasi-static adjoint linear
  virtual void quasiStaticAdjointSolve(double /*dt*/)
  {
    // By default, use a homogeneous essential boundary condition
    mfem::HypreParVector adjoint_essential(displacement_adjoint_load_);
    adjoint_essential = 0.0;

    auto [_, drdu] = (*residual_)(time_, shape_displacement_, differentiate_wrt(displacement_), acceleration_,
                                  *parameters_[parameter_indices].state...);
    auto jacobian  = assemble(drdu);
    auto J_T       = std::unique_ptr<mfem::HypreParMatrix>(jacobian->Transpose());

    for (const auto& bc : bcs_.essentials()) {
      bc.apply(*J_T, displacement_adjoint_load_, adjoint_essential);
    }

    auto& lin_solver = nonlin_solver_->linearSolver();
    lin_solver.SetOperator(*J_T);
    lin_solver.Mult(displacement_adjoint_load_, adjoint_displacement_);

    // Reset the equation solver to use the full nonlinear residual operator.  MRT, is this needed?
    nonlin_solver_->setOperator(*residual_with_bcs_);
  }

  /**
   * @brief Calculate a list of constrained dofs in the true displacement vector from a function that
   * returns true if a physical coordinate is in the constrained set
   *
   * @param is_node_constrained A function that takes a point in physical space and returns true if the contained
   * degrees of freedom should be constrained
   * @param component which component is constrained (uninitialized implies all components are constrained)
   * @return An array of the constrained true dofs
   */
  mfem::Array<int> calculateConstrainedDofs(std::function<bool(const mfem::Vector&)> is_node_constrained,
                                            std::optional<int>                       component = {}) const
  {
    // Get the nodal positions for the displacement vector in grid function form
    mfem::ParGridFunction nodal_positions(
        const_cast<mfem::ParFiniteElementSpace*>(&displacement_.space()));  // MRT mfem const correctness issue
    mesh_.GetNodes(nodal_positions);

    const int        num_nodes = nodal_positions.Size() / dim;
    mfem::Array<int> constrained_dofs;

    for (int i = 0; i < num_nodes; i++) {
      // Determine if this "local" node (L-vector node) is in the local true vector. I.e. ensure this node is not a
      // shared node owned by another processor
      int idof = mfem::Ordering::Map<serac::ordering>(nodal_positions.FESpace()->GetNDofs(),
                                                      nodal_positions.FESpace()->GetVDim(), i, 0);
      if (nodal_positions.ParFESpace()->GetLocalTDofNumber(idof) >= 0) {
        mfem::Vector     node_coords(dim);
        mfem::Array<int> node_dofs;
        for (int d = 0; d < dim; d++) {
          // Get the local dof number for the prescribed component
          int local_vector_dof = mfem::Ordering::Map<serac::ordering>(nodal_positions.FESpace()->GetNDofs(),
                                                                      nodal_positions.FESpace()->GetVDim(), i, d);

          // Save the spatial position for this coordinate dof
          node_coords(d) = nodal_positions(local_vector_dof);

          // Check if this component of the displacement vector is constrained
          bool is_active_component = true;
          if (component) {
            if (*component != d) {
              is_active_component = false;
            }
          }

          if (is_active_component) {
            // Add the true dof for this component to the related dof list
            node_dofs.Append(nodal_positions.ParFESpace()->GetLocalTDofNumber(local_vector_dof));
          }
        }

        // Do the user-defined spatial query to determine if these dofs should be constrained
        if (is_node_constrained(node_coords)) {
          constrained_dofs.Append(node_dofs);
        }

        // Reset the temporary container for the dofs associated with a particular node
        node_dofs.DeleteAll();
      }
    }
    return constrained_dofs;
  }

  /**
   * @brief Sets the Dirichlet BCs for the current time and computes an initial guess for parameters and displacement
   *
   * @note
   * We want to solve
   *\f$
   *r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) = 0
   *\f$
   *for $u_{n+1}$, given new values of parameters, essential b.c.s and time. The problem is that if we use $u_n$ as the
   initial guess for this new solve, most nonlinear solver algorithms will start off by linearizing at (or near) the
   initial guess. But, if the essential boundary conditions change by an amount on the order of the mesh size, then it's
   possible to invert elements and make that linearization point inadmissible (either because it converges slowly or
   that the inverted elements crash the program). *So, we need a better initial guess. This "warm start" generates a
   guess by linear extrapolation from the previous known solution:

   *\f$
   *0 = r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) \approx {r(u_n, p_n, U_n, t_n)} +  \frac{dr_n}{du} \Delta u +
   \frac{dr_n}{dp} \Delta p + \frac{dr_n}{dU} \Delta U + \frac{dr_n}{dt} \Delta t
   *\f$
   *If we assume that suddenly changing p and t will not lead to inverted elements, we can simplify the approximation to
   *\f$
   *0 = r(u_{n+1}, p_{n+1}, U_{n+1}, t_{n+1}) \approx r(u_n, p_{n+1}, U_n, t_{n+1}) +  \frac{dr_n}{du} \Delta u +
   \frac{dr_n}{dU} \Delta U
   *\f$
   *Move all the known terms to the rhs and solve for \f$\Delta u\f$,
   *\f$
   *\Delta u = - \bigg(  \frac{dr_n}{du} \bigg)^{-1} \bigg( r(u_n, p_{n+1}, U_n, t_{n+1}) + \frac{dr_n}{dU} \Delta U
   \bigg)
   *\f$
   *It is especially important to use the previously solved Jacobian in problems with material instabilities, as good
   nonlinear solvers will ensure positive definiteness at equilibrium. *Once any parameter is changed, it is no longer
   certain to be positive definite, which will cause issues for many types linear solvers.
   */
  void warmStartDisplacement(double dt)
  {
    SERAC_MARK_FUNCTION;

    du_ = 0.0;
    for (auto& bc : bcs_.essentials()) {
      // apply the future boundary conditions, but use the most recent Jacobians stiffness.
      bc.setDofs(du_, time_ + dt);
    }

    auto& constrained_dofs = bcs_.allEssentialTrueDofs();
    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j = constrained_dofs[i];
      du_[j] -= displacement_(j);
    }

    if (use_warm_start_) {
      // Update the linearized Jacobian matrix
      auto r = (*residual_)(time_ + dt, shape_displacement_, displacement_, acceleration_,
                            *parameters_[parameter_indices].state...);

      // use the most recently evaluated Jacobian
      auto [_, drdu] = (*residual_)(time_, shape_displacement_, differentiate_wrt(displacement_), acceleration_,
                                    *parameters_[parameter_indices].previous_state...);
      J_             = assemble(drdu);
      J_e_           = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

      r *= -1.0;

      mfem::EliminateBC(*J_, *J_e_, constrained_dofs, du_, r);
      for (int i = 0; i < constrained_dofs.Size(); i++) {
        int j = constrained_dofs[i];
        r[j]  = du_[j];
      }

      auto& lin_solver = nonlin_solver_->linearSolver();

      lin_solver.SetOperator(*J_);

      lin_solver.Mult(r, du_);
    }

    displacement_ += du_;
  }
};

}  // namespace serac
