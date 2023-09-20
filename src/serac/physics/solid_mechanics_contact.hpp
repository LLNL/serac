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

#include "serac/physics/base_physics.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/contact/contact_data.hpp"

namespace serac {

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidMechanicsContact;

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear total Lagrangian quasi-static and dynamic hyperelastic solver object. This uses Functional to compute
 * the tangent stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidMechanicsContact<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>
    : public SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>> {
public:
  /**
   * @brief Construct a new SolidMechanicsContact object
   *
   * @param nonlinear_opts The nonlinear solver options for solving the nonlinear residual equations
   * @param lin_opts The linear solver options for solving the linearized Jacobian equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param name An optional name for the physics module instance
   * @param pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  SolidMechanicsContact(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
                 const serac::TimesteppingOptions timestepping_opts,
                 const GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On, const std::string& name = "",
                 mfem::ParMesh* pmesh = nullptr)
      : SolidMechanicsContact(std::make_unique<EquationSolver>(
                           nonlinear_opts, lin_opts, StateManager::mesh(StateManager::collectionID(pmesh)).GetComm()),
                       timestepping_opts, geom_nonlin, name, pmesh)
  {
  }

  /**
   * @brief Construct a new SolidMechanicsContact object
   *
   * @param solver The nonlinear equation solver for the implicit solid mechanics equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param name An optional name for the physics module instance
   * @param pmesh The mesh to conduct the simulation on, if different than the default mesh
   */
  SolidMechanicsContact(std::unique_ptr<serac::EquationSolver> solver, const serac::TimesteppingOptions timestepping_opts,
                 const GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On, const std::string& name = "",
                 mfem::ParMesh* pmesh = nullptr)
      : SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>(std::move(solver), timestepping_opts, geom_nonlin, name, pmesh),
      contact_(mesh_)
  {
  }

  /**
   * @brief Construct a new Nonlinear SolidMechanicsContact Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] name An optional name for the physics module instance. Note that this is NOT the mesh tag.
   */
  SolidMechanicsContact(const SolidMechanicsInputOptions& input_options, const std::string& name = "")
      : SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>(input_options, name),
      contact_(mesh_)
  {
  }

  /// @brief Build the quasi-static operator corresponding to the total Lagrangian formulation
  std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator() override
  {
    // the quasistatic case is entirely described by the residual,
    // there is no ordinary differential equation
    std::function<void(const mfem::Vector&, mfem::Vector&)> residual_fn = [this](const mfem::Vector& u,
                                                                                 mfem::Vector&       r) {
      const mfem::Vector res = (*residual_)(u, zero_, shape_displacement_, *parameters_[parameter_indices].state...);

      // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
      // tracking strategy
      // See https://github.com/mfem/mfem/issues/3531
      r = res;
    };
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const mfem::Vector&)> jacobian_fn =
        [this](const mfem::Vector& u) -> std::unique_ptr<mfem::HypreParMatrix> {
      auto [r, drdu] =
          (*residual_)(differentiate_wrt(u), zero_, shape_displacement_, *parameters_[parameter_indices].state...);
      auto J = assemble(drdu);
      return J;
    };

    // add contact contribution to residual
    if (contact_.haveContactInteractions()) {
      residual_fn = contact_.residualFunction(residual_fn);
    }
    // process dirichlet bcs for residual (same for contact/non-contact)
    residual_fn = [this, residual_fn](const mfem::Vector& u, mfem::Vector& r) {
      residual_fn(u, r);
      r.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
    };

    // Lagrange multiplier contact returns a block jacobian and non-contact/penalty contact returns the jacobian as a
    // hypre par matrix.  also, bcs need to be applied to the contact blocks.
    if (contact_.haveLagrangeMultipliers()) {
      J_offsets_ = mfem::Array<int>({0, displacement_.Size(), displacement_.Size() + contact_.numPressureDofs()});
      // add the contact contribution to the jacobian
      auto block_jacobian_fn = contact_.jacobianFunction(jacobian_fn);
      // apply dirichlet bcs
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize() + contact_.numPressureDofs(),

          // residual function
          residual_fn,

          // gradient of residual function
          [this, block_jacobian_fn](const mfem::Vector& u) -> mfem::Operator& {
            // create block operator holding jacobian contributions
            J_constraint_ = block_jacobian_fn(u);

            // take ownership of blocks
            J_constraint_->owns_blocks = false;
            J_                         = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 0)));
            J_12_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(0, 1)));
            J_21_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 0)));
            J_22_ = std::unique_ptr<mfem::HypreParMatrix>(
                static_cast<mfem::HypreParMatrix*>(&J_constraint_->GetBlock(1, 1)));

            // eliminate bcs and compute eliminated blocks
            J_e_    = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            J_e_21_ = std::unique_ptr<mfem::HypreParMatrix>(J_21_->EliminateCols(bcs_.allEssentialTrueDofs()));
            J_12_->EliminateRows(bcs_.allEssentialTrueDofs());

            // create block operator for constraints
            J_constraint_e_ = std::make_unique<mfem::BlockOperator>(J_offsets_);
            J_constraint_e_->SetBlock(0, 0, J_e_.get());
            J_constraint_e_->SetBlock(1, 0, J_e_21_.get());

            J_operator_ = J_constraint_.get();
            return *J_constraint_;
          });
    } else {
      // add contact contribution to residual and jacobian with penalty contact
      if (contact_.haveContactInteractions()) {
        auto block_jacobian_fn = contact_.jacobianFunction(jacobian_fn);
        jacobian_fn            = [block_jacobian_fn](const mfem::Vector& u) -> std::unique_ptr<mfem::HypreParMatrix> {
          auto block_J         = block_jacobian_fn(u);
          block_J->owns_blocks = false;
          return std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));
        };
      }
      // apply dirichlet bcs
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(),

          // residual function
          residual_fn,

          // gradient of residual function
          [this, jacobian_fn](const mfem::Vector& u) -> mfem::Operator& {
            J_          = jacobian_fn(u);
            J_e_        = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);
            J_operator_ = J_.get();
            return *J_;
          });
    }
  }

  /**
   * @brief Add a mortar contact boundary condition
   *
   * @param interaction_id Unique identifier for the ContactInteraction
   * @param bdry_attr_surf1 MFEM boundary attributes for the first surface
   * @param bdry_attr_surf2 MFEM boundary attributes for the second surface
   * @param contact_opts Defines contact method, enforcement, type, and penalty
   */
  void addContactInteraction(int interaction_id, const std::set<int>& bdry_attr_surf1,
                             const std::set<int>& bdry_attr_surf2, ContactOptions contact_opts)
  {
    SLIC_ERROR_ROOT_IF(!is_quasistatic_, "Contact can only be applied to quasistatic problems.");
    SLIC_ERROR_ROOT_IF(order_ > 1, "Contact can only be applied to linear (order = 1) meshes.");
    contact_.addContactInteraction(interaction_id, bdry_attr_surf1, bdry_attr_surf2, contact_opts);
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    int    cycle = 0;
    double time  = 0.0;
    double dt    = 0.0;
    contact_.update(cycle, time, dt);
    
    SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::completeSetup();
  }

  /// @brief Solve the Quasi-static Newton system
  void quasiStaticSolve(double dt) override
  {
    time_ += dt;

    // the ~85 lines of code below are essentially equivalent to the 1-liner
    // u += dot(inv(J), dot(J_elim[:, dofs], (U(t + dt) - u)[dofs]));
    // or, with Lagrange multiplier contact enforcement
    // [u; p] += dot(inv(block_J), dot(block_J_elim[:, dofs], (U(t + dt) - u)[dofs]))
    // where block_J = | J  B^T |
    //                 | B   0  |

    // Update the linearized Jacobian matrix
    // In general, the solution vector is a stacked (block) vector:
    //  | displacement     |
    //  | contact pressure |
    // Contact pressure is only active when solving a contact problem with Lagrange multipliers.
    // The gradient is not a function of the Lagrange multipliers, so they do not need to be copied to the solution.
    // However, the solution vector must be sized to the Operator, which includes the Lagrange multipliers.
    mfem::Vector augmented_solution(displacement_.Size() + contact_.numPressureDofs());
    augmented_solution = 0.0;
    augmented_solution.SetVector(displacement_, 0);
    residual_with_bcs_->GetGradient(augmented_solution);

    du_ = 0.0;
    for (auto& bc : bcs_.essentials()) {
      bc.setDofs(du_, time_);
    }

    auto& constrained_dofs = bcs_.allEssentialTrueDofs();
    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j = constrained_dofs[i];
      du_[j] -= displacement_(j);
    }

    dr_ = 0.0;
    mfem::EliminateBC(*J_, *J_e_, constrained_dofs, du_, dr_);

    // Update the initial guess for changes in the parameters if this is not the first solve
    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
      // Compute the change in parameters parameter_diff = parameter_new - parameter_old
      serac::FiniteElementState parameter_difference = *parameters_[parameter_index].state;
      parameter_difference -= *parameters_[parameter_index].previous_state;

      // Compute a linearized estimate of the residual forces due to this change in parameter
      auto drdparam        = serac::get<SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::DERIVATIVE>(d_residual_d_[parameter_index]());
      auto residual_update = drdparam(parameter_difference);

      // Flip the sign to get the RHS of the Newton update system
      // J^-1 du = - residual
      residual_update *= -1.0;

      dr_ += residual_update;

      // Save the current parameter value for the next timestep
      *parameters_[parameter_index].previous_state = *parameters_[parameter_index].state;
    }

    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j  = constrained_dofs[i];
      dr_[j] = du_[j];
    }

    auto& lin_solver = nonlin_solver_->linearSolver();

    // J_operator_ points to a) a HypreParMatrix if no contact Lagrange multipliers are present or
    //                       b) a BlockOperator if contact Lagrange multipliers are present
    lin_solver.SetOperator(*J_operator_);

    // solve augmented_solution = (J_operator)^-1 * augmented_residual where
    // augmented_solution = du_ if no Lagrange multiplier contact, [du_; 0] otherwise
    // augmented_residual = dr_ if no Lagrange multiplier contact, [dr_; dgap = B*du_] otherwise
    augmented_solution = 0.0;
    augmented_solution.SetVector(du_, 0);

    mfem::Vector augmented_residual(augmented_solution.Size());
    augmented_residual = 0.0;
    augmented_residual.SetVector(dr_, 0);
    if (contact_.haveLagrangeMultipliers()) {
      // calculate dgap = B*du_
      mfem::Vector dgap(augmented_residual, displacement_.Size(), contact_.numPressureDofs());
      J_21_->Mult(du_, dgap);
    }
    lin_solver.Mult(augmented_residual, augmented_solution);

    // update du_, displacement_, and pressure based on linearized kinematics
    du_.Set(1.0, mfem::Vector(augmented_solution, 0, displacement_.Size()));
    displacement_ += du_;
    if (contact_.haveContactInteractions()) {
      // call update to update gaps for new displacements
      contact_.update(cycle_, time_, dt);
      // update pressures based on pressures in augmented_solution (for Lagrange multiplier) and updated gaps (for
      // penalty)
      contact_.setPressures(mfem::Vector(augmented_solution, displacement_.Size(), contact_.numPressureDofs()));
    }

    // solve the non-linear system resid = 0 and pressure * gap = 0 for Lagrange multiplier contact
    augmented_solution.SetVector(displacement_, 0);
    if (contact_.haveLagrangeMultipliers()) {
      augmented_solution.SetVector(contact_.mergedPressures(), displacement_.Size());
    }
    nonlin_solver_->solve(augmented_solution);
    displacement_.Set(1.0, mfem::Vector(augmented_solution, 0, displacement_.Size()));
    contact_.setPressures(mfem::Vector(augmented_solution, displacement_.Size(), contact_.numPressureDofs()));
  }

protected:

  using BasePhysics::is_quasistatic_;
  using BasePhysics::mesh_;
  using BasePhysics::shape_displacement_;
  using BasePhysics::parameters_;
  using BasePhysics::bcs_;
  using BasePhysics::order_;
  using BasePhysics::time_;
  using BasePhysics::cycle_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::residual_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::displacement_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::residual_with_bcs_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::du_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::dr_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::nonlin_solver_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::d_residual_d_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::zero_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::J_;
  using SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>::J_e_;

  /// Pointer to the Jacobian operator (J_ if no Lagrange multiplier contact, J_constraint_ otherwise)
  mfem::Operator* J_operator_;

  /// 21 Jacobian block if using Lagrange multiplier contact (dg/dx)
  std::unique_ptr<mfem::HypreParMatrix> J_21_;

  /// 12 Jacobian block if using Lagrange multiplier contact (df/dp)
  std::unique_ptr<mfem::HypreParMatrix> J_12_;

  /// 22 Jacobian block if using Lagrange multiplier contact (ones on diagonal for inactive t-dofs)
  std::unique_ptr<mfem::HypreParMatrix> J_22_;

  /// Block offsets for the J_constraint_ BlockOperator (must be owned outside J_constraint_)
  mfem::Array<int> J_offsets_;

  /// Assembled sparse matrix for the Jacobian with constraint blocks
  std::unique_ptr<mfem::BlockOperator> J_constraint_;

  /// Columns of J_21_ that have been separated out because they are associated with essential boundary conditions
  std::unique_ptr<mfem::HypreParMatrix> J_e_21_;

  /// rows and columns of J_constraint_ that have been separated out
  /// because are associated with essential boundary conditions
  std::unique_ptr<mfem::BlockOperator> J_constraint_e_;

  /// @brief Class holding contact constraint data
  ContactData contact_;

};

}  // namespace serac
