// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics_contact.hpp
 *
 * @brief An object containing the solver for total Lagrangian finite deformation solid mechanics with contact
 */

#pragma once

#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/contact/contact_data.hpp"

namespace serac {

template <int order, int dim, typename parameters = Parameters<>,
          typename parameter_indices = std::make_integer_sequence<int, parameters::n>>
class SolidMechanicsContact;

/**
 * @brief The nonlinear solid with contact solver class
 *
 * The nonlinear total Lagrangian quasi-static with contact solver object. This uses Functional to compute
 * the tangent stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement field
 * @tparam dim The spatial dimension of the mesh
 */
template <int order, int dim, typename... parameter_space, int... parameter_indices>
class SolidMechanicsContact<order, dim, Parameters<parameter_space...>,
                            std::integer_sequence<int, parameter_indices...>>
    : public SolidMechanics<order, dim, Parameters<parameter_space...>,
                            std::integer_sequence<int, parameter_indices...>> {
  using SolidMechanicsBase =
      SolidMechanics<order, dim, Parameters<parameter_space...>, std::integer_sequence<int, parameter_indices...>>;

public:
  /**
   * @brief Construct a new SolidMechanicsContact object
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
   */
  SolidMechanicsContact(const NonlinearSolverOptions nonlinear_opts, const LinearSolverOptions lin_opts,
                        const serac::TimesteppingOptions timestepping_opts, const GeometricNonlinearities geom_nonlin,
                        const std::string& physics_name, std::string mesh_tag,
                        std::vector<std::string> parameter_names = {}, int cycle = 0, double time = 0.0)
      : SolidMechanicsContact(
            std::make_unique<EquationSolver>(nonlinear_opts, lin_opts, StateManager::mesh(mesh_tag).GetComm()),
            timestepping_opts, geom_nonlin, physics_name, mesh_tag, parameter_names, cycle, time)
  {
  }

  /**
   * @brief Construct a new SolidMechanicsContact object
   *
   * @param solver The nonlinear equation solver for the implicit solid mechanics equations
   * @param timestepping_opts The timestepping options for the solid mechanics time evolution operator
   * @param geom_nonlin Flag to include geometric nonlinearities
   * @param physics_name A name for the physics module instance
   * @param mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param parameter_names A vector of the names of the requested parameter fields
   * @param cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param time The simulation time to initialize the physics module to
   */
  SolidMechanicsContact(std::unique_ptr<serac::EquationSolver> solver,
                        const serac::TimesteppingOptions timestepping_opts, const GeometricNonlinearities geom_nonlin,
                        const std::string& physics_name, std::string mesh_tag,
                        std::vector<std::string> parameter_names = {}, int cycle = 0, double time = 0.0)
      : SolidMechanicsBase(std::move(solver), timestepping_opts, geom_nonlin, physics_name, mesh_tag, parameter_names,
                           cycle, time),
        contact_(mesh_),
        forces_(StateManager::newDual(displacement_.space(), detail::addPrefix(physics_name, "contact_forces")))
  {
    forces_ = 0;
    duals_.push_back(&forces_);
  }

  /**
   * @brief Construct a new Nonlinear SolidMechanicsContact Solver object
   *
   * @param[in] input_options The solver information parsed from the input file
   * @param[in] physics_name A name for the physics module instance
   * @param[in] mesh_tag The tag for the mesh in the StateManager to construct the physics module on
   * @param[in] cycle The simulation cycle (i.e. timestep iteration) to intialize the physics module to
   * @param[in] time The simulation time to initialize the physics module to
   */
  SolidMechanicsContact(const SolidMechanicsInputOptions& input_options, const std::string& physics_name,
                        std::string mesh_tag, int cycle = 0, double time = 0.0)
      : SolidMechanicsBase(input_options, physics_name, mesh_tag, cycle, time),
        contact_(mesh_),
        forces_(StateManager::newDual(displacement_.space(), detail::addPrefix(physics_name, "contact_forces")))
  {
    forces_ = 0;
    duals_.push_back(&forces_);
  }

  /// @brief Build the quasi-static operator corresponding to the total Lagrangian formulation
  std::unique_ptr<mfem_ext::StdFunctionOperator> buildQuasistaticOperator() override
  {
    auto residual_fn = [this](const mfem::Vector& u, mfem::Vector& r) {
      const mfem::Vector u_blk(const_cast<mfem::Vector&>(u), 0, displacement_.Size());
      const mfem::Vector res = (*residual_)(ode_time_point_, shape_displacement_, u_blk, acceleration_,
                                            *parameters_[parameter_indices].state...);

      // TODO this copy is required as the sundials solvers do not allow move assignments because of their memory
      // tracking strategy
      // See https://github.com/mfem/mfem/issues/3531
      mfem::Vector r_blk(r, 0, displacement_.Size());
      r_blk = res;
      contact_.residualFunction(u, r);
      r_blk.SetSubVector(bcs_.allEssentialTrueDofs(), 0.0);
    };
    // This if-block below breaks up building the Jacobian operator depending if there is Lagrange multiplier
    // enforcement or not
    if (contact_.haveLagrangeMultipliers()) {
      // The quasistatic operator has blocks if any of the contact interactions are enforced using Lagrange multipliers.
      // Jacobian operator is an mfem::BlockOperator
      J_offsets_ = mfem::Array<int>({0, displacement_.Size(), displacement_.Size() + contact_.numPressureDofs()});
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize() + contact_.numPressureDofs(), residual_fn,
          // gradient of residual function
          [this](const mfem::Vector& u) -> mfem::Operator& {
            const mfem::Vector u_blk(const_cast<mfem::Vector&>(u), 0, displacement_.Size());
            auto [r, drdu] = (*residual_)(ode_time_point_, shape_displacement_, differentiate_wrt(u_blk), acceleration_,
                                          *parameters_[parameter_indices].state...);
            J_             = assemble(drdu);

            // create block operator holding jacobian contributions
            J_constraint_ = contact_.jacobianFunction(u, J_.release());

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
      // If all of the contact interactions are penalty, then there will be no blocks.  Jacobian operator is a single
      // mfem::HypreParMatrix
      return std::make_unique<mfem_ext::StdFunctionOperator>(
          displacement_.space().TrueVSize(), residual_fn, [this](const mfem::Vector& u) -> mfem::Operator& {
            auto [r, drdu] = (*residual_)(ode_time_point_, shape_displacement_, differentiate_wrt(u), acceleration_,
                                          *parameters_[parameter_indices].state...);
            J_             = assemble(drdu);

            // get 11-block holding jacobian contributions
            auto block_J         = contact_.jacobianFunction(u, J_.release());
            block_J->owns_blocks = false;
            J_ = std::unique_ptr<mfem::HypreParMatrix>(static_cast<mfem::HypreParMatrix*>(&block_J->GetBlock(0, 0)));

            J_e_ = bcs_.eliminateAllEssentialDofsFromMatrix(*J_);

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
    SLIC_ERROR_ROOT_IF(order > 1, "Contact can only be applied to linear (order = 1) meshes.");
    contact_.addContactInteraction(interaction_id, bdry_attr_surf1, bdry_attr_surf2, contact_opts);
  }

  /**
   * @brief Complete the initialization and allocation of the data structures.
   *
   * @note This must be called before AdvanceTimestep().
   */
  void completeSetup() override
  {
    double dt = 0.0;
    contact_.update(cycle_, time_, dt);

    SolidMechanicsBase::completeSetup();
  }

protected:
  /// @brief Solve the Quasi-static Newton system
  void quasiStaticSolve(double dt) override
  {
    // we can use the base class method if we don't have Lagrange multipliers
    if (!contact_.haveLagrangeMultipliers()) {
      SolidMechanicsBase::quasiStaticSolve(dt);
      contact_.update(cycle_, ode_time_point_, dt);
      forces_.SetVector(contact_.forces(), 0);
      return;
    }

    // this method is essentially equivalent to the 1-liner
    // u += dot(inv(J), dot(J_elim[:, dofs], (U(t + dt) - u)[dofs]));
    // warm start for contact needs to include the previous stiffness terms associated with contact
    // otherwise the system will interpenetrate instantly on warm-starting.
    warmStartDisplacement(dt);

    time_ += dt;
    // Set the ODE time point for the time-varying loads in quasi-static problems
    ode_time_point_ = time_;

    // In general, the solution vector is a stacked (block) vector:
    //  | displacement     |
    //  | contact pressure |
    // Contact pressure is only active when solving a contact problem with Lagrange multipliers.
    mfem::Vector augmented_solution(displacement_.Size() + contact_.numPressureDofs());
    augmented_solution.SetVector(displacement_, 0);
    augmented_solution.SetVector(contact_.mergedPressures(), displacement_.Size());

    // solve the non-linear system resid = 0 and pressure * gap = 0
    nonlin_solver_->solve(augmented_solution);
    displacement_.Set(1.0, mfem::Vector(augmented_solution, 0, displacement_.Size()));
    contact_.setPressures(mfem::Vector(augmented_solution, displacement_.Size(), contact_.numPressureDofs()));
    contact_.update(cycle_, ode_time_point_, dt);
    forces_.SetVector(contact_.forces(), 0);
  }

  using BasePhysics::bcs_;
  using BasePhysics::cycle_;
  using BasePhysics::duals_;
  using BasePhysics::is_quasistatic_;
  using BasePhysics::mesh_;
  using BasePhysics::name_;
  using BasePhysics::parameters_;
  using BasePhysics::shape_displacement_;
  using BasePhysics::states_;
  using BasePhysics::time_;
  using SolidMechanicsBase::acceleration_;
  using SolidMechanicsBase::d_residual_d_;
  using SolidMechanicsBase::DERIVATIVE;
  using SolidMechanicsBase::displacement_;
  using SolidMechanicsBase::du_;
  using SolidMechanicsBase::J_;
  using SolidMechanicsBase::J_e_;
  using SolidMechanicsBase::nonlin_solver_;
  using SolidMechanicsBase::ode_time_point_;
  using SolidMechanicsBase::residual_;
  using SolidMechanicsBase::residual_with_bcs_;
  using SolidMechanicsBase::warmStartDisplacement;

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

  /// forces for output
  FiniteElementDual forces_;
};

}  // namespace serac
