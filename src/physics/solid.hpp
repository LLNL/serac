// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solid_solver.hpp
 *
 * @brief The solver object for finite deformation hyperelasticity
 */

#ifndef NONLIN_SOLID
#define NONLIN_SOLID

#include "infrastructure/input.hpp"
#include "mfem.hpp"
#include "physics/base_physics.hpp"
#include "physics/operators/odes.hpp"
#include "physics/operators/stdfunction_operator.hpp"

namespace serac {

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear hyperelastic quasi-static and dynamic
 * hyperelastic solver object. It is derived from MFEM
 * example 10p.
 */
class Solid : public BasePhysics {
public:
  /**
   * @brief A timestep method and config for the M solver
   */
  struct DynamicSolverParameters {
    TimestepMethod             timestepper;
    DirichletEnforcementMethod enforcement_method;
  };
  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M params
   */
  struct SolverParameters {
    LinearSolverParameters                 H_lin_params;
    NonlinearSolverParameters              H_nonlin_params;
    std::optional<DynamicSolverParameters> dyn_params = std::nullopt;
  };

  /**
   * @brief Stores all information held in the input file that
   * is used to configure the solver
   */
  struct InputInfo {
    /**
     * @brief Input file parameters specific to this class
     *
     * @param[in] table Inlet's SchemaCreator that input files will be added to
     * @param[in] dynamic Whether the solid solver will be dynamic
     **/
    // FIXME: The dynamic parameter can be removed once the enhanced axom::inlet::Proxy::contains
    // logic is merged in
    static void defineInputFileSchema(axom::inlet::Table& table, const bool dynamic = false);

    // The order of the field
    int              order;
    SolverParameters solver_params;
    // Lame parameters
    double mu;
    double K;

    // Boundary condition information
    std::vector<input::BoundaryConditionInputInfo> boundary_conditions;
  };

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] order The order of the displacement field
   * @param[in] mesh The MFEM parallel mesh to solve on
   * @param[in] solver The system solver parameters
   */
  Solid(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverParameters& params);

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] mesh The MFEM parallel mesh to solve on
   * @param[in] info The solver information parsed from the input file
   */
  Solid(std::shared_ptr<mfem::ParMesh> mesh, const InputInfo& info);

  /**
   * @brief Set displacement boundary conditions
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef);

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   * @param[in] component The component to set the displacment on
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component);

  /**
   * @brief Set the traction boundary conditions
   *
   * @param[in] trac_bdr The set of boundary attributes to apply a traction to
   * @param[in] trac_bdr_coef The vector valued traction coefficient
   * @param[in] component The component to apply the traction on
   */
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef);

  /**
   * @brief Set the hyperelastic material parameters
   *
   * @param[in] mu Set the mu Lame parameter for the hyperelastic solid
   * @param[in] K Set the K Lame parameter for the hyperelastic solid
   */
  void setHyperelasticMaterialParameters(double mu, double K);

  /**
   * @brief Set the initial displacement value
   *
   * @param[in] disp_state The initial displacement state
   */
  void setDisplacement(mfem::VectorCoefficient& disp_state);

  /**
   * @brief Set the velocity state
   *
   * @param[in] velo_state The velocity state
   */
  void setVelocity(mfem::VectorCoefficient& velo_state);

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  const FiniteElementState& displacement() const { return displacement_; };
  FiniteElementState&       displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  const FiniteElementState& velocity() const { return velocity_; };
  FiniteElementState&       velocity() { return velocity_; };

  /**
   * @brief Complete the setup of all of the internal MFEM objects and prepare for timestepping
   */
  void completeSetup() override;

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping
   * schemes
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Destroy the Nonlinear Solid Solver object
   */
  virtual ~Solid();

protected:
  /**
   * @brief Extensible means of constructing the nonlinear quasistatic
   * operator
   *
   * @return The quasi-static operator
   */
  virtual std::unique_ptr<mfem::Operator> buildQuasistaticOperator();

  /**
   * @brief Complete a quasi-static solve
   */
  virtual void quasiStaticSolve();

  /**
   * @brief Velocity field
   */
  FiniteElementState velocity_;

  /**
   * @brief Displacement field
   */
  FiniteElementState displacement_;

  /**
   * @brief The quasi-static operator for use with the MFEM newton solvers
   */
  std::unique_ptr<mfem::Operator> residual_;

  /**
   * @brief Configuration for dynamic equation solver
   */
  std::optional<LinearSolverParameters> timedep_oper_params_;

  /**
   * @brief The time dependent operator for use with the MFEM ODE solvers
   */
  std::unique_ptr<mfem::TimeDependentOperator> timedep_oper_;

  /**
   * @brief The viscosity coefficient
   */
  std::unique_ptr<mfem::Coefficient> viscosity_;

  /**
   * @brief The hyperelastic material model
   */
  std::unique_ptr<mfem::HyperelasticModel> model_;

  /**
   * @brief Pointer to the reference mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /**
   * @brief Pointer to the deformed mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /**
   * @brief Mass matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> M_mat_;

  /**
   * @brief Damping matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> C_mat_;

  /**
   * @brief Jacobian (or "effective mass") matrix
   */
  std::unique_ptr<mfem::HypreParMatrix> J_mat_;

  /**
   * @brief Mass bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> M_;

  /**
   * @brief Damping bilinear form object
   */
  std::unique_ptr<mfem::ParBilinearForm> C_;

  /**
   * @brief Stiffness bilinear form object
   */
  std::unique_ptr<mfem::ParNonlinearForm> H_;

  /**
   * @brief zero vector of the appropriate dimensions
   */
  mfem::Vector zero_;

  /**
   * @brief Nonlinear system solver instance
   */
  EquationSolver nonlin_solver_;

  /**
   * @brief the system of ordinary differential equations for the physics module
   */
  SecondOrderODE ode2_;

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

  // current and previous timesteps
  double c0_, c1_;
};

}  // namespace serac

template <>
struct FromInlet<serac::Solid::InputInfo> {
  serac::Solid::InputInfo operator()(const axom::inlet::Table& base);
};

#endif
