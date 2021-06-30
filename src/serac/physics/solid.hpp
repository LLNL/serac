// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid.hpp
 *
 * @brief The solver object for finite deformation hyperelasticity
 */

#pragma once

#include <optional>

#include "mfem.hpp"

#include "serac/infrastructure/input.hpp"
#include "serac/physics/base_physics.hpp"
#include "serac/physics/operators/odes.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/physics/materials/hyperelastic_material.hpp"
#include "serac/physics/integrators/displacement_hyperelastic_integrator.hpp"

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
  struct TimesteppingOptions {
    /**
     * @brief The timestep method to apply
     *
     */
    TimestepMethod timestepper;

    /**
     * @brief The essential boundary condition enforcement method
     *
     */
    DirichletEnforcementMethod enforcement_method;
  };
  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M options
   */
  struct SolverOptions {
    /**
     * @brief the options for the included linear solve
     *
     */
    LinearSolverOptions H_lin_options;

    /**
     * @brief The options for the inlucded nonlinear solve
     *
     */
    NonlinearSolverOptions H_nonlin_options;

    /**
     * @brief The optional parameters for dynamic problems
     * @note If this is not included, quasi-static analysis is performed
     *
     */
    std::optional<TimesteppingOptions> dyn_options = std::nullopt;
  };

  /**
   * @brief Stores all information held in the input file that
   * is used to configure the solver
   */
  struct InputOptions {
    /**
     * @brief Input file parameters specific to this class
     *
     * @param[in] container Inlet container on which the input schema will be defined
     **/
    static void defineInputFileSchema(axom::inlet::Container& container);

    /**
     * @brief The order of the discretization
     *
     */
    int order;

    /**
     * @brief The options for the linear, nonlinear, and ODE solvers
     *
     */
    SolverOptions solver_options;

    /**
     * @brief The shear modulus
     *
     */
    double mu;

    /**
     * @brief The bulk modulus
     *
     */
    double K;

    /**
     * @brief The linear viscosity coefficient
     *
     */
    double viscosity;

    /**
     * @brief Initial density
     *
     */
    double initial_mass_density;

    /**
     * @brief Geometric nonlinearities flag
     *
     */
    GeometricNonlinearities geom_nonlin;

    /**
     * @brief Material nonlinearities flag
     *
     */
    bool material_nonlin;

    /**
     * @brief Boundary condition information
     *
     */
    std::unordered_map<std::string, input::BoundaryConditionInputOptions> boundary_conditions;

    /**
     * @brief The initial displacement
     * @note This can be used as an initialization field for dynamic problems or an initial guess
     *       for quasi-static solves
     *
     */
    std::optional<input::CoefficientInputOptions> initial_displacement;

    /**
     * @brief The initial velocity
     *
     */
    std::optional<input::CoefficientInputOptions> initial_velocity;
  };

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] order The order of the displacement field
   * @param[in] options The options for the linear, nonlinear, and ODE solves
   * @param[in] geom_nonlin Flag to include geometric nonlinearities
   * @param[in] keep_deformation Flag to keep the deformation in the underlying mesh post-destruction
   * @param[in] name An optional name for the physics module instance
   */
  Solid(int order, const SolverOptions& options, GeometricNonlinearities geom_nonlin = GeometricNonlinearities::On,
        FinalMeshOption keep_deformation = FinalMeshOption::Deformed, const std::string& name = "");

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] options The solver information parsed from the input file
   * @param[in] name An optional name for the physics module instance
   */
  Solid(const InputOptions& options, const std::string& name = "");

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
   * @param[in] compute_on_reference Flag to compute on the reference stress-free configuration vs. the deformed
   * configuration
   * @param[in] component The component to apply the traction on
   */
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      bool compute_on_reference, std::optional<int> component = {});

  /**
   * @brief Set the pressure boundary conditions
   *
   * @param[in] pres_bdr The set of boundary attributes to apply a pressure to
   * @param[in] pres_bdr_coef The scalar valued pressure coefficient
   * @param[in] compute_on_reference Flag to compute on the reference stress-free configuration vs. the deformed
   * configuration
   */
  void setPressureBCs(const std::set<int>& pres_bdr, std::shared_ptr<mfem::Coefficient> pres_bdr_coef,
                      bool compute_on_reference);

  /**
   * @brief Add body force vectors on the domain
   *
   * @param[in] ext_force_coef Add a vector-valued external force coefficient applied to the domain
   */
  void addBodyForce(std::shared_ptr<mfem::VectorCoefficient> ext_force_coef);

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef);

  /**
   * @brief Set the mass density coefficient
   *
   * @param[in] rho_coef The mass density coefficient
   */
  void setMassDensity(std::unique_ptr<mfem::Coefficient>&& rho_coef);

  /**
   * @brief Set the material parameters
   *
   * @param[in] mu Set the shear modulus for the solid
   * @param[in] K Set the bulk modulus for the solid
   * @param[in] material_nonlin Flag to include material nonlinearities (linear elastic vs. neo-Hookean model)
   */
  void setMaterialParameters(std::unique_ptr<mfem::Coefficient>&& mu, std::unique_ptr<mfem::Coefficient>&& K,
                             bool material_nonlin = true);

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
   * @brief Reset the underlying state to the reference configuration with zero velocity
   */
  void resetToReferenceConfiguration();

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  const FiniteElementState& displacement() const { return displacement_; };

  /**
   * @overload
   */
  FiniteElementState& displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  const FiniteElementState& velocity() const { return velocity_; };

  /**
   * @overload
   */
  FiniteElementState& velocity() { return velocity_; };

  /**
   * @brief Get the adjoint variable
   *
   * @return The adjoint state field
   */
  FiniteElementState& adjoint() { return adjoint_; };

  /**
   * @overload
   */
  const FiniteElementState& adjoint() const { return adjoint_; };

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
   * @brief Solve the adjoint problem
   * @note It is expected that the forward analysis is complete and the current displacement state is valid
   * @note If the essential boundary state is not specified, homogeneous essential boundary conditions are applied
   *
   * @param[in] adjoint_load_form The linear form that when assembled contains the right hand side of the adjoint system
   * @param[in] state_with_essential_boundary A optional finite element state containing the non-homogenous essential
   * boundary condition data for the adjoint problem
   * @return The computed adjoint finite element state
   */
  virtual const serac::FiniteElementState& solveAdjoint(mfem::ParLinearForm& adjoint_load_form,
                                                        FiniteElementState*  state_with_essential_boundary = nullptr);

  /**
   * @brief Destroy the Nonlinear Solid Solver object
   */
  virtual ~Solid();

  /**
   * @brief Compute the current residual vector at the current internal state value
   *
   * @note This is of length true degrees of freedom, i.e. the length of the underlying mfem::HypreParVector (true_vec)
   */
  mfem::Vector currentResidual();

  /**
   * Get the current gradient (tangent stiffness) MFEM operator at the current internal state value
   *
   * @note This is of size true degrees of freedom x true degrees of freedom, i.e. the length of the underlying
   *mfem::HypreParVector (true_vec)
   * @note This is for expert users only, changing any values inside of the returned data structures can have drastic
   *and unrecoverable runtime consequences.
   **/
  const mfem::Operator& currentGradient();

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
   * @brief Adjoint displacement field
   */
  FiniteElementState adjoint_;

  /**
   * @brief The quasi-static operator for use with the MFEM newton solvers
   */
  std::unique_ptr<mfem::Operator> residual_;

  /**
   * @brief The viscosity coefficient
   */
  std::unique_ptr<mfem::Coefficient> viscosity_;

  /**
   * @brief The mass density coefficient
   */
  std::unique_ptr<mfem::Coefficient> initial_mass_density_;

  /**
   * @brief The hyperelastic material model
   */
  std::unique_ptr<HyperelasticMaterial> material_;

  /**
   * @brief Flag for enabling geometric nonlinearities in the residual calculation
   */
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
   * @brief external force coefficents
   */
  std::vector<std::shared_ptr<mfem::VectorCoefficient>> ext_force_coefs_;

  /**
   * @brief zero vector of the appropriate dimensions
   */
  mfem::Vector zero_;

  /**
   * @brief Nonlinear system solver instance
   */
  mfem_ext::EquationSolver nonlin_solver_;

  /**
   * @brief the system of ordinary differential equations for the physics module
   */
  mfem_ext::SecondOrderODE ode2_;

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
};

}  // namespace serac

/**
 * @brief Prototype the specialization for Inlet parsing
 *
 * @tparam The object to be created by inlet
 */
template <>
struct FromInlet<serac::Solid::InputOptions> {
  /// @brief Returns created object from Inlet container
  serac::Solid::InputOptions operator()(const axom::inlet::Container& base);
};
