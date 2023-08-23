/**
 * @file contact_physics.hpp
 *
 * @brief The contact library interface class
 */

#pragma once

#include <memory>

#include "mfem.hpp"
#include "serac/serac_config.hpp" // TODO change to add serac_
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/numerics/solver_config.hpp"
#include "serac/numerics/solver_config.hpp"

namespace serac {

/**
 * @brief Container for all interactions with the Tribol contact library
 *
 * This class provides a wrapper for all of the information needed by
 * Tribol to generate the mortar weights for the current penalty contact formulation.
 * All of the secondary nodes are currently considered active and therefore we are
 * enforcing an equality constraint.
 **/

/**
 * @brief Two sided contact enforcement information
 */
struct ContactData {
  /**
   * @brief The primary side boundary attribute indicator array
   */
  mfem::Array<int> primary_markers;

  /**
   * @brief The secondary side boundary attribute indicator array
   */
  mfem::Array<int> secondary_markers;

  /**
   * @brief The primary side true degrees of freedom
   */
  mfem::Array<int> primary_true_dofs;

  /**
   * @brief The secondary side true degrees of freedom
   */
  mfem::Array<int> secondary_true_dofs;

  /**
   * @brief The secondary side true degrees of freedom
   */
  int component;

  /**
   * @brief The penalty parameter to enforce the boundary condition
   */
  double penalty;

  /**
   * @brief Formulation of contact problem
   */
  ContactFormulation formulation;
};

class ContactPhysics {
public:
  /**
   * @brief The constructor
   *
   * @param[in] mesh The MFEM parallel mesh
   * @param[in] displacement The displacement state
   * @param[in] reference_nodes The stress-free nodal positions
   * @param[in] contact_data The contact boundary condition information bundles
   */
  ContactPhysics(mfem::ParMesh& mesh, serac::FiniteElementState& displacement, mfem::ParGridFunction& reference_nodes,
                 std::vector<std::shared_ptr<serac::ContactData>>& contact_data);

  /**
   * @brief Updates the positions, forces, and jacobian contributions associated with contact
   *
   * @param[in] cycle The current simulation cycle
   * @param[in] time The current time
   * @param[in] dt The current time
   */
  void update(int cycle, double time, double dt);

  /**
   * @brief Get the abstract constraint residual
   *
   * @return The abstract constraint residual/RHS
   */
  mfem::Vector& constraintResidual() { return constraint_rhs_; };

  /**
   * @brief Get the abstract constraint sparse matrix
   *
   * @return the serial MFEM sparse constraint matrix
   */
  mfem::SparseMatrix& constraintMatrix() { return *constraint_matrix_; };

  /**
   * @brief Get the nodal forces
   *
   * @return The nodal force vector
   */
  mfem::Vector& nodalForces() { return force_.gridFunction(); };

  /**
   * @brief Get the nodal gap distances
   *
   * @return The nodal gap distance
   */
  mfem::Vector& gapDistance() { return gap_.gridFunction(); };

  /**
   * @brief Get the nodal pressures (lagrange multipliers)
   *
   * @return The pressure grid function
   */
  mfem::Vector& pressures() { return pres_.gridFunction(); };

  /**
   * @brief Get the reference nodes
   *
   * @return The reference nodal positions
   */
  mfem::Vector& referenceNodes() { return reference_nodes_; };

  /**
   * @brief Get the number of primary faces
   *
   * @return The number of primary faces
   */
  int numPrimaryFaces() { return num_primary_faces_; };

  /**
   * @brief Get the number of secondary faces
   *
   * @return The number of secondary faces
   */
  int numSecondaryFaces() { return num_secondary_faces_; };

  /**
   * @brief Get the number of primary nodes
   *
   * @return The number of primary nodes
   */
  int numPrimaryNodes() { return static_cast<int>(p_nodes_.size()); };

  /**
   * @brief Get the number of secondary nodes
   *
   * @return The number of secondary nodes
   */
  int numSecondaryNodes() { return static_cast<int>(s_nodes_.size()); };

  /**
   * @brief Get the penalty parameter for abstract constraint enforcement
   *
   * @return The global scalar penalty parameter
   */
  double penalty() { return contact_bc_.front()->penalty; };

  /**
   * @brief Get the formulation of the contact constraint
   *
   * @return The global formulation enum for the conact constraint
   */
  ContactFormulation formulation() { return contact_bc_.front()->formulation; };

  /**
   * @brief The destructor
   */
  ~ContactPhysics(){};

private:
  /**
   * @brief The full problem mesh
   */
  mfem::ParMesh& mesh_;

  /**
   * @brief The contact boundary condition information
   */
  std::vector<std::shared_ptr<serac::ContactData>>& contact_bc_;

  /**
   * @brief Storage for the deformed nodes
   */
  mfem::ParGridFunction& reference_nodes_;

  /**
   * @brief The split position vectors for Tribol
   */
  mfem::Vector x_, y_, z_;

  /**
   * @brief The primary nodal force return vectors for Tribol
   */
  mfem::Vector p_fx_, p_fy_, p_fz_;

  /**
   * @brief The secondary nodal force return vectors for Tribol
   */
  mfem::Vector s_fx_, s_fy_, s_fz_;

  /**
   * @brief The abstract constraint RHS
   */
  mfem::Vector constraint_rhs_;

  /**
   * @brief The gap distances for Tribol
   */
  serac::FiniteElementState gap_;

  /**
   * @brief The Lagrange multipliers (pressures) for Tribol
   */
  serac::FiniteElementState pres_;

  /**
   * @brief The nodal forces for Tribol in grid function form
   */
  serac::FiniteElementState force_;

  /**
   * @brief The reduced (rectangular) sparse constraint matrix
   */
  std::unique_ptr<mfem::SparseMatrix> constraint_matrix_;

  /**
   * @brief The primary side connectivities
   */
  std::vector<int> p_conn_;

  /**
   * @brief The secondary side connectivities
   */
  std::vector<int> s_conn_;

  /**
   * @brief The primary nodes
   */
  std::set<int> p_nodes_;

  /**
   * @brief The secondary nodes
   */
  std::set<int> s_nodes_;

  /**
   * @brief Number primary faces
   */
  int num_primary_faces_;

  /**
   * @brief Number secondary faces
   */
  int num_secondary_faces_;

#ifdef SERAC_USE_TRIBOL

  /**
   * @brief The full unowned nodal (square) constraint matrix (owned by tribol)
   */
  mfem::SparseMatrix* full_constraint_matrix_;

  /**
   * @brief The displacement state
   */
  serac::FiniteElementState& displacement_;

  /**
   * @brief Tribol element type
   */
  int elem_type_;

  /**
   * @brief Tribol boundary type
   */
  int bdr_type_;

  /**
   * @brief Number of vertices
   */
  int num_verts_;

#endif

  /**
   * @brief Update the tribol mesh containers
   */
  void updatePosition();
};

}  // namespace serac
