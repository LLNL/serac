// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file state_manager.hpp
 *
 * @brief This file contains the declaration of the StateManager class
 */

#pragma once

#include <optional>
#include <unordered_map>

#include "mfem.hpp"
#include "axom/sidre/core/MFEMSidreDataCollection.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"
#include "serac/numerics/quadrature_data.hpp"

namespace serac {

/**
 * @brief Manages the lifetimes of FEState objects such that restarts are abstracted
 * from physics modules
 */
class StateManager {
public:
  /**
   * @brief Initializes the StateManager with a sidre DataStore (into which state will be written/read)
   * @param[in] ds The DataStore to use
   * @param[in] output_directory The directory to output files to - cannot be empty
   */
  static void initialize(axom::sidre::DataStore& ds, const std::string& output_directory);

  /**
   * @brief Factory method for creating a new FEState object, signature is identical to FEState constructor
   * @param[in] options Configuration options for the FEState, if a new state is created
   * @param[in] mesh_tag A string that uniquely identifies the mesh on which the field is to be defined
   * @see FiniteElementState::FiniteElementState
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementState newState(FiniteElementVector::Options&& options  = {},
                                     const std::string&             mesh_tag = default_mesh_name_);

  /**
   * @brief Factory method for creating a new FEDual object, signature is identical to FEDual constructor
   * @param[in] options Configuration options for the FEDual, if a new state is created
   * @param[in] mesh_tag A string that uniquely identifies the mesh on which the dual is to be defined
   * @see FiniteElementDual::FiniteElementDual
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementDual newDual(FiniteElementVector::Options&& options  = {},
                                   const std::string&             mesh_tag = default_mesh_name_);

  /**
   * @brief Updates the Conduit Blueprint state in the datastore and saves to a file
   * @param[in] t The current sim time
   * @param[in] cycle The current iteration number of the simulation
   * @param[in] mesh_tag A string that uniquely identifies the mesh (and accompanying fields) to save
   */
  static void save(const double t, const int cycle, const std::string& mesh_tag = default_mesh_name_);

  /**
   * @brief Loads an existing DataCollection
   * @param[in] cycle_to_load What cycle to load the DataCollection from
   * @param[in] mesh_tag The mesh_tag associated with the DataCollection when it was saved
   * @return The time from specified restart cycle. Otherwise zero.
   */
  static double load(const int cycle_to_load, const std::string& mesh_tag = default_mesh_name_)
  {
    // FIXME: Assumes that if one DataCollection is going to be reloaded all DataCollections will be
    is_restart_ = true;
    return newDataCollection(mesh_tag, cycle_to_load);
  }

  /**
   * @brief Resets the underlying global datacollection object
   */
  static void reset()
  {
    datacolls_.clear();
    is_restart_ = false;
    ds_         = nullptr;
  };

  /**
   * @brief Gives ownership of mesh to StateManager
   * @param[in] pmesh The mesh to register
   * @param[in] mesh_tag A string that uniquely identifies the mesh
   * @return A pointer to the stored mesh whose ownership was just passed to StateManager
   */
  static mfem::ParMesh* setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& mesh_tag = default_mesh_name_);

  /**
   * @brief Returns a non-owning reference to mesh held by StateManager
   * @param[in] mesh_tag A string that uniquely identifies the mesh
   * @pre A mesh identified by @a mesh_tag must be registered - either via @p load() or @p setMesh()
   */
  static mfem::ParMesh& mesh(const std::string& mesh_tag = default_mesh_name_);

  /**
   * @brief Returns the datacollection ID for a given mesh
   * @param[in] pmesh Pointer to a mesh (non-owning)
   * @return The collection ID corresponding to the DataCollection that owns the mesh
   * pointed to by @a pmesh.  If @a pmesh is @p nullptr then the default collection ID is returned.
   * @note A raw pointer comparison is used to identify the datacollection, i.e.,
   * @a pmesh must either been returned by either the setMesh() or mesh() method
   */
  static std::string collectionID(mfem::ParMesh* pmesh);

  /// @brief Returns true if data was loaded into a DataCollection
  static bool isRestart() { return is_restart_; }

private:
  /**
   * @brief Creates a new datacollection based on a registered mesh
   * @param[in] name The name of the new datacollection
   * @param[in] cycle_to_load What cycle to load the DataCollection from, if applicable
   * @return The time from specified restart cycle. Otherwise zero.
   */
  static double newDataCollection(const std::string& name, const std::optional<int> cycle_to_load = {});

  /**
   * @brief The datacollection instances
   * The object is constructed when the user calls StateManager::initialize.
   */
  static std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> datacolls_;
  /**
   * @brief Whether this simulation has been restarted from another simulation
   */
  static bool is_restart_;

  /// @brief Pointer (non-owning) to the datastore
  static axom::sidre::DataStore* ds_;
  /// @brief Output directory to which all datacollections are saved
  static std::string output_dir_;
  /// @brief Default name for the mesh - mostly for backwards compatibility
  const static std::string default_mesh_name_;
};

}  // namespace serac
