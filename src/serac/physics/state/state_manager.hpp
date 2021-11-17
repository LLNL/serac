// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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
   * @param[in] collection_name_prefix The prefix for the name of the Sidre DataCollection to be created
   * FIXME: Remove collection_name_prefix - No longer needed since we associate a mesh/datacoll with a string elsewhere
   * @param[in] output_directory The directory to output files to - cannot be empty
   * @param[in] cycle_to_load The cycle to load - required for restarts
   * FIXME: Would we want to have different meshes at different cycles? Maybe need a method @p restart(name,cycle)
   */
  static void initialize(axom::sidre::DataStore& ds, const std::string output_directory,
                         const std::optional<int> cycle_to_load = {});

  /**
   * @brief Factory method for creating a new FEState object, signature is identical to FEState constructor
   * @param[in] options Configuration options for the FEState, if a new state is created
   * @see FiniteElementState::FiniteElementState
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementState newState(FiniteElementVector::Options&& options = {},
                                     const std::string&             tag     = primary_mesh_name_);

  /**
   * @brief Factory method for creating a new FEDual object, signature is identical to FEDual constructor
   * @param[in] options Configuration options for the FEDual, if a new state is created
   * @see FiniteElementDual::FiniteElementDual
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementDual newDual(FiniteElementVector::Options&& options = {},
                                   const std::string&             tag     = primary_mesh_name_);

  /**
   * @brief Factory method for creating a new QuadratureData object
   * @tparam T The type of the per-qpt data
   * @param[in] name The name of the quadrature data field
   * @param[in] p The order of the quadrature rule
   * @see QuadratureData::QuadratureData
   */
  template <typename T>
  static QuadratureData<T>& newQuadratureData(const std::string& name, const int p,
                                              const std::string& tag = primary_mesh_name_)
  {
    auto& datacoll = datacolls_.at(tag);
    if (is_restart_) {
      auto field = datacoll.GetQField(name);
      syncable_data_.push_back(std::make_unique<QuadratureData<T>>(*field));
      return static_cast<QuadratureData<T>&>(*syncable_data_.back());
    } else {
      SLIC_ERROR_ROOT_IF(datacoll.HasQField(name),
                         axom::fmt::format("Serac's datacollection was already given a qfield named '{0}'", name));
      syncable_data_.push_back(std::make_unique<QuadratureData<T>>(mesh(), p, false));
      // The static_cast is safe here because we "know" what we just inserted into the vector
      auto& qdata = static_cast<QuadratureData<T>&>(*syncable_data_.back());
      datacoll.RegisterQField(name, &(qdata.QFunc()));
      return qdata;
    }
  }

  /**
   * @brief Updates the Conduit Blueprint state in the datastore and saves to a file
   * @param[in] t The current sim time
   * @param[in] cycle The current iteration number of the simulation
   */
  static void save(const double t, const int cycle, const std::string& tag = primary_mesh_name_);

  /**
   * @brief Resets the underlying global datacollection object
   */
  static void reset()
  {
    datacolls_.clear();
    is_restart_ = false;
    syncable_data_.clear();
    ds_ = nullptr;
  };

  /**
   * @brief Gives ownership of mesh to StateManager
   */
  static void setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& tag = primary_mesh_name_);

  /**
   * @brief Returns a non-owning reference to mesh held by StateManager
   */
  static mfem::ParMesh& mesh(const std::string& tag = primary_mesh_name_);

  /**
   * @brief Returns the Sidre DataCollection name
   */
  static const std::string collectionName() { return collection_name_; }

private:
  /**
   * @brief Creates a new datacollection based on a registered mesh
   */
  static void newDataCollection(const std::string& name);

  /**
   * @brief The datacollection instances
   * The object is constructed when the user calls StateManager::initialize.
   */
  static std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> datacolls_;
  /**
   * @brief Whether this simulation has been restarted from another simulation
   */
  static bool is_restart_;
  /**
   * @brief Name of the Sidre DataCollection
   */
  static std::string collection_name_;
  /**
   * @brief A set of @p QuadratureData<T> objects that need to be synchronized before saving to disk
   * FIXME Need one set per datacoll since we can save at different times?
   */
  static std::vector<std::unique_ptr<SyncableData>> syncable_data_;

  /// @brief Pointer (non-owning) to the datastore
  static axom::sidre::DataStore* ds_;
  /// @brief Output directory to which all datacollections are saved
  static std::string output_dir_;
  /// @brief Cycle to load - for restarts - FIXME can this be different between datacollections?
  static std::optional<int> cycle_to_load_;
  /// @brief Default name for the mesh - mostly for backwards compatibility
  const static std::string primary_mesh_name_;
};

}  // namespace serac
