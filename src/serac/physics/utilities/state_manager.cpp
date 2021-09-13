// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/state_manager.hpp"

#include "axom/core.hpp"

namespace serac {

/**
 * @brief Definition of extern variable
 */
QuadratureData<void>     dummy_qdata;
QuadratureDataView<void> dummy_qdata_view;

// Initialize StateManager's static members - both of these will be fully initialized in StateManager::initialize
std::optional<axom::sidre::MFEMSidreDataCollection> StateManager::datacoll_;
bool                                                StateManager::is_restart_      = false;
std::string                                         StateManager::collection_name_ = "";
std::vector<std::unique_ptr<SyncableData>>          StateManager::syncable_data_;

void StateManager::initialize(axom::sidre::DataStore& ds, const std::string& collection_name_prefix,
                              const std::string output_directory, const std::optional<int> cycle_to_load)
{
  // If the global object has already been initialized, clear it out
  if (datacoll_) {
    reset();
  }

  collection_name_ = collection_name_prefix + "_datacoll";

  auto global_grp   = ds.getRoot()->createGroup(collection_name_ + "_global");
  auto bp_index_grp = global_grp->createGroup("blueprint_index/" + collection_name_);
  auto domain_grp   = ds.getRoot()->createGroup(collection_name_);

  // Needs to be configured to own the mesh data so all mesh data is saved to datastore/output file
  const bool owns_mesh_data = true;
  datacoll_.emplace(collection_name_, bp_index_grp, domain_grp, owns_mesh_data);
  datacoll_->SetComm(MPI_COMM_WORLD);

  if (!output_directory.empty()) {
    datacoll_->SetPrefixPath(output_directory);
  }

  if (cycle_to_load) {
    is_restart_ = true;
    // NOTE: Load invalidates previous Sidre pointers
    datacoll_->Load(*cycle_to_load);
    datacoll_->SetGroupPointers(
        ds.getRoot()->getGroup(collection_name_ + "_global/blueprint_index/" + collection_name_),
        ds.getRoot()->getGroup(collection_name_));
    SLIC_ERROR_ROOT_IF(datacoll_->GetBPGroup()->getNumGroups() == 0,
                       "Loaded datastore is empty, was the datastore created on a "
                       "different number of nodes?");

    datacoll_->UpdateStateFromDS();
    datacoll_->UpdateMeshAndFieldsFromDS();
  } else {
    datacoll_->SetCycle(0);   // Iteration counter
    datacoll_->SetTime(0.0);  // Simulation time
  }
}

FiniteElementState StateManager::newState(FiniteElementVector::Options&& options)
{
  SLIC_ERROR_ROOT_IF(!datacoll_, "Serac's datacollection was not initialized - call StateManager::initialize first");
  const std::string name = options.name;
  if (is_restart_) {
    auto field = datacoll_->GetParField(name);
    return {mesh(), *field, name};
  } else {
    SLIC_ERROR_ROOT_IF(datacoll_->HasField(name),
                       fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    options.managed_by_sidre = true;
    FiniteElementState state(mesh(), std::move(options));
    datacoll_->RegisterField(name, &(state.gridFunc()));
    // Now that it's been allocated, we can set it to zero
    state.gridFunc() = 0.0;
    return state;
  }
}

FiniteElementDual StateManager::newDual(FiniteElementVector::Options&& options)
{
  SLIC_ERROR_ROOT_IF(!datacoll_, "Serac's datacollection was not initialized - call StateManager::initialize first");
  const std::string name = options.name;
  if (is_restart_) {
    auto field = datacoll_->GetParField(name);
    return {mesh(), *field, name};
  } else {
    SLIC_ERROR_ROOT_IF(datacoll_->HasField(name),
                       fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    options.managed_by_sidre = true;
    FiniteElementDual dual(mesh(), std::move(options));

    // Create a grid function view of the local vector for plotting
    // Note: this is a static cast because we know this vector under the hood is a grid function
    // This is hidden from the user because the interpolation capabilities of a grid function
    // are inappropriate for dual vectors.
    auto gf_view_of_local_dual_vector = static_cast<mfem::ParGridFunction*>(&dual.localVec());

    datacoll_->RegisterField(name, gf_view_of_local_dual_vector);
    // Now that it's been allocated, we can set it to zero
    dual.localVec() = 0.0;
    return dual;
  }
}

void StateManager::save(const double t, const int cycle)
{
  SLIC_ERROR_ROOT_IF(!datacoll_, "Serac's datacollection was not initialized - call StateManager::initialize first");
  for (const auto& data : syncable_data_) {
    data->sync();
  }
  datacoll_->SetTime(t);
  datacoll_->SetCycle(cycle);
  datacoll_->Save();
}

void StateManager::setMesh(std::unique_ptr<mfem::ParMesh> mesh)
{
  datacoll_->SetMesh(mesh.release());
  datacoll_->SetOwnData(true);
}

mfem::ParMesh& StateManager::mesh()
{
  auto mesh = datacoll_->GetMesh();
  SLIC_ERROR_ROOT_IF(!mesh, "The datastore does not contain a mesh object");
  return static_cast<mfem::ParMesh&>(*mesh);
}

}  // namespace serac
