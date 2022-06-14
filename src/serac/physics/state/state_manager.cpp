// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/state_manager.hpp"

#include "axom/core.hpp"

namespace serac {

/**
 * @brief Definition of extern QuadratureData variable
 */
QuadratureData<void> dummy_qdata;

/**
 * @brief Definition of extern QuadratureDataView variable
 */
QuadratureDataView<void> dummy_qdata_view;

// Initialize StateManager's static members - these will be fully initialized in StateManager::initialize
std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> StateManager::datacolls_;
bool                                                                  StateManager::is_restart_ = false;
std::vector<std::unique_ptr<SyncableData>>                            StateManager::syncable_data_;
axom::sidre::DataStore*                                               StateManager::ds_                = nullptr;
std::string                                                           StateManager::output_dir_        = "";
const std::string                                                     StateManager::default_mesh_name_ = "default";

double StateManager::newDataCollection(const std::string& name, const std::optional<int> cycle_to_load)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Cannot construct a DataCollection without a DataStore");
  std::string coll_name = name + "_datacoll";

  auto global_grp   = ds_->getRoot()->createGroup(coll_name + "_global");
  auto bp_index_grp = global_grp->createGroup("blueprint_index/" + coll_name);
  auto domain_grp   = ds_->getRoot()->createGroup(coll_name);

  // Needs to be configured to own the mesh data so all mesh data is saved to datastore/output file
  constexpr bool owns_mesh_data = true;
  auto [iter, _]                = datacolls_.emplace(std::piecewise_construct, std::forward_as_tuple(name),
                                      std::forward_as_tuple(coll_name, bp_index_grp, domain_grp, owns_mesh_data));
  auto& datacoll                = iter->second;
  datacoll.SetComm(MPI_COMM_WORLD);

  datacoll.SetPrefixPath(output_dir_);

  if (cycle_to_load) {
    // NOTE: Load invalidates previous Sidre pointers
    datacoll.Load(*cycle_to_load);
    datacoll.SetGroupPointers(ds_->getRoot()->getGroup(coll_name + "_global/blueprint_index/" + coll_name),
                              ds_->getRoot()->getGroup(coll_name));
    SLIC_ERROR_ROOT_IF(datacoll.GetBPGroup()->getNumGroups() == 0,
                       "Loaded datastore is empty, was the datastore created on a "
                       "different number of nodes?");

    datacoll.UpdateStateFromDS();
    datacoll.UpdateMeshAndFieldsFromDS();

    // Functional needs the nodal grid function and neighbor data in the mesh
    mesh(name).EnsureNodes();
    mesh(name).ExchangeFaceNbrData();

  } else {
    datacoll.SetCycle(0);   // Iteration counter
    datacoll.SetTime(0.0);  // Simulation time
  }

  return datacoll.GetTime();
}

void StateManager::initialize(axom::sidre::DataStore& ds, const std::string& output_directory)
{
  // If the global object has already been initialized, clear it out
  if (ds_) {
    reset();
  }
  ds_         = &ds;
  output_dir_ = output_directory;
  if (output_directory.empty()) {
    SLIC_ERROR_ROOT(
        "DataCollection output directory cannot be empty - this will result in problems if executables are run in "
        "parallel");
  }
}

FiniteElementState StateManager::newState(FiniteElementVector::Options&& options, const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&             datacoll = datacolls_.at(mesh_tag);
  const std::string name     = options.name;
  if (is_restart_) {
    auto field = datacoll.GetParField(name);
    return {mesh(mesh_tag), *field, name};
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    options.managed_by_sidre = true;
    FiniteElementState state(mesh(mesh_tag), std::move(options));
    datacoll.RegisterField(name, &(state.gridFunc()));
    // Now that it's been allocated, we can set it to zero
    state.gridFunc() = 0.0;
    return state;
  }
}

FiniteElementDual StateManager::newDual(FiniteElementVector::Options&& options, const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&             datacoll = datacolls_.at(mesh_tag);
  const std::string name     = options.name;
  if (is_restart_) {
    auto field = datacoll.GetParField(name);
    return {mesh(mesh_tag), *field, name};
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    options.managed_by_sidre = true;
    FiniteElementDual dual(mesh(mesh_tag), std::move(options));

    // Create a grid function view of the local vector for plotting
    // Note: this is a static cast because we know this vector under the hood is a grid function
    // This is hidden from the user because the interpolation capabilities of a grid function
    // are inappropriate for dual vectors.
    auto gf_view_of_local_dual_vector = static_cast<mfem::ParGridFunction*>(&dual.localVec());

    datacoll.RegisterField(name, gf_view_of_local_dual_vector);
    // Now that it's been allocated, we can set it to zero
    dual.localVec() = 0.0;
    return dual;
  }
}

void StateManager::save(const double t, const int cycle, const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&       datacoll  = datacolls_.at(mesh_tag);
  std::string file_path = axom::utilities::filesystem::joinPath(datacoll.GetPrefixPath(), datacoll.GetCollectionName());
  SLIC_INFO_ROOT(axom::fmt::format("Saving data collection at time: {} to path: {}", t, file_path));

  for (const auto& data : syncable_data_) {
    data->sync();
  }
  datacoll.SetTime(t);
  datacoll.SetCycle(cycle);
  datacoll.Save();
}

mfem::ParMesh* StateManager::setMesh(std::unique_ptr<mfem::ParMesh> pmesh, const std::string& mesh_tag)
{
  newDataCollection(mesh_tag);
  auto& datacoll = datacolls_.at(mesh_tag);
  datacoll.SetMesh(pmesh.release());
  datacoll.SetOwnData(true);

  // Functional needs the nodal grid function and neighbor data in the mesh
  auto& new_pmesh = mesh(mesh_tag);
  new_pmesh.EnsureNodes();
  new_pmesh.ExchangeFaceNbrData();
  return &new_pmesh;
}

mfem::ParMesh& StateManager::mesh(const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto mesh = datacolls_.at(mesh_tag).GetMesh();
  SLIC_ERROR_ROOT_IF(!mesh, "The datacollection does not contain a mesh object");
  return static_cast<mfem::ParMesh&>(*mesh);
}

std::string StateManager::collectionID(mfem::ParMesh* pmesh)
{
  if (!pmesh) {
    return default_mesh_name_;
  } else {
    for (auto& [name, datacoll] : datacolls_) {
      if (datacoll.GetMesh() == pmesh) {
        return name;
      }
    }
    SLIC_ERROR_ROOT("The mesh has not been registered with StateManager");
    return {};
  }
}

}  // namespace serac
