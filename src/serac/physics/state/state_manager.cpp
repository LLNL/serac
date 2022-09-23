// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/state/state_manager.hpp"

#include "axom/core.hpp"

namespace serac {

// Initialize StateManager's static members - these will be fully initialized in StateManager::initialize
std::unordered_map<std::string, axom::sidre::MFEMSidreDataCollection> StateManager::datacolls_;
bool                                                                  StateManager::is_restart_        = false;
axom::sidre::DataStore*                                               StateManager::ds_                = nullptr;
std::string                                                           StateManager::output_dir_        = "";
const std::string                                                     StateManager::default_mesh_name_ = "default";
std::vector<std::pair<std::unique_ptr<FiniteElementState>, mfem::ParGridFunction*>> StateManager::managed_states_ = {};
std::vector<std::pair<std::unique_ptr<FiniteElementDual>, mfem::ParGridFunction*>>  StateManager::managed_duals_  = {};

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

FiniteElementState& StateManager::newState(FiniteElementVector::Options&& options, const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  const std::string      name     = options.name;
  auto                   state    = std::make_unique<FiniteElementState>(mesh(mesh_tag), std::move(options));
  mfem::ParGridFunction* sidre_grid_function;
  if (is_restart_) {
    sidre_grid_function = datacoll.GetParField(name);
    state->setFromGridFunction(*sidre_grid_function);
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));

    // Create a new grid function with unallocated data. This will be managed by sidre.
    sidre_grid_function = new mfem::ParGridFunction(&state->space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, sidre_grid_function);
    state->setFromGridFunction(*sidre_grid_function);
  }

  managed_states_.emplace_back(std::move(state), sidre_grid_function);

  return *managed_states_.back().first;
}

FiniteElementState& StateManager::newState(const mfem::ParFiniteElementSpace& space, const std::string& name,
                                           const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  auto                   state    = std::make_unique<FiniteElementState>(mesh(mesh_tag), space, name);
  mfem::ParGridFunction* sidre_grid_function;
  if (is_restart_) {
    sidre_grid_function = datacoll.GetParField(name);
    state->setFromGridFunction(*sidre_grid_function);
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));

    // Create a new grid function with unallocated data. This will be managed by sidre.
    sidre_grid_function = new mfem::ParGridFunction(&state->space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, sidre_grid_function);
    state->setFromGridFunction(*sidre_grid_function);
  }

  managed_states_.emplace_back(std::move(state), sidre_grid_function);
  return *managed_states_.back().first;
}

FiniteElementDual& StateManager::newDual(FiniteElementVector::Options&& options, const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  const std::string      name     = options.name;
  auto                   dual     = std::make_unique<FiniteElementDual>(mesh(mesh_tag), std::move(options));
  mfem::ParGridFunction* sidre_grid_function;
  if (is_restart_) {
    sidre_grid_function = datacoll.GetParField(name);
    std::unique_ptr<mfem::HypreParVector> true_dofs(sidre_grid_function->GetTrueDofs());
    *dual = *true_dofs;
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    // Create a new grid function with unallocated data. This will be managed by sidre.
    sidre_grid_function = new mfem::ParGridFunction(&dual->space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, sidre_grid_function);
    std::unique_ptr<mfem::HypreParVector> true_dofs(sidre_grid_function->GetTrueDofs());
    *dual = *true_dofs;
  }
  managed_duals_.emplace_back(std::move(dual), sidre_grid_function);

  return *managed_duals_.back().first;
}

FiniteElementDual& StateManager::newDual(const mfem::ParFiniteElementSpace& space, const std::string& name,
                                         const std::string& mesh_tag)
{
  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&                  datacoll = datacolls_.at(mesh_tag);
  auto                   dual     = std::make_unique<FiniteElementDual>(mesh(mesh_tag), space, name);
  mfem::ParGridFunction* sidre_grid_function;
  if (is_restart_) {
    sidre_grid_function = datacoll.GetParField(name);
    std::unique_ptr<mfem::HypreParVector> true_dofs(sidre_grid_function->GetTrueDofs());
    *dual = *true_dofs;
  } else {
    SLIC_ERROR_ROOT_IF(datacoll.HasField(name),
                       axom::fmt::format("Serac's datacollection was already given a field named '{0}'", name));
    // Create a new grid function with unallocated data. This will be managed by sidre.
    sidre_grid_function = new mfem::ParGridFunction(&dual->space(), static_cast<double*>(nullptr));
    datacoll.RegisterField(name, sidre_grid_function);
    std::unique_ptr<mfem::HypreParVector> true_dofs(sidre_grid_function->GetTrueDofs());
    *dual = *true_dofs;
  }
  managed_duals_.emplace_back(std::move(dual), sidre_grid_function);

  return *managed_duals_.back().first;
}

void StateManager::save(const double t, const int cycle, const std::string& mesh_tag)
{
  // Sync the sidre grid functions representing finite element states
  for (auto& state_pair : managed_states_) {
    state_pair.first->fillGridFunction(*state_pair.second);
  }

  // Sync the sidre grid functions representing finite element duals
  for (auto& dual_pair : managed_duals_) {
    dual_pair.first->space().GetRestrictionMatrix()->MultTranspose(*dual_pair.first, *dual_pair.second);
  }

  SLIC_ERROR_ROOT_IF(!ds_, "Serac's data store was not initialized - call StateManager::initialize first");
  SLIC_ERROR_ROOT_IF(datacolls_.find(mesh_tag) == datacolls_.end(),
                     axom::fmt::format("Mesh tag \"{}\" not found in the data store", mesh_tag));
  auto&       datacoll  = datacolls_.at(mesh_tag);
  std::string file_path = axom::utilities::filesystem::joinPath(datacoll.GetPrefixPath(), datacoll.GetCollectionName());
  SLIC_INFO_ROOT(axom::fmt::format("Saving data collection at time: {} to path: {}", t, file_path));

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
