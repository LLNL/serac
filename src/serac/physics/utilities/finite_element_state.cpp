// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/finite_element_state.hpp"

namespace serac {

FiniteElementState::FiniteElementState(mfem::ParMesh& mesh, FiniteElementState::Options&& options)
    : mesh_(&mesh),
      coll_(options.coll ? std::move(options.coll)
                         : std::make_unique<mfem::H1_FECollection>(options.order, mesh.Dimension())),
      space_(std::make_unique<mfem::ParFiniteElementSpace>(
          &mesh, &retrieve(coll_), options.space_dim ? *options.space_dim : mesh.Dimension(), options.ordering)),
      gf_(std::make_unique<mfem::ParGridFunction>(&retrieve(space_))),
      true_vec_(&retrieve(space_)),
      name_(options.name)
{
  retrieve(gf_) = 0.0;
  true_vec_     = 0.0;
}

std::optional<axom::sidre::MFEMSidreDataCollection> StateManager::datacoll_;

void StateManager::initialize(axom::sidre::DataStore& ds)
{
  SLIC_ERROR_IF(datacoll_, "Serac's datacollection can only be initialized once");

  const std::string coll_name    = "serac_datacoll";
  auto              global_grp   = ds.getRoot()->createGroup(coll_name + "_global");
  auto              bp_index_grp = global_grp->createGroup("blueprint_index/" + coll_name);
  auto              domain_grp   = ds.getRoot()->createGroup(coll_name);

  // Needs to be configured to own the mesh data so all mesh data is saved to datastore/output file
  const bool owns_mesh_data = true;
  datacoll_.emplace("serac_datacoll", bp_index_grp, domain_grp, owns_mesh_data);
  datacoll_->SetComm(MPI_COMM_WORLD);
  if (false) {
    datacoll_->Load();
    datacoll_->SetGroupPointers(ds.getRoot()->getGroup(coll_name + "_global/blueprint_index/" + coll_name),
                                ds.getRoot()->getGroup(coll_name));
    SLIC_ERROR_IF(datacoll_->GetBPGroup()->getNumGroups() == 0,
                  "Loaded datastore is empty, was the datastore created on a "
                  "different number of nodes?");

    datacoll_->UpdateStateFromDS();
  }
}

}  // namespace serac
