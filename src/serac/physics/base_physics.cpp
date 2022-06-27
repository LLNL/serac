// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/base_physics.hpp"

#include <fstream>

#include "axom/fmt.hpp"

#include "axom/core.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

BasePhysics::BasePhysics(mfem::ParMesh* pmesh)
    : sidre_datacoll_id_(StateManager::collectionID(pmesh)),
      mesh_(StateManager::mesh(sidre_datacoll_id_)),
      comm_(mesh_.GetComm()),
      time_(0.0),
      cycle_(0),
      bcs_(mesh_)
{
  std::tie(mpi_size_, mpi_rank_) = getMPIInfo(comm_);
  order_                         = 1;
  root_name_                     = "serac";
}

BasePhysics::BasePhysics(int n, int p, mfem::ParMesh* pmesh) : BasePhysics(pmesh)
{
  order_ = p;
  // If this is a restart run, things have already been initialized
  gf_initialized_.assign(static_cast<std::size_t>(n), StateManager::isRestart());
}

const std::vector<std::reference_wrapper<serac::FiniteElementState> >& BasePhysics::getState() const { return state_; }

void BasePhysics::setTime(const double time)
{
  time_ = time;
  bcs_.setTime(time_);
}

double BasePhysics::time() const { return time_; }

void BasePhysics::setCycle(const int cycle) { cycle_ = cycle; }

int BasePhysics::cycle() const { return cycle_; }

void BasePhysics::outputState() const { StateManager::save(time_, cycle_, sidre_datacoll_id_); }

void BasePhysics::initializeSummary(axom::sidre::DataStore& datastore, double t_final, double dt) const
{
  // Summary Sidre Structure
  // Sidre root
  // └── serac_summary
  //     ├── user_name : const char*
  //     ├── host_name : const char*
  //     ├── mpi_rank_count : int
  //     └── curves
  //         ├── t : Sidre::Array<axom::IndexType>
  //         ├── <FiniteElementState name>
  //         │    ├── l1norm : Sidre::Array<double>
  //         │    └── l2norm : Sidre::Array<double>
  //         ...
  //         └── <FiniteElementState name>
  //              ├── l1norm : Sidre::Array<double>
  //              └── l2norm : Sidre::Array<double>

  auto [count, rank] = getMPIInfo();
  if (rank != 0) {
    // Don't initialize except on root node
    return;
  }
  const std::string   summary_group_name = "serac_summary";
  axom::sidre::Group* sidre_root         = datastore.getRoot();
  SLIC_ERROR_ROOT_IF(
      sidre_root->hasGroup(summary_group_name),
      axom::fmt::format("Sidre Group '{0}' cannot exist when initializeSummary is called", summary_group_name));
  axom::sidre::Group* summary_group = sidre_root->createGroup(summary_group_name);

  // Write run info
  summary_group->createViewString("user_name", axom::utilities::getUserName());
  summary_group->createViewString("host_name", axom::utilities::getHostName());
  summary_group->createViewScalar("mpi_rank_count", count);

  // Write curves info
  axom::sidre::Group* curves_group = summary_group->createGroup("curves");

  // Calculate how many time steps which is the array size
  axom::IndexType array_size = static_cast<axom::IndexType>(ceil(t_final / dt));

  // t: array of each time step value
  axom::sidre::View*         t_array_view = curves_group->createView("t");
  axom::sidre::Array<double> ts(t_array_view, 0, array_size);

  for (FiniteElementState& state : state_) {
    // Group for this Finite Element State (Field)
    axom::sidre::Group* state_group = curves_group->createGroup(state.name());

    // Create an array for each stat type to hold a value at each time step
    for (std::string stat_name : {"l1norms", "l2norms", "linfnorms", "avgs", "mins", "maxs"}) {
      axom::sidre::View*         curr_array_view = state_group->createView(stat_name);
      axom::sidre::Array<double> array(curr_array_view, 0, array_size);
    }
  }
}

void BasePhysics::saveSummary(axom::sidre::DataStore& datastore, const double t) const
{
  auto [_, rank] = getMPIInfo();

  // Find curves sidre group
  axom::sidre::Group* curves_group = nullptr;
  // Only save on root node
  if (rank == 0) {
    axom::sidre::Group* sidre_root        = datastore.getRoot();
    const std::string   curves_group_name = "serac_summary/curves";
    SLIC_ERROR_IF(!sidre_root->hasGroup(curves_group_name),
                  axom::fmt::format("Sidre Group '{0}' did not exist when saveCurves was called", curves_group_name));
    curves_group = sidre_root->getGroup(curves_group_name);

    // Save time step
    axom::sidre::Array<double> ts(curves_group->getView("t"));
    ts.push_back(t);
  }

  // For each Finite Element State (Field)
  double l1norm_value, l2norm_value, linfnorm_value, avg_value, max_value, min_value;
  for (FiniteElementState& state : state_) {
    // Calculate current stat value
    // Note: These are collective operations.
    l1norm_value   = norm(state, 1.0);
    l2norm_value   = norm(state, 2.0);
    linfnorm_value = norm(state, mfem::infinity());
    avg_value      = avg(state);
    max_value      = max(state);
    min_value      = min(state);

    // Only save on root node
    if (rank == 0) {
      // Group for this Finite Element State (Field)
      axom::sidre::Group* state_group = curves_group->getGroup(state.name());

      // Save all current stat values in their respective sidre arrays
      axom::sidre::View*         l1norms_view = state_group->getView("l1norms");
      axom::sidre::Array<double> l1norms(l1norms_view);
      l1norms.push_back(l1norm_value);

      axom::sidre::View*         l2norms_view = state_group->getView("l2norms");
      axom::sidre::Array<double> l2norms(l2norms_view);
      l2norms.push_back(l2norm_value);

      axom::sidre::View*         linfnorms_view = state_group->getView("linfnorms");
      axom::sidre::Array<double> linfnorms(linfnorms_view);
      linfnorms.push_back(linfnorm_value);

      axom::sidre::View*         avgs_view = state_group->getView("avgs");
      axom::sidre::Array<double> avgs(avgs_view);
      avgs.push_back(avg_value);

      axom::sidre::View*         maxs_view = state_group->getView("maxs");
      axom::sidre::Array<double> maxs(maxs_view);
      maxs.push_back(max_value);

      axom::sidre::View*         mins_view = state_group->getView("mins");
      axom::sidre::Array<double> mins(mins_view);
      mins.push_back(min_value);
    }
  }
}

namespace detail {
std::string addPrefix(const std::string& prefix, const std::string& target)
{
  if (prefix.empty()) {
    return target;
  }
  return prefix + "_" + target;
}
}  // namespace detail

}  // namespace serac
