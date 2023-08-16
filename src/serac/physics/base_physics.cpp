// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/base_physics.hpp"

#include <fstream>

#include "axom/fmt.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

BasePhysics::BasePhysics(std::string name, mfem::ParMesh* pmesh)
    : name_(name),
      sidre_datacoll_id_(StateManager::collectionID(pmesh)),
      mesh_(StateManager::mesh(sidre_datacoll_id_)),
      comm_(mesh_.GetComm()),
      shape_displacement_(StateManager::shapeDisplacement(sidre_datacoll_id_)),
      shape_displacement_sensitivity_(StateManager::shapeDisplacementSensitivity(sidre_datacoll_id_)),
      time_(StateManager::time(sidre_datacoll_id_)),
      cycle_(StateManager::cycle(sidre_datacoll_id_)),
      ode_time_point_(0.0),
      bcs_(mesh_)
{
  std::tie(mpi_size_, mpi_rank_) = getMPIInfo(comm_);
  order_                         = 1;
}

BasePhysics::BasePhysics(int n, int p, std::string name, mfem::ParMesh* pmesh) : BasePhysics(name, pmesh)
{
  order_ = p;
  // If this is a restart run, things have already been initialized
  gf_initialized_.assign(static_cast<std::size_t>(n), StateManager::isRestart());
}

double BasePhysics::time() const { return time_; }

int BasePhysics::cycle() const { return cycle_; }

void BasePhysics::setTimestep(double dt) { timestep_ = dt; }

std::unique_ptr<FiniteElementState> BasePhysics::generateParameter(const std::string& parameter_name,
                                                                   size_t             parameter_index)
{
  SLIC_ERROR_ROOT_IF(
      parameter_index >= parameters_.size(),
      axom::fmt::format("Parameter index '{}' is not available in physics module '{}'", parameter_index, name_));
  SLIC_ERROR_ROOT_IF(
      parameters_[parameter_index].state,
      axom::fmt::format("Parameter index '{}' is already set in physics module '{}'", parameter_index, name_));

  auto new_state = std::make_unique<FiniteElementState>(*parameters_[parameter_index].trial_space, parameter_name);
  StateManager::storeState(*new_state);
  parameters_[parameter_index].state           = new_state.get();
  parameters_[parameter_index].previous_state  = std::make_unique<serac::FiniteElementState>(*new_state);
  *parameters_[parameter_index].previous_state = 0.0;
  parameters_[parameter_index].sensitivity =
      StateManager::newDual(new_state->space(), new_state->name() + "_sensitivity");
  return new_state;
}

void BasePhysics::registerParameter(size_t parameter_index, FiniteElementState& parameter_state)
{
  SLIC_ERROR_ROOT_IF(&parameter_state.mesh() != &mesh_,
                     axom::fmt::format("Mesh of parameter state is not the same as the physics mesh"));

  SLIC_ERROR_ROOT_IF(
      parameter_index >= parameters_.size(),
      axom::fmt::format("Parameter index '{}' is not available in physics module '{}'", parameter_index, name_));

  SLIC_ERROR_ROOT_IF(
      parameter_state.space().GetTrueVSize() != parameters_[parameter_index].state.space().GetTrueVSize(),
      axom::fmt::format(
          "Physics module parameter '{}' has size '{}' while given state has size '{}'. The finite element "
          "spaces are inconsistent.",
          parameter_index, parameters_[parameter_index].state.space().GetTrueVSize(),
          parameter_state.space().GetTrueVSize()));
      StateManager::newDual(parameter_state.space(), detail::addPrefix(name_, parameter_state.name() + "_sensitivity"));
}

void BasePhysics::setShapeDisplacement(FiniteElementState& shape_displacement)
{
  SLIC_ERROR_ROOT_IF(&shape_displacement_.mesh() != &mesh_,
                     axom::fmt::format("Mesh of shape displacement is not the same as the physics mesh"));

  SLIC_ERROR_ROOT_IF(
      shape_displacement.space().GetTrueVSize() != shape_displacement_.space().GetTrueVSize(),
      axom::fmt::format(
          "Physics module shape displacement has size '{}' while given state has size '{}'. The finite element "
          "spaces are inconsistent.",
          shape_displacement_.space().GetTrueVSize(), shape_displacement.space().GetTrueVSize()));
  shape_displacement_ = shape_displacement;
}

void BasePhysics::outputState(std::optional<std::string> paraview_output_dir) const
{
  // Update the states and duals in the state manager
  for (auto& state : states_) {
    StateManager::updateState(*state);
  }

  for (auto& dual : duals_) {
    StateManager::updateDual(*dual);
  }

  for (auto& parameter : parameters_) {
    StateManager::updateState(parameter.state);
    StateManager::updateDual(parameter.sensitivity);
  }

  StateManager::updateState(shape_displacement_);
  StateManager::updateDual(shape_displacement_sensitivity_);

  // Save the restart/Sidre file
  StateManager::save(time_, cycle_, sidre_datacoll_id_);

  // Optionally output a paraview datacollection for visualization
  if (paraview_output_dir) {
    // Check to see if the paraview data collection exists. If not, create it.
    if (!paraview_dc_) {
      std::string output_name = name_;
      if (output_name == "") {
        output_name = "default";
      }

      paraview_dc_ = std::make_unique<mfem::ParaViewDataCollection>(
          output_name, const_cast<mfem::ParMesh*>(&states_.front()->mesh()));
      int max_order_in_fields = 0;

      // Find the maximum polynomial order in the physics module's states
      for (const FiniteElementState* state : states_) {
        paraview_dc_->RegisterField(state->name(), &state->gridFunction());
        max_order_in_fields = std::max(max_order_in_fields, state->space().GetOrder(0));
      }

      for (auto& parameter : parameters_) {
        paraview_dc_->RegisterField(parameter.state.name(), &parameter.state.gridFunction());
        max_order_in_fields = std::max(max_order_in_fields, parameter.state.space().GetOrder(0));
      }

      paraview_dc_->RegisterField(shape_displacement_.name(), &shape_displacement_.gridFunction());
      max_order_in_fields = std::max(max_order_in_fields, shape_displacement_.space().GetOrder(0));

      // Set the options for the paraview output files
      paraview_dc_->SetLevelsOfDetail(max_order_in_fields);
      paraview_dc_->SetHighOrderOutput(true);
      paraview_dc_->SetDataFormat(mfem::VTKFormat::BINARY);
      paraview_dc_->SetCompression(true);
    } else {
      for (const FiniteElementState* state : states_) {
        state->gridFunction();  // update grid function values
      }
      for (auto& parameter : parameters_) {
        parameter.state.gridFunction();
      }
      shape_displacement_.gridFunction();
    }

    // Set the current time, cycle, and requested paraview directory
    paraview_dc_->SetCycle(cycle_);
    paraview_dc_->SetTime(time_);
    paraview_dc_->SetPrefixPath(*paraview_output_dir);

    // Write the paraview file
    paraview_dc_->Save();
  }
}

void BasePhysics::initializeSummary(axom::sidre::DataStore& datastore, double t_final, double dt) const
{
  // Summary Sidre Structure
  // Sidre root
  // └── serac_summary
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
  summary_group->createViewScalar("mpi_rank_count", count);

  // Write curves info
  axom::sidre::Group* curves_group = summary_group->createGroup("curves");

  // Calculate how many time steps which is the array size
  axom::IndexType array_size = static_cast<axom::IndexType>(ceil(t_final / dt));

  // t: array of each time step value
  axom::sidre::View*         t_array_view = curves_group->createView("t");
  axom::sidre::Array<double> ts(t_array_view, 0, array_size);

  for (const FiniteElementState* state : states_) {
    // Group for this Finite Element State (Field)
    axom::sidre::Group* state_group = curves_group->createGroup(state->name());

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
  for (const FiniteElementState* state : states_) {
    // Calculate current stat value
    // Note: These are collective operations.
    l1norm_value   = norm(*state, 1.0);
    l2norm_value   = norm(*state, 2.0);
    linfnorm_value = norm(*state, mfem::infinity());
    avg_value      = avg(*state);
    max_value      = max(*state);
    min_value      = min(*state);

    // Only save on root node
    if (rank == 0) {
      // Group for this Finite Element State (Field)
      axom::sidre::Group* state_group = curves_group->getGroup(state->name());

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
