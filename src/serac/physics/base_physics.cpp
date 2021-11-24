// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
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

BasePhysics::BasePhysics()
    : mesh_(StateManager::mesh()),
      comm_(mesh_.GetComm()),
      output_type_(serac::OutputType::VisIt),
      time_(0.0),
      cycle_(0),
      bcs_(mesh_)
{
  std::tie(mpi_size_, mpi_rank_) = getMPIInfo(comm_);
  order_                         = 1;
  root_name_                     = "serac";
}

BasePhysics::BasePhysics(int n, int p) : BasePhysics()
{
  order_ = p;
  gf_initialized_.assign(static_cast<std::size_t>(n), false);
}

void BasePhysics::setTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef, int component)
{
  bcs_.addEssentialTrueDofs(true_dofs, ess_bdr_coef, component);
}

const std::vector<std::reference_wrapper<serac::FiniteElementState> >& BasePhysics::getState() const { return state_; }

void BasePhysics::setTime(const double time)
{
  time_ = time;
  bcs_.setTime(time_);
}

double BasePhysics::time() const { return time_; }

int BasePhysics::cycle() const { return cycle_; }

void BasePhysics::initializeOutput(const serac::OutputType output_type, const std::string& root_name,
                                   const std::string output_directory)
{
  root_name_        = root_name;
  output_type_      = output_type;
  output_directory_ = output_directory;

  switch (output_type_) {
    case serac::OutputType::VisIt: {
      dc_ = std::make_unique<mfem::VisItDataCollection>(root_name_, &state_.front().get().mesh());
      break;
    }

    case serac::OutputType::ParaView: {
      auto pv_dc = std::make_unique<mfem::ParaViewDataCollection>(root_name_, &state_.front().get().mesh());
      int  max_order_in_fields = 0;
      for (FiniteElementState& state : state_) {
        pv_dc->RegisterField(state.name(), &state.gridFunc());
        max_order_in_fields = std::max(max_order_in_fields, state.space().GetOrder(0));
      }
      pv_dc->SetLevelsOfDetail(max_order_in_fields);
      pv_dc->SetHighOrderOutput(true);
      pv_dc->SetDataFormat(mfem::VTKFormat::BINARY);
      pv_dc->SetCompression(true);
      dc_ = std::move(pv_dc);
      break;
    }

    case serac::OutputType::GLVis:
      [[fallthrough]];
    case OutputType::SidreVisIt: {
      break;
    }

    default:
      SLIC_ERROR_ROOT("OutputType not recognized!");
  }

  if (output_type_ == OutputType::VisIt) {
    // Implicitly convert from ref_wrapper
    for (FiniteElementState& state : state_) {
      dc_->RegisterField(state.name(), &state.gridFunc());
    }
  }
}

void BasePhysics::outputState() const
{
  switch (output_type_) {
    case serac::OutputType::VisIt:
      [[fallthrough]];
    case serac::OutputType::ParaView:
      dc_->SetCycle(cycle_);
      dc_->SetTime(time_);
      if (!output_directory_.empty()) {
        dc_->SetPrefixPath(output_directory_);
      }
      dc_->Save();
      break;
    case serac::OutputType::SidreVisIt: {
      // Implemented through a helper method as the full interface of the MFEMSidreDataCollection
      // is restricted from global access
      StateManager::save(time_, cycle_);
      break;
    }

    case serac::OutputType::GLVis: {
      const std::string mesh_name = axom::fmt::format("{0}-mesh.{1:0>6}.{2:0>6}", root_name_, cycle_, mpi_rank_);
      const std::string mesh_path = axom::utilities::filesystem::joinPath(output_directory_, mesh_name);
      std::ofstream     omesh(mesh_path);
      omesh.precision(FLOAT_PRECISION_);
      state_.front().get().mesh().Print(omesh);

      for (FiniteElementState& state : state_) {
        std::string sol_name =
            axom::fmt::format("{0}-{1}.{2:0>6}.{3:0>6}", root_name_, state.name(), cycle_, mpi_rank_);
        const std::string sol_path = axom::utilities::filesystem::joinPath(output_directory_, sol_name);
        std::ofstream     osol(sol_path);
        osol.precision(FLOAT_PRECISION_);
        state.gridFunc().Save(osol);
      }
      break;
    }

    default:
      SLIC_ERROR_ROOT("OutputType not recognized!");
  }
}

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
  summary_group->createViewString("user_name", serac::getUserName());
  summary_group->createViewString("host_name", serac::getHostName());
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
  double l1norm_value, l2norm_value, linfnorm_value, avg_value, max_value, min_value;

  auto [_, rank]                      = getMPIInfo();
  const std::string curves_group_name = "serac_summary/curves";

  axom::sidre::Group* sidre_root = datastore.getRoot();

  SLIC_ERROR_ROOT_IF(
      !sidre_root->hasGroup(curves_group_name),
      axom::fmt::format("Sidre Group '{0}' did not exist when saveCurves was called", curves_group_name));

  axom::sidre::Group* curves_group = sidre_root->getGroup(curves_group_name);

  // Don't save curves on anything other than root node
  if (rank == 0) {
    axom::sidre::Array<double> ts(curves_group->getView("t"));
    ts.push_back(t);
  }

  // For each Finite Element State (Field)
  for (FiniteElementState& state : state_) {
    // Calculate current stat value
    l1norm_value   = norm(state, 1);
    l2norm_value   = norm(state, 2);
    linfnorm_value = norm(state, mfem::infinity());
    avg_value      = avg(state);
    max_value      = max(state);
    min_value      = min(state);

    // Don't save curves on anything other than root node
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
