// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/base_physics.hpp"

#include <fstream>

#include "fmt/fmt.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

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

void BasePhysics::initializeOutput(const serac::OutputType output_type, const std::string& root_name)
{
  root_name_   = root_name;
  output_type_ = output_type;

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
      dc_->Save();
      break;
    case serac::OutputType::SidreVisIt: {
      StateManager::save(time_, cycle_);
      break;
    }

    case serac::OutputType::GLVis: {
      std::string   mesh_name = fmt::format("{0}-mesh.{1:0>6}.{2:0>6}", root_name_, cycle_, mpi_rank_);
      std::ofstream omesh(mesh_name);
      omesh.precision(FLOAT_PRECISION_);
      state_.front().get().mesh().Print(omesh);

      for (FiniteElementState& state : state_) {
        std::string   sol_name = fmt::format("{0}-{1}.{2:0>6}.{3:0>6}", root_name_, state.name(), cycle_, mpi_rank_);
        std::ofstream osol(sol_name);
        osol.precision(FLOAT_PRECISION_);
        state.gridFunc().Save(osol);
      }
      break;
    }

    default:
      SLIC_ERROR_ROOT("OutputType not recognized!");
  }
}

}  // namespace serac
