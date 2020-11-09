// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/base_physics.hpp"

#include <fstream>

#include "fmt/fmt.hpp"
#include "infrastructure/initialize.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"

namespace serac {

BasePhysics::BasePhysics(std::shared_ptr<mfem::ParMesh> mesh)
    : comm_(mesh->GetComm()), mesh_(mesh), output_type_(serac::OutputType::VisIt), time_(0.0), cycle_(0), bcs_(*mesh)
{
  std::tie(mpi_size_, mpi_rank_) = getMPIInfo(comm_);
  BasePhysics::setTimestepper(serac::TimestepMethod::ForwardEuler);
  order_ = 1;
}

BasePhysics::BasePhysics(std::shared_ptr<mfem::ParMesh> mesh, int n, int p) : BasePhysics(mesh)
{
  order_ = p;
  gf_initialized_.assign(n, false);
}

void BasePhysics::setTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef, int component)
{
  bcs_.addEssentialTrueDofs(true_dofs, ess_bdr_coef, component);
}

void BasePhysics::setState(const std::vector<serac::GeneralCoefficient>& state_coef)
{
  SLIC_ASSERT_MSG(state_coef.size() == state_.size(), "State and coefficient bundles not the same size.");

  for (unsigned int i = 0; i < state_coef.size(); ++i) {
    state_[i].get().project(state_coef[i]);
  }
}

void BasePhysics::setState(std::vector<serac::FiniteElementState>&& state)
{
  SLIC_ASSERT_MSG(state.size() > 0, "State vector array of size 0.");
  // To avoid making a shallow copy of the references, we move individually
  for (std::size_t i = 0; i < state.size(); i++) {
    // Assigning to the FiniteElementState being referenced,
    // not the reference itself
    state_[i].get() = std::move(state[i]);
  }
}

const std::vector<std::reference_wrapper<serac::FiniteElementState> >& BasePhysics::getState() const { return state_; }

void BasePhysics::setTimestepper(const serac::TimestepMethod             timestepper,
                                 const serac::DirichletEnforcementMethod enforcement_method)
{
  timestepper_ = timestepper;
  enforcement_method_ = enforcement_method;

  switch (timestepper_) {
    case serac::TimestepMethod::QuasiStatic:
      break;
    case serac::TimestepMethod::BackwardEuler:
      ode_solver_ = std::make_unique<mfem::BackwardEulerSolver>();
      break;
    case serac::TimestepMethod::SDIRK33:
      ode_solver_ = std::make_unique<mfem::SDIRK33Solver>();
      break;
    case serac::TimestepMethod::ForwardEuler:
      ode_solver_ = std::make_unique<mfem::ForwardEulerSolver>();
      break;
    case serac::TimestepMethod::RK2:
      ode_solver_ = std::make_unique<mfem::RK2Solver>(0.5);
      break;
    case serac::TimestepMethod::RK3SSP:
      ode_solver_ = std::make_unique<mfem::RK3SSPSolver>();
      break;
    case serac::TimestepMethod::RK4:
      ode_solver_ = std::make_unique<mfem::RK4Solver>();
      break;
    case serac::TimestepMethod::GeneralizedAlpha:
      ode_solver_ = std::make_unique<mfem::GeneralizedAlphaSolver>(0.5);
      break;
    case serac::TimestepMethod::ImplicitMidpoint:
      ode_solver_ = std::make_unique<mfem::ImplicitMidpointSolver>();
      break;
    case serac::TimestepMethod::SDIRK23:
      ode_solver_ = std::make_unique<mfem::SDIRK23Solver>();
      break;
    case serac::TimestepMethod::SDIRK34:
      ode_solver_ = std::make_unique<mfem::SDIRK34Solver>();
      break;


    case serac::TimestepMethod::HHTAlpha:
      second_order_ode_solver_ = std::make_unique<mfem::HHTAlphaSolver>();
      break;
    case serac::TimestepMethod::WBZAlpha:
      second_order_ode_solver_ = std::make_unique<mfem::WBZAlphaSolver>();
      break;
    case serac::TimestepMethod::AverageAcceleration:
      second_order_ode_solver_ = std::make_unique<mfem::AverageAccelerationSolver>();
      break;
    case serac::TimestepMethod::LinearAcceleration:
      second_order_ode_solver_ = std::make_unique<mfem::LinearAccelerationSolver>();
      break;
    case serac::TimestepMethod::CentralDifference:
      second_order_ode_solver_ = std::make_unique<mfem::CentralDifferenceSolver>();
      break;
    case serac::TimestepMethod::FoxGoodwin:
      second_order_ode_solver_ = std::make_unique<mfem::FoxGoodwinSolver>();
      break;


    default:
      SLIC_ERROR_ROOT(mpi_rank_, "Timestep method not recognized!");
      serac::exitGracefully(true);
  }
}

void BasePhysics::setTime(const double time) { time_ = time; }

double BasePhysics::time() const { return time_; }

int BasePhysics::cycle() const { return cycle_; }

void BasePhysics::initializeOutput(const serac::OutputType output_type, const std::string& root_name)
{
  root_name_ = root_name;

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

    case serac::OutputType::GLVis: {
      break;
    }

    case OutputType::SidreVisIt: {
      dc_ = std::make_unique<axom::sidre::MFEMSidreDataCollection>(root_name_, &state_.front().get().mesh());
      break;
    }

    default:
      SLIC_ERROR_ROOT(mpi_rank_, "OutputType not recognized!");
      serac::exitGracefully(true);
  }

  if ((output_type_ == OutputType::VisIt) || (output_type_ == OutputType::SidreVisIt)) {
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
      [[fallthrough]];
    case serac::OutputType::SidreVisIt: {
      dc_->SetCycle(cycle_);
      dc_->SetTime(time_);
      dc_->Save();
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
      SLIC_ERROR_ROOT(mpi_rank_, "OutputType not recognized!");
      serac::exitGracefully(true);
  }
}

}  // namespace serac
