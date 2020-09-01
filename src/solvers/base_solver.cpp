// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_solver.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "common/common.hpp"
#include "fmt/fmt.hpp"

namespace serac {

BaseSolver::BaseSolver(std::shared_ptr<mfem::ParMesh> mesh)
    : comm_(mesh->GetComm()), mesh_(mesh), output_type_(serac::OutputType::VisIt), time_(0.0), cycle_(0)
{
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);
  BaseSolver::setTimestepper(serac::TimestepMethod::ForwardEuler);
  order_ = 1;
}

BaseSolver::BaseSolver(std::shared_ptr<mfem::ParMesh> mesh, int n, int p) : BaseSolver(mesh)
{
  order_ = p;
  state_.resize(n);
  gf_initialized_.assign(n, false);
}

void BaseSolver::setEssentialBCs(const std::set<int>& ess_bdr, serac::GeneralCoefficient ess_bdr_coef,
                                 FiniteElementState& state, const int component)
{
  bcs_.addEssential(ess_bdr, ess_bdr_coef, state, component);
}

void BaseSolver::setTrueDofs(const mfem::Array<int>& true_dofs, serac::GeneralCoefficient ess_bdr_coef, int component)
{
  bcs_.setTrueDofs(true_dofs, ess_bdr_coef, component);
}

void BaseSolver::setNaturalBCs(const std::set<int>& nat_bdr, serac::GeneralCoefficient nat_bdr_coef,
                               const int component)
{
  bcs_.addNatural(nat_bdr, nat_bdr_coef, *(state_.front()), component);
}

void BaseSolver::setState(const std::vector<serac::GeneralCoefficient>& state_coef)
{
  SLIC_ASSERT_MSG(state_coef.size() == state_.size(), "State and coefficient bundles not the same size.");

  for (unsigned int i = 0; i < state_coef.size(); ++i) {
    state_[i]->project(state_coef[i]);
  }
}

void BaseSolver::setState(const std::vector<std::shared_ptr<serac::FiniteElementState> >& state)
{
  SLIC_ASSERT_MSG(state.size() > 0, "State vector array of size 0.");
  state_ = state;
}

std::vector<std::shared_ptr<serac::FiniteElementState> > BaseSolver::getState() const { return state_; }

void BaseSolver::setTimestepper(const serac::TimestepMethod timestepper)
{
  timestepper_ = timestepper;

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
    default:
      SLIC_ERROR_ROOT(mpi_rank_, "Timestep method not recognized!");
      serac::exitGracefully(true);
  }
}

void BaseSolver::setTime(const double time) { time_ = time; }

double BaseSolver::time() const { return time_; }

int BaseSolver::cycle() const { return cycle_; }

void BaseSolver::initializeOutput(const serac::OutputType output_type, const std::string& root_name)
{
  root_name_ = root_name;

  output_type_ = output_type;

  switch (output_type_) {
    case serac::OutputType::VisIt: {
      visit_dc_ = std::make_unique<mfem::VisItDataCollection>(root_name_, &state_.front()->mesh());
      for (const auto& state : state_) {
        visit_dc_->RegisterField(state->name(), &state->gridFunc());
      }
      break;
    }

    case serac::OutputType::GLVis: {
      break;
    }

    default:
      SLIC_ERROR_ROOT(mpi_rank_, "OutputType not recognized!");
      serac::exitGracefully(true);
  }
}

void BaseSolver::outputState() const
{
  switch (output_type_) {
    case serac::OutputType::VisIt: {
      visit_dc_->SetCycle(cycle_);
      visit_dc_->SetTime(time_);
      visit_dc_->Save();
      break;
    }

    case serac::OutputType::GLVis: {
      std::string   mesh_name = fmt::format("{0}-mesh.{1:0>6}.{2:0>6}", root_name_, cycle_, mpi_rank_);
      std::ofstream omesh(mesh_name);
      omesh.precision(8);
      state_.front()->mesh().Print(omesh);

      for (auto& state : state_) {
        std::string   sol_name = fmt::format("{0}-{1}.{2:0>6}.{3:0>6}", root_name_, state->name(), cycle_, mpi_rank_);
        std::ofstream osol(sol_name);
        osol.precision(8);
        state->gridFunc().Save(osol);
      }
      break;
    }

    default:
      SLIC_ERROR_ROOT(mpi_rank_, "OutputType not recognized!");
      serac::exitGracefully(true);
  }
}

}  // namespace serac
