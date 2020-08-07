// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_solver.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "common/logger.hpp"
#include "common/serac_types.hpp"
#include "fmt/fmt.hpp"

namespace serac {

BaseSolver::BaseSolver(MPI_Comm comm) : comm_(comm), output_type_(serac::OutputType::VisIt), time_(0.0), cycle_(0)
{
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);
  BaseSolver::setTimestepper(serac::TimestepMethod::ForwardEuler);
  order_ = 1;
}

BaseSolver::BaseSolver(MPI_Comm comm, int n, int p) : BaseSolver(comm)
{
  order_ = p;
  state_.resize(n);

  std::generate(state_.begin(), state_.end(), std::make_shared<serac::FiniteElementState>);

  gf_initialized_.assign(n, false);
}

void BaseSolver::setEssentialBCs(const std::set<int>& ess_bdr, serac::BoundaryCondition::Coef ess_bdr_coef,
                                 const mfem::ParFiniteElementSpace& fes, const int component)
{
  serac::BoundaryCondition bc;

  bc.markers.SetSize(state_.front()->mesh->bdr_attributes.Max());
  bc.markers = 0;

  for (int attr : ess_bdr) {
    SLIC_ASSERT_MSG(attr <= bc.markers.Size(), "Attribute specified larger than what is found in the mesh.");
    bc.markers[attr - 1] = 1;
    if (std::any_of(ess_bdr_.cbegin(), ess_bdr_.cend(),
                    [attr](auto&& existing_bc) { return existing_bc.markers[attr - 1] == 1; })) {
      SLIC_WARNING("Multiple definition of essential boundary! Using first definition given.");
      bc.markers[attr - 1] = 0;
    }
  }

  bc.component = component;

  // This function can and should be marked const in MFEM
  // TODO: Raise an issue against MFEM
  // Leave this explicit non-const action in as a stopgap
  const_cast<mfem::ParFiniteElementSpace&>(fes).GetEssentialTrueDofs(bc.markers, bc.true_dofs, component);

  bc.coef = ess_bdr_coef;
  ess_bdr_.push_back(std::move(bc));
}

void BaseSolver::setTrueDofs(const mfem::Array<int>& true_dofs, serac::BoundaryCondition::Coef ess_bdr_coef,
                             int component)
{
  serac::BoundaryCondition bc;

  bc.markers.SetSize(0);

  bc.true_dofs = true_dofs;

  bc.coef = ess_bdr_coef;

  bc.component = component;
  ess_bdr_.push_back(std::move(bc));
}

void BaseSolver::setNaturalBCs(const std::set<int>& nat_bdr, serac::BoundaryCondition::Coef nat_bdr_coef,
                               const int component)
{
  serac::BoundaryCondition bc;

  bc.markers.SetSize(state_.front()->mesh->bdr_attributes.Max());
  bc.markers = 0;

  for (int attr : nat_bdr) {
    SLIC_ASSERT_MSG(attr <= bc->markers.Size(), "Attribute specified larger than what is found in the mesh.");
    bc.markers[attr - 1] = 1;
  }
  bc.component = component;
  bc.coef      = nat_bdr_coef;
  nat_bdr_.push_back(std::move(bc));
}

void BaseSolver::setState(const std::vector<serac::BoundaryCondition::Coef>& state_coef)
{
  SLIC_ASSERT_MSG(state_coef.size() == state_.size(), "State and coefficient bundles not the same size.");

  for (unsigned int i = 0; i < state_coef.size(); ++i) {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    std::visit([this, i](auto&& coef) { state_[i]->gf->ProjectCoefficient(*coef); }, state_coef[i]);
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
      visit_dc_ = std::make_unique<mfem::VisItDataCollection>(root_name_, state_.front()->mesh.get());
      for (const auto& state : state_) {
        visit_dc_->RegisterField(state->name, state->gf.get());
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
      state_.front()->mesh->Print(omesh);

      for (auto& state : state_) {
        std::string   sol_name = fmt::format("{0}-{1}.{2:0>6}.{3:0>6}", root_name_, state->name, cycle_, mpi_rank_);
        std::ofstream osol(sol_name);
        osol.precision(8);
        state->gf->Save(osol);
      }
      break;
    }

    default:
      SLIC_ERROR_ROOT(mpi_rank_, "OutputType not recognized!");
      serac::exitGracefully(true);
  }
}

SystemSolver::SystemSolver(MPI_Comm comm, const LinearSolverParameters& lin_params,
                           const std::optional<NonlinearSolverParameters>& nonlin_params)
{
  // Preconditioner configuration is too varied, maybe a PrecondParams is needed?
  // Maybe a redesign to better support custom preconditioners as well
  switch (lin_params.lin_solver) {
    case LinearSolver::CG:
      iter_lin_solver_ = std::make_unique<mfem::CGSolver>(comm);
      break;
    case LinearSolver::GMRES:
      iter_lin_solver_ = std::make_unique<mfem::GMRESSolver>(comm);
      break;
    case LinearSolver::MINRES:
      iter_lin_solver_ = std::make_unique<mfem::MINRESSolver>(comm);
      break;
    default:
      SLIC_ERROR("Linear solver type not recognized.");
      exitGracefully(true);
  }
  iter_lin_solver_->SetRelTol(lin_params.rel_tol);
  iter_lin_solver_->SetAbsTol(lin_params.abs_tol);
  iter_lin_solver_->SetMaxIter(lin_params.max_iter);
  iter_lin_solver_->SetPrintLevel(lin_params.print_level);

  if (nonlin_params) {
    auto newton_solver = std::make_unique<mfem::NewtonSolver>(comm);
    newton_solver->SetSolver(*iter_lin_solver_);
    newton_solver->SetRelTol(nonlin_params->rel_tol);
    newton_solver->SetAbsTol(nonlin_params->abs_tol);
    newton_solver->SetMaxIter(nonlin_params->max_iter);
    newton_solver->SetPrintLevel(nonlin_params->print_level);
    newton_solver->iterative_mode = nonlin_params->iterative_mode;
    nonlin_solver_                = std::move(newton_solver);
  }
}

}  // namespace serac
