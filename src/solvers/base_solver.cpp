// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_solver.hpp"
#include "common/serac_types.hpp"
#include <iostream>
#include <fstream>

BaseSolver::BaseSolver()
  : m_ess_bdr_coef(nullptr), m_nat_bdr_coef(nullptr), m_output_type(OutputType::VisIt),
    m_time(0.0), m_cycle(0), m_visit_dc(nullptr), m_gf_initialized(false)
{
  SetTimestepper(TimestepMethod::ForwardEuler);
}

//BaseSolver::BaseSolver(std::map < std::string, mfem::Array<FiniteElementState> &state)
//  : m_state(state), m_ess_bdr_coef(nullptr), m_nat_bdr_coef(nullptr),
//    m_output_type(OutputType::VisIt),
//    m_timestepper(TimestepMethod::ForwardEuler), m_ode_solver(nullptr), m_time(0.0), m_cycle(0), m_visit_dc(nullptr),
//    m_gf_initialized(false)
//{
//  MFEM_ASSERT(state.size() > 0, "State vector array of size 0 in BaseSolver constructor.");
//
//  MPI_Comm_rank(m_state[].space->GetComm(), &m_rank);
//
//  m_ode_solver = new mfem::ForwardEulerSolver;
//
//}

void BaseSolver::SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::VectorCoefficient *ess_bdr_vec_coef)
{
  m_ess_bdr = ess_bdr;
  m_ess_bdr_vec_coef = ess_bdr_vec_coef;
}

void BaseSolver::SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::VectorCoefficient *nat_bdr_vec_coef)
{
  m_nat_bdr = nat_bdr;
  m_nat_bdr_vec_coef = nat_bdr_vec_coef;
}

void BaseSolver::SetEssentialBCs(mfem::Array<int> &ess_bdr, mfem::Coefficient *ess_bdr_coef)
{
  m_ess_bdr = ess_bdr;
  m_ess_bdr_coef = ess_bdr_coef;
}

void BaseSolver::SetNaturalBCs(mfem::Array<int> &nat_bdr, mfem::Coefficient *nat_bdr_coef)
{
  m_nat_bdr = nat_bdr;
  m_nat_bdr_coef = nat_bdr_coef;
}

void BaseSolver::SetState(const std::map<std::string, FiniteElementState> &state)
{
  MFEM_ASSERT(state.Size() > 0, "State vector array of size 0 in BaseSolver::SetState.");
  m_state = state;

  MPI_Comm_rank(m_state.begin()->second.space->GetComm(), &m_rank);
}

std::map< std::string, FiniteElementState > BaseSolver::GetState() const
{
  return m_state;
}

void BaseSolver::SetTimestepper(const TimestepMethod timestepper)
{
  m_timestepper = timestepper;

  switch (m_timestepper) {
  case TimestepMethod::QuasiStatic :
    m_ode_solver = nullptr;
    break;
  case TimestepMethod::BackwardEuler  :
    m_ode_solver = std::make_shared< mfem::BackwardEulerSolver >();
    break;
  case TimestepMethod::SDIRK33    :
    m_ode_solver = std::make_shared< mfem::SDIRK33Solver >();
    break;
  case TimestepMethod::ForwardEuler :
    m_ode_solver = std::make_shared< mfem::ForwardEulerSolver >();
    break;
  case TimestepMethod::RK2    :
    m_ode_solver = std::make_shared< mfem::RK2Solver >(0.5);
    break;
  case TimestepMethod::RK3SSP   :
    m_ode_solver = std::make_shared< mfem::RK3SSPSolver >();
    break;
  case TimestepMethod::RK4    :
    m_ode_solver = std::make_shared< mfem::RK4Solver >();
    break;
  case TimestepMethod::GeneralizedAlpha :
    m_ode_solver = std::make_shared< mfem::GeneralizedAlphaSolver >(0.5);
    break;
  case TimestepMethod::ImplicitMidpoint :
    m_ode_solver = std::make_shared< mfem::ImplicitMidpointSolver >();
    break;
  case TimestepMethod::SDIRK23    :
    m_ode_solver = std::make_shared< mfem::SDIRK23Solver >();
    break;
  case TimestepMethod::SDIRK34    :
    m_ode_solver = std::make_shared< mfem::SDIRK34Solver >();
    break;
  default:
    mfem::mfem_error("Timestep method not recognized!");
  }
}

void BaseSolver::SetTime(const double time)
{
  m_time = time;
}

double BaseSolver::GetTime() const
{
  return m_time;
}

int BaseSolver::GetCycle() const
{
  return m_cycle;
}

void BaseSolver::InitializeOutput(const OutputType output_type, std::string root_name)
{
  MFEM_ASSERT(names.Size() == m_state.Size(), "State vector and name arrays are not the same size.");

  m_root_name = root_name;

  m_output_type = output_type;

  switch(m_output_type) {
  case OutputType::VisIt: {
    m_visit_dc = new mfem::VisItDataCollection(m_root_name, m_state.begin()->second.mesh);

    for (auto & key_value : m_state) {
      auto name = key_value.first;
      auto fe_state = key_value.second;
      m_visit_dc->RegisterField(name, fe_state.gf.get());
    }
    break;
  }

  case OutputType::GLVis: {
    std::ostringstream mesh_name;
    mesh_name << m_root_name << "-mesh." << std::setfill('0') << std::setw(6) << m_rank - 1;
    std::ofstream omesh(mesh_name.str().c_str());
    omesh.precision(8);
    m_state[0].mesh->Print(omesh);
    break;
  }

  default:
    mfem::mfem_error("OutputType not recognized!");
  }
}

void BaseSolver::OutputState() const
{
  switch(m_output_type) {
  case OutputType::VisIt: {
    m_visit_dc->SetCycle(m_cycle);
    m_visit_dc->SetTime(m_time);
    m_visit_dc->Save();
    break;
  }

  case OutputType::GLVis: {

    for (auto & key_value : m_state) {
      auto name = key_value.first;
      auto fe_state = key_value.second;

      std::ostringstream sol_name;
      sol_name << m_root_name << "-" << name << "." << std::setfill('0') << std::setw(
                 6) << m_cycle << "." << std::setfill('0') << std::setw(6) << m_rank - 1;
      std::ofstream osol(sol_name.str().c_str());
      osol.precision(8);
      fe_state.gf->Save(osol);
    }
    break;
  }

  default:
    mfem::mfem_error("OutputType not recognized!");
  }
}
