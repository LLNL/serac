// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_solver.hpp"
#include "common/serac_types.hpp"
#include <iostream>
#include <fstream>

BaseSolver::BaseSolver()
  : m_pmesh(nullptr), m_ess_bdr_coef(nullptr), m_nat_bdr_coef(nullptr), m_output_type(OutputType::VisIt),
    m_timestepper(TimestepMethod::ForwardEuler), m_ode_solver(nullptr), m_time(0.0), m_cycle(0), m_visit_dc(nullptr)
{
  m_ode_solver = new mfem::ForwardEulerSolver;
}

BaseSolver::BaseSolver(mfem::Array<mfem::ParGridFunction*> &stategf)
  : m_state_gf(stategf), m_pmesh(nullptr), m_ess_bdr_coef(nullptr), m_nat_bdr_coef(nullptr),
    m_output_type(OutputType::VisIt),
    m_timestepper(TimestepMethod::ForwardEuler), m_ode_solver(nullptr), m_time(0.0), m_cycle(0), m_visit_dc(nullptr)
{
  MFEM_ASSERT(stategf.Size() > 0, "State vector array of size 0 in BaseSolver constructor.");

  m_fespaces.SetSize(m_state_gf.Size());
  m_fecolls.SetSize(m_state_gf.Size());
  m_true_vec.SetSize(m_state_gf.Size());

  for (int i=0; i<m_state_gf.Size(); ++i) {
    m_fespaces[i] = m_state_gf[i]->ParFESpace();
    m_fecolls[i] = m_fespaces[i]->FEColl();
    m_true_vec[i] = new mfem::Vector;
    m_state_gf[i]->GetTrueDofs(*m_true_vec[i]);
  }

  m_pmesh = m_fespaces[0]->GetParMesh();
  MPI_Comm_rank(m_fespaces[0]->GetComm(), &m_rank);

  m_ode_solver = new mfem::ForwardEulerSolver;

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

void BaseSolver::SetState(const mfem::Array<mfem::ParGridFunction*> &state_gf)
{
  MFEM_ASSERT(state_gf.Size() > 0, "State vector array of size 0 in BaseSolver::SetState.");

  m_fespaces.SetSize(m_state_gf.Size());

  for (int i=0; i<m_state_gf.Size(); ++i) {
    m_fespaces[i] = m_state_gf[i]->ParFESpace();
  }

  m_pmesh = m_fespaces[0]->GetParMesh();

  m_time = 0.0;
  m_cycle = 0;
  MPI_Comm_rank(m_fespaces[0]->GetComm(), &m_rank);

  m_state_gf = state_gf;
}

mfem::Array<mfem::ParGridFunction*> BaseSolver::GetState() const
{
  return m_state_gf;
}

void BaseSolver::SetTimestepper(const TimestepMethod timestepper)
{
  m_timestepper = timestepper;
  delete m_ode_solver;

  switch (m_timestepper) {
  case TimestepMethod::BackwardEuler  :
    m_ode_solver = new mfem::BackwardEulerSolver;
    break;
  case TimestepMethod::SDIRK33    :
    m_ode_solver = new mfem::SDIRK33Solver;
    break;
  case TimestepMethod::ForwardEuler :
    m_ode_solver = new mfem::ForwardEulerSolver;
    break;
  case TimestepMethod::RK2    :
    m_ode_solver = new mfem::RK2Solver(0.5);
    break;
  case TimestepMethod::RK3SSP   :
    m_ode_solver = new mfem::RK3SSPSolver;
    break;
  case TimestepMethod::RK4    :
    m_ode_solver = new mfem::RK4Solver;
    break;
  case TimestepMethod::GeneralizedAlpha :
    m_ode_solver = new mfem::GeneralizedAlphaSolver(0.5);
    break;
  case TimestepMethod::ImplicitMidpoint :
    m_ode_solver = new mfem::ImplicitMidpointSolver;
    break;
  case TimestepMethod::SDIRK23    :
    m_ode_solver = new mfem::SDIRK23Solver;
    break;
  case TimestepMethod::SDIRK34    :
    m_ode_solver = new mfem::SDIRK34Solver;
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

void BaseSolver::InitializeOutput(const OutputType output_type, const mfem::Array<std::string> names)
{
  MFEM_ASSERT(names.Size() == m_state_gf.Size(), "State vector and name arrays are not the same size.");

  m_state_names = names;
  m_output_type = output_type;

  switch(m_output_type) {
  case OutputType::VisIt: {
    m_visit_dc = new mfem::VisItDataCollection("serac", m_pmesh);

    for (int i=0; i<m_state_names.Size(); ++i) {
      m_visit_dc->RegisterField(m_state_names[i], m_state_gf[i]);
    }
    break;
  }

  case OutputType::GLVis: {
    std::ostringstream mesh_name;
    mesh_name << "serac-mesh." << std::setfill('0') << std::setw(6) << m_rank - 1;
    std::ofstream omesh(mesh_name.str().c_str());
    omesh.precision(8);
    m_pmesh->Print(omesh);
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
    for (int i=0; i<m_state_gf.Size(); ++i) {
      std::ostringstream sol_name;
      sol_name << m_state_names[i] << "." << std::setfill('0') << std::setw(6) << m_cycle << "." << std::setfill('0') << std::setw(
                 6) << m_rank - 1;
      std::ofstream osol(sol_name.str().c_str());
      osol.precision(8);
      m_state_gf[i]->Save(osol);
    }
    break;
  }

  default:
    mfem::mfem_error("OutputType not recognized!");
  }
}

BaseSolver::~BaseSolver()
{
  for (int i=0; i<m_fespaces.Size(); ++i) {
    delete m_fecolls[i];
    delete m_fespaces[i];
    delete m_state_gf[i];
    delete m_true_vec[i];
  }
  delete m_ode_solver;
  delete m_visit_dc;
}

