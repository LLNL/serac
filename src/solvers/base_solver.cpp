// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "base_solver.hpp"
#include "common/serac_types.hpp"

BaseSolver::BaseSolver(mfem::Array<mfem::ParGridFunction*> &stategf)
  : m_fespaces(stategf.Size()), m_state_gf(stategf), m_visit_dc(nullptr)
{
  MFEM_ASSERT(stategf.Size() > 0, "State vector array of size 0 in BaseSolver constructor.");

  for (int i=0; i<stategf.Size(); ++i) {
    m_fespaces[i] = m_state_gf[i]->ParFESpace();
  }
  m_pmesh = m_fespaces[0]->GetParMesh(); 

  m_time = 0.0;
  m_cycle = 0;
  MPI_Comm_rank(m_fespaces[0]->GetComm(), &m_rank); 
}

void BaseSolver::SetEssentialBCs(const mfem::Array<int> &ess_bdr, const mfem::Array<int> &ess_bdr_coef)
{
  m_ess_bdr = ess_bdr;
  m_ess_bdr_coef = &ess_bdr_coef;
}

void BaseSolver::SetNaturalBCs(const mfem::Array<int> &nat_bdr, const mfem::Array<int> &nat_bdr_coef)
{
  m_nat_bdr = nat_bdr;
  m_nat_bdr_coef = &nat_bdr_coef;
}

void BaseSolver::SetState(const mfem::Array<mfem::ParGridFunction*> &state_gf)
{
  m_state_gf = state_gf;
}

mfem::Array<mfem::ParGridFunction*> BaseSolver::GetState() const
{
  return m_state_gf;
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

  switch(m_output_type)
  {
    case OutputType::VisIt:
      m_visit_dc = new VisItDataCollection("serac", m_pmesh);
      
      for (int i=0; i<m_state_names.Size(); ++i) {
	m_visit_dc->RegisterField(m_state_names(i), m_state_gf(i));
      }	
      break;

    case OutputType::GLVis:
      std::ostringstream mesh_name;
      mesh_name << "serac-mesh" << setfill('0') << setw(6) << m_rank;
      std::ofstream omesh(mesh_name.str().c_str());
      omesh.precision(8);
      pmesh->Print(omesh);
      break;

    default:
      mfem_error("OutputType not recognized!");	  
  }
}

void BaseSolver::OutputState() const
{
  switch(m_output_type)
  {
    case VisIt:
      m_visit_dc.SetCycle(m_cycle);
      m_visit_dc.SetTime(m_time);
      m_visit_dc.Save();
    case GLVis:
      for (int i=0; i<m_state_gf->Size(); ++i) {
	std::ostringstream sol_name;
	sol_name << m_state_names(i) << setfill('0') << setw(6) << m_cycle << setfill('0') << setw(6) << m_rank;
	std::ofstream osol(sol_name.str().c_str());
	osol.precision(8);
	m_state_gf(i)->Save(osol);
      }
    default:
      mfem_error("OutputType not recognized!");
  }

}
