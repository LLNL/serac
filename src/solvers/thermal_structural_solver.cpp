// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_structural_solver.hpp"

#include "common/serac_types.hpp"

const int num_fields = 3;

ThermalStructuralSolver::ThermalStructuralSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields), m_therm_solver(order, pmesh), m_solid_solver(order, pmesh)
{
  m_temperature  = m_therm_solver.GetTemperature();
  m_velocity     = m_solid_solver.GetVelocity();
  m_displacement = m_solid_solver.GetDisplacement();

  m_state[0] = m_temperature;
  m_state[1] = m_velocity;
  m_state[2] = m_displacement;

  m_coupling = CouplingScheme::OperatorSplit;
}

ThermalStructuralSolver::~ThermalStructuralSolver() {}
