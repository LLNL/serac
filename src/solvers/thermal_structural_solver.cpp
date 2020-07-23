// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "thermal_structural_solver.hpp"

#include "common/serac_types.hpp"

const int num_fields = 3;

ThermalStructuralSolver::ThermalStructuralSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh)
    : BaseSolver(pmesh->GetComm(), num_fields, order), m_therm_solver(order, pmesh), m_solid_solver(order, pmesh)
{
  m_temperature  = m_therm_solver.GetTemperature();
  m_velocity     = m_solid_solver.GetVelocity();
  m_displacement = m_solid_solver.GetDisplacement();

  m_state[0] = m_temperature;
  m_state[1] = m_velocity;
  m_state[2] = m_displacement;

  m_coupling = serac::CouplingScheme::OperatorSplit;
}

void ThermalStructuralSolver::CompleteSetup()
{
  MFEM_VERIFY(m_coupling == serac::CouplingScheme::OperatorSplit,
              "Only operator split is currently implemented in the thermal structural solver.");

  m_therm_solver.CompleteSetup();
  m_solid_solver.CompleteSetup();
}

void ThermalStructuralSolver::SetTimestepper(serac::TimestepMethod timestepper)
{
  m_timestepper = timestepper;
  m_therm_solver.SetTimestepper(timestepper);
  m_solid_solver.SetTimestepper(timestepper);
}

// Advance the timestep
void ThermalStructuralSolver::AdvanceTimestep(double &dt)
{
  if (m_coupling == serac::CouplingScheme::OperatorSplit) {
    double initial_dt = dt;
    m_therm_solver.AdvanceTimestep(dt);
    m_solid_solver.AdvanceTimestep(dt);
    MFEM_VERIFY(std::abs(dt - initial_dt) < 1.0e-6,
                "Operator split coupled solvers cannot adaptively change the timestep");

  } else {
    MFEM_ABORT("Only operator split coupling is currently implemented");
  }

  m_cycle += 1;
}
