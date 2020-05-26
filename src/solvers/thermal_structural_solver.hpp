// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef THERMSTRUCT_SOLVER
#define THERMSTRUCT_SOLVER

#include "base_solver.hpp"
#include "mfem.hpp"
#include "nonlinear_solid_solver.hpp"
#include "thermal_solver.hpp"

/// The coupled thermal-structural solver object
class ThermalStructuralSolver : public BaseSolver {
 protected:
  /// The state variables
  std::shared_ptr<FiniteElementState> m_temperature;
  std::shared_ptr<FiniteElementState> m_velocity;
  std::shared_ptr<FiniteElementState> m_displacement;

  /// The thermal solver object
  ThermalSolver m_therm_solver;

  /// The nonlinear solid solver object
  NonlinearSolidSolver m_solid_solver;

  /// The coupling strategy
  CouplingScheme m_coupling;

 public:
  /// Constructor from order and parallel mesh
  ThermalStructuralSolver(int order, std::shared_ptr<mfem::ParMesh> pmesh);

  /// Complete the data structure initialization
  void CompleteSetup();

  /// Advance the timestep
  void AdvanceTimestep(double &dt);

  /// Destructor
  virtual ~ThermalStructuralSolver();
};

#endif
