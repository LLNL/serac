// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "physics/thermal_solid.hpp"

#include "infrastructure/logger.hpp"
#include "physics/utilities/solver_config.hpp"

namespace serac {

constexpr int NUM_FIELDS = 3;

ThermalSolid::ThermalSolid(int order, std::shared_ptr<mfem::ParMesh> mesh,
                           const ThermalConduction::SolverParameters& therm_params,
                           const Solid::SolverParameters&             solid_params)
    : BasePhysics(mesh, NUM_FIELDS, order),
      therm_solver_(order, mesh, therm_params),
      solid_solver_(order, mesh, solid_params),
      temperature_(therm_solver_.temperature()),
      velocity_(solid_solver_.velocity()),
      displacement_(solid_solver_.displacement())
{
  // The temperature_, velocity_, displacement_ members are not currently used
  // but presumably will be needed when further coupling schemes are implemented
  // This calls the non-const version
  state_.push_back(therm_solver_.temperature());
  state_.push_back(solid_solver_.velocity());
  state_.push_back(solid_solver_.displacement());

  coupling_ = serac::CouplingScheme::OperatorSplit;
}

void ThermalSolid::completeSetup()
{
  SLIC_ERROR_ROOT_IF(coupling_ != serac::CouplingScheme::OperatorSplit, mpi_rank_,
                     "Only operator split is currently implemented in the thermal structural solver.");

  therm_solver_.completeSetup();
  solid_solver_.completeSetup();
}

// Advance the timestep
void ThermalSolid::advanceTimestep(double& dt)
{
  if (coupling_ == serac::CouplingScheme::OperatorSplit) {
    double initial_dt = dt;
    therm_solver_.advanceTimestep(dt);
    solid_solver_.advanceTimestep(dt);
    SLIC_ERROR_ROOT_IF(std::abs(dt - initial_dt) > 1.0e-6, mpi_rank_,
                       "Operator split coupled solvers cannot adaptively change the timestep");
  } else {
    SLIC_ERROR_ROOT(mpi_rank_, "Only operator split coupling is currently implemented");
  }

  cycle_ += 1;
}

}  // namespace serac
