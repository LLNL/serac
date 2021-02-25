// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/physics/utilities/solver_config.hpp"

namespace serac {

constexpr int NUM_FIELDS = 3;

ThermalSolid::ThermalSolid(int order, std::shared_ptr<mfem::ParMesh> mesh,
                           const ThermalConduction::SolverOptions& therm_options,
                           const NonlinearSolid::SolverOptions&    solid_options)
    : BasePhysics(mesh, NUM_FIELDS, order),
      therm_solver_(order, mesh, therm_options),
      solid_solver_(order, mesh, solid_options),
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

ThermalSolid::ThermalSolid(std::shared_ptr<mfem::ParMesh> mesh, const ThermalConduction::InputOptions& thermal_input,
                           const NonlinearSolid::InputOptions& solid_input)
    : BasePhysics(mesh, NUM_FIELDS, std::max(thermal_input.order, solid_input.order)),
      therm_solver_(mesh, thermal_input),
      solid_solver_(mesh, solid_input),
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
  SLIC_ERROR_ROOT_IF(coupling_ != serac::CouplingScheme::OperatorSplit,
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
    SLIC_ERROR_ROOT_IF(std::abs(dt - initial_dt) > 1.0e-6,
                       "Operator split coupled solvers cannot adaptively change the timestep");
  } else {
    SLIC_ERROR_ROOT("Only operator split coupling is currently implemented");
  }

  cycle_ += 1;
}

}  // namespace serac
