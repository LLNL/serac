// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/solver_config.hpp"

namespace serac {

constexpr int NUM_FIELDS = 3;

ThermalSolid::ThermalSolid(int order, const ThermalConduction::SolverOptions& therm_options,
                           const Solid::SolverOptions& solid_options, const std::string& name)
    : BasePhysics(NUM_FIELDS, order),
      therm_solver_(order, therm_options, name),
      solid_solver_(order, solid_options, GeometricNonlinearities::On, FinalMeshOption::Deformed, name),
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

ThermalSolid::ThermalSolid(const ThermalConduction::InputOptions& thermal_input, const Solid::InputOptions& solid_input,
                           const std::string& name)
    : BasePhysics(NUM_FIELDS, std::max(thermal_input.order, solid_input.order)),
      therm_solver_(thermal_input, name),
      solid_solver_(solid_input, name),
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

ThermalSolid::ThermalSolid(const ThermalSolid::InputOptions& thermal_solid_input, const std::string& name)
    : ThermalSolid(thermal_solid_input.thermal_input, thermal_solid_input.solid_input, name)
{
  if (thermal_solid_input.coef_thermal_expansion) {
    std::unique_ptr<mfem::Coefficient> cte(thermal_solid_input.coef_thermal_expansion->constructScalar());
    std::unique_ptr<mfem::Coefficient> ref_temp(thermal_solid_input.reference_temperature->constructScalar());

    setThermalExpansion(std::move(cte), std::move(ref_temp));
  }
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

void ThermalSolid::InputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // The solid mechanics options
  auto& solid_solver_table = container.addStruct("solid", "Finite deformation solid mechanics module").required();
  serac::Solid::InputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  auto& thermal_solver_table = container.addStruct("thermal_conduction", "Thermal conduction module").required();
  serac::ThermalConduction::InputOptions::defineInputFileSchema(thermal_solver_table);

  auto& ref_temp = container.addStruct("reference_temperature",
                                       "Coefficient for the reference temperature for isotropic thermal expansion");
  serac::input::CoefficientInputOptions::defineInputFileSchema(ref_temp);

  auto& coef_therm_expansion =
      container.addStruct("coef_thermal_expansion", "Coefficient of thermal expansion for isotropic thermal expansion");
  serac::input::CoefficientInputOptions::defineInputFileSchema(coef_therm_expansion);

  container.registerVerifier([](const axom::inlet::Container& base) -> bool {
    bool cte_found      = base.contains("coef_thermal_expansion");
    bool ref_temp_found = base.contains("reference_temperature");

    if (ref_temp_found && cte_found) {
      return true;
    } else if ((!ref_temp_found) && (!cte_found)) {
      return true;
    }

    SLIC_WARNING_ROOT(
        "Either both a coefficient of thermal expansion and reference temperature should be specified"
        "in the thermal solid input file or neither should be.");

    return false;
  });
}

}  // namespace serac

serac::ThermalSolid::InputOptions FromInlet<serac::ThermalSolid::InputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::ThermalSolid::InputOptions result;

  result.solid_input = base["solid"].get<serac::Solid::InputOptions>();

  result.thermal_input = base["thermal_conduction"].get<serac::ThermalConduction::InputOptions>();

  if (base.contains("coef_thermal_expansion")) {
    result.coef_thermal_expansion = base["coef_thermal_expansion"].get<serac::input::CoefficientInputOptions>();
    result.reference_temperature  = base["reference_temperature"].get<serac::input::CoefficientInputOptions>();
  }

  return result;
}