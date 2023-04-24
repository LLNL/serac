// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermal_solid_legacy.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/numerics/solver_config.hpp"

namespace serac {

constexpr int NUM_FIELDS = 3;

ThermalSolidLegacy::ThermalSolidLegacy(int order, const ThermalConductionLegacy::SolverOptions& therm_options,
                                       const SolidLegacy::SolverOptions& solid_options, const std::string& name,
                                       mfem::ParMesh* pmesh)
    : BasePhysics(NUM_FIELDS, order, name, pmesh),
      // Note that the solid solver must be constructed before the thermal solver as it mutates the mesh node grid
      // function
      solid_solver_(order, solid_options, GeometricNonlinearities::On, FinalMeshOption::Deformed, name, pmesh),
      therm_solver_(order, therm_options, name, pmesh),
      temperature_(therm_solver_.temperature()),
      velocity_(solid_solver_.velocity()),
      displacement_(solid_solver_.displacement())
{
  // The temperature_, velocity_, displacement_ members are not currently used
  // but presumably will be needed when further coupling schemes are implemented
  // This calls the non-const version
  states_.push_back(&therm_solver_.temperature());
  states_.push_back(&solid_solver_.velocity());
  states_.push_back(&solid_solver_.displacement());
}

ThermalSolidLegacy::ThermalSolidLegacy(const ThermalConductionLegacy::InputOptions& thermal_input,
                                       const SolidLegacy::InputOptions& solid_input, const std::string& name)
    : BasePhysics(NUM_FIELDS, std::max(thermal_input.order, solid_input.order), name),
      // Note that the solid solver must be constructed before the thermal solver as it mutates the mesh node grid
      // function
      solid_solver_(solid_input, name),
      therm_solver_(thermal_input, name),
      temperature_(therm_solver_.temperature()),
      velocity_(solid_solver_.velocity()),
      displacement_(solid_solver_.displacement())
{
  // The temperature_, velocity_, displacement_ members are not currently used
  // but presumably will be needed when further coupling schemes are implemented
  // This calls the non-const version
  states_.push_back(&therm_solver_.temperature());
  states_.push_back(&solid_solver_.velocity());
  states_.push_back(&solid_solver_.displacement());
}

ThermalSolidLegacy::ThermalSolidLegacy(const ThermalSolidLegacy::InputOptions& thermal_solid_input,
                                       const std::string&                      name)
    : ThermalSolidLegacy(thermal_solid_input.thermal_input, thermal_solid_input.solid_input, name)
{
  if (thermal_solid_input.coef_thermal_expansion) {
    std::unique_ptr<mfem::Coefficient> cte(thermal_solid_input.coef_thermal_expansion->constructScalar());
    std::unique_ptr<mfem::Coefficient> ref_temp(thermal_solid_input.reference_temperature->constructScalar());

    setThermalExpansion(std::move(cte), std::move(ref_temp));
  }
}

void ThermalSolidLegacy::completeSetup()
{
  solid_solver_.completeSetup();
  therm_solver_.completeSetup();
}

// Advance the timestep
void ThermalSolidLegacy::advanceTimestep(double& dt)
{
  double initial_dt = dt;

  therm_solver_.advanceTimestep(dt);
  solid_solver_.advanceTimestep(dt);

  time_ += dt;

  SLIC_ERROR_ROOT_IF(std::abs(dt - initial_dt) > 1.0e-6,
                     "Operator split coupled solvers cannot adaptively change the timestep");

  cycle_ += 1;
}

void ThermalSolidLegacy::InputOptions::defineInputFileSchema(axom::inlet::Container& container)
{
  // The solid mechanics options
  auto& solid_solver_table = container.addStruct("solid", "Finite deformation solid mechanics module").required();
  serac::SolidLegacy::InputOptions::defineInputFileSchema(solid_solver_table);

  // The thermal conduction options
  auto& thermal_solver_table = container.addStruct("thermal_conduction", "Thermal conduction module").required();
  serac::ThermalConductionLegacy::InputOptions::defineInputFileSchema(thermal_solver_table);

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

serac::ThermalSolidLegacy::InputOptions FromInlet<serac::ThermalSolidLegacy::InputOptions>::operator()(
    const axom::inlet::Container& base)
{
  serac::ThermalSolidLegacy::InputOptions result;

  result.solid_input = base["solid"].get<serac::SolidLegacy::InputOptions>();

  result.thermal_input = base["thermal_conduction"].get<serac::ThermalConductionLegacy::InputOptions>();

  if (base.contains("coef_thermal_expansion")) {
    result.coef_thermal_expansion = base["coef_thermal_expansion"].get<serac::input::CoefficientInputOptions>();
    result.reference_temperature  = base["reference_temperature"].get<serac::input::CoefficientInputOptions>();
  }

  return result;
}
