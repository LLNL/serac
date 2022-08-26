// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/thermal_mechanics_functional.hpp"

using namespace serac;

using serac::solid_mechanics::default_static_options;

template <typename T>
auto greenStrain(const tensor<T, 3, 3>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

struct ParameterizedThermoelasticMaterial {
  using State = Empty;

  static constexpr int VALUE = 0, GRADIENT = 1;

  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double theta_ref;  ///< datum temperature for thermal expansion

  template <typename T1, typename T2, typename T3>
  auto operator()(State& /*state*/, const tensor<T1, 3, 3>& grad_u, T2 temperature,
                  T3 coefficient_of_thermal_expansion) const
  {
    auto theta = get<VALUE>(temperature);
    auto alpha = get<VALUE>(coefficient_of_thermal_expansion);

    const double          K    = E / (3.0 * (1.0 - 2.0 * nu));
    const double          G    = 0.5 * E / (1.0 + nu);
    static constexpr auto I    = Identity<3>();
    auto                  F    = grad_u + I;
    const auto            Eg   = greenStrain(grad_u);
    const auto            trEg = tr(Eg);

    const auto S = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto P = dot(F, S);
    return dot(P, transpose(F));
  }
};

int main(int argc, char* argv[])
{
  serac::initialize(argc, argv);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "parameterized_thermomechanics");

  size_t radial_divisions   = 3;
  size_t angular_divisions  = 16;
  size_t vertical_divisions = 8;

  double inner_radius = 1.0;
  double outer_radius = 1.25;
  double height       = 2.0;

  {
    // clang-format off
    auto mesh = mesh::refineAndDistribute(build_hollow_quarter_cylinder(radial_divisions, 
                                                                        angular_divisions, 
                                                                        vertical_divisions,
                                                                        inner_radius, 
                                                                        outer_radius, 
                                                                        height), serial_refinement, parallel_refinement);

    // clang-format on
    serac::StateManager::setMesh(std::move(mesh));
  }

  SolidFunctional<p, dim, Parameters<H1<p>, H1<p> > > simulation(default_static_options, GeometricNonlinearities::On,
                                                                 "thermomechanics_simulation");

  double density   = 1.0;     ///< density
  double E         = 1000.0;  ///< Young's modulus
  double nu        = 0.25;    ///< Poisson's ratio
  double theta_ref = 0.0;     ///< datum temperature for thermal expansion

  ParameterizedThermoelasticMaterial material{density, E, nu, theta_ref};

  simulation.setMaterial(DependsOn<0, 1>{}, material);

  auto temperature_fec = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::H1_FECollection(p, dim));
  FiniteElementState temperature(StateManager::newState(
      FiniteElementState::Options{.order = p, .coll = std::move(temperature_fec), .name = "theta"}));
  temperature = theta_ref;
  simulation.setParameter(temperature, 0);

  double             alpha0    = 1.0e-3;
  auto               alpha_fec = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::H1_FECollection(p, dim));
  FiniteElementState alpha(
      StateManager::newState(FiniteElementState::Options{.order = p, .coll = std::move(alpha_fec), .name = "alpha"}));
  alpha = alpha0;
  simulation.setParameter(alpha, 1);

  // set up essential boundary conditions
  std::set<int> x_equals_0 = {4};
  std::set<int> y_equals_0 = {2};
  std::set<int> z_equals_0 = {1};

  auto zero_scalar = [](const mfem::Vector&) -> double { return 0.0; };
  simulation.setDisplacementBCs(x_equals_0, zero_scalar, 0);
  simulation.setDisplacementBCs(y_equals_0, zero_scalar, 1);
  simulation.setDisplacementBCs(z_equals_0, zero_scalar, 2);

  // set up initial conditions
  auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  simulation.setDisplacement(zero_vector);

  // Finalize the data structures
  simulation.completeSetup();

  simulation.outputState("paraview");

  // Perform the quasi-static solve
  double dt   = 1.0;
  temperature = theta_ref + 1.0;
  simulation.advanceTimestep(dt);

  // define quantities of interest
  auto& mesh = serac::StateManager::mesh();

  Functional<double(H1<p, dim>)> qoi({&simulation.displacement().space()});
  qoi.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](auto x, auto n, auto displacement) {
        auto [u, du_dxi] = displacement;
        return dot(u, n) * ((x[2] > 0.99 * height) ? 1.0 : 0.0);
      },
      mesh);

  double initial_qoi = qoi(simulation.displacement());
  std::cout << "vertical displacement integrated over the top surface: " << initial_qoi << std::endl;

  Functional<double(H1<p, dim>)> area({&simulation.displacement().space()});
  area.AddSurfaceIntegral(DependsOn<>{}, [=](auto x, auto /*n*/) { return (x[2] > 0.99 * height) ? 1.0 : 0.0; }, mesh);

  std::cout << "total area of the top surface: " << area(simulation.displacement()) << std::endl;

  double exact_area = M_PI_4 * ((outer_radius * outer_radius) - (inner_radius * inner_radius));

  std::cout << "exact area of the top surface: " << exact_area << std::endl;

  std::cout << "average vertical displacement: " << qoi(simulation.displacement()) / area(simulation.displacement())
            << std::endl;

  double deltaT = 1.0;
  std::cout << "expected average vertical displacement: " << alpha0 * deltaT * height << std::endl;

  serac::FiniteElementDual adjoint_load(mesh, simulation.displacement().space(), "adjoint_load");
  auto                     dqoi_du = get<1>(qoi(DifferentiateWRT<0>{}, simulation.displacement()));
  adjoint_load                     = *assemble(dqoi_du);

  simulation.solveAdjoint(adjoint_load);

  auto& dqoi_dalpha = simulation.computeSensitivity<1>();

  double       epsilon = 1.0e-6;
  mfem::Vector dalpha(alpha.Size());
  dalpha.Randomize();
  alpha += epsilon * dalpha;

  // rerun the simulation to the beginning,
  // but this time use perturbed values of alpha
  simulation.setDisplacement(zero_vector);

  simulation.advanceTimestep(dt);

  double final_qoi = qoi(simulation.displacement());

  // compare the expected change in the QoI to the actual change:
  std::cout << "directional derivative of QoI by adjoint-state method: ";
  std::cout << mfem::InnerProduct(dqoi_dalpha, dalpha) << std::endl;
  std::cout << "directional derivative of QoI by finite-difference: ";
  std::cout << (final_qoi - initial_qoi) / epsilon << std::endl;

  serac::exitGracefully();
}

// output:
// vertical displacement integrated over the top surface: 0.000883477
// total area of the top surface: 0.441959
// exact area of the top surface: 0.441786
// average vertical displacement: 0.001999
// expected average vertical displacement: 0.002
// directional derivative of QoI by adjoint-state method: 0.441931
// directional derivative of QoI by finite-difference: 0.441931
