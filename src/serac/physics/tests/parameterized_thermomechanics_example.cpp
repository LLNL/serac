// Copyright (c) 2019-2023, Lawrence Livermore National Security, LLC and
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
#include "serac/physics/thermomechanics.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;

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

TEST(Thermomechanics, ParameterizedMaterial)
{
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

  SolidMechanics<p, dim, Parameters<H1<p>, H1<p>>> simulation(
      solid_mechanics::default_nonlinear_options, solid_mechanics::direct_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermomechanics_simulation");

  double density   = 1.0;     ///< density
  double E         = 1000.0;  ///< Young's modulus
  double nu        = 0.25;    ///< Poisson's ratio
  double theta_ref = 0.0;     ///< datum temperature for thermal expansion

  ParameterizedThermoelasticMaterial material{density, E, nu, theta_ref};

  simulation.setMaterial(DependsOn<0, 1>{}, material);

  double             deltaT = 1.0;
  FiniteElementState temperature(StateManager::newState(FiniteElementState::Options{.order = p, .name = "theta"}));
  temperature = theta_ref;
  simulation.setParameter(0, temperature);

  double             alpha0    = 1.0e-3;
  auto               alpha_fec = std::unique_ptr<mfem::FiniteElementCollection>(new mfem::H1_FECollection(p, dim));
  FiniteElementState alpha(StateManager::newState(FiniteElementState::Options{.order = p, .name = "alpha"}));
  alpha = alpha0;
  simulation.setParameter(1, alpha);

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
  temperature = theta_ref + deltaT;
  simulation.advanceTimestep(dt);

  simulation.outputState("paraview");

  // define quantities of interest
  auto& mesh = serac::StateManager::mesh();

  Functional<double(H1<p, dim>)> qoi({&simulation.displacement().space()});
  qoi.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](auto position, auto displacement) {
        auto [X, dX_dxi] = position;
        auto [u, du_dxi] = displacement;
        auto n           = normalize(cross(dX_dxi));
        return dot(u, n) * ((X[2] > 0.99 * height) ? 1.0 : 0.0);
      },
      mesh);

  double initial_qoi = qoi(simulation.displacement());
  SLIC_INFO_ROOT(axom::fmt::format("vertical displacement integrated over the top surface: {}", initial_qoi));

  Functional<double(H1<p, dim>)> area({&simulation.displacement().space()});
  area.AddSurfaceIntegral(
      DependsOn<>{},
      [=](auto position) {
        auto [X, dX_dxi] = position;
        return (X[2] > 0.99 * height) ? 1.0 : 0.0;
      },
      mesh);

  double top_area = area(simulation.displacement());

  SLIC_INFO_ROOT(axom::fmt::format("total area of the top surface: {}", top_area));

  double exact_area = M_PI_4 * ((outer_radius * outer_radius) - (inner_radius * inner_radius));

  SLIC_INFO_ROOT(axom::fmt::format("exact area of the top surface: {}", exact_area));

  double avg_disp = qoi(simulation.displacement()) / area(simulation.displacement());

  SLIC_INFO_ROOT(axom::fmt::format("average vertical displacement: {}", avg_disp));

  SLIC_INFO_ROOT(axom::fmt::format("expected average vertical displacement: {}", alpha0 * deltaT * height));

  serac::FiniteElementDual adjoint_load(simulation.displacement().space(), "adjoint_load");
  auto                     dqoi_du = get<1>(qoi(DifferentiateWRT<0>{}, simulation.displacement()));
  adjoint_load                     = *assemble(dqoi_du);

  check_gradient(qoi, simulation.displacement());

  simulation.solveAdjoint({{"displacement", adjoint_load}});

  auto& dqoi_dalpha = simulation.computeSensitivity(1);

  double epsilon = 1.0e-5;
  auto   dalpha  = alpha.CreateCompatibleVector();
  dalpha         = 1.0;
  alpha.Add(epsilon, dalpha);

  // rerun the simulation to the beginning,
  // but this time use perturbed values of alpha
  simulation.advanceTimestep(dt);

  simulation.outputState("paraview");

  double final_qoi = qoi(simulation.displacement());

  double adjoint_qoi_derivative = mfem::InnerProduct(dqoi_dalpha, dalpha);
  double fd_qoi_derivative      = (final_qoi - initial_qoi) / epsilon;

  // compare the expected change in the QoI to the actual change:
  SLIC_INFO_ROOT(
      axom::fmt::format("directional derivative of QoI by adjoint-state method: {}", adjoint_qoi_derivative));
  SLIC_INFO_ROOT(axom::fmt::format("directional derivative of QoI by finite-difference:    {}", fd_qoi_derivative));

  EXPECT_NEAR(0.0, (fd_qoi_derivative - adjoint_qoi_derivative) / fd_qoi_derivative, 3.0e-5);
}

// output:
// vertical displacement integrated over the top surface: 0.000883477
// total area of the top surface: 0.441077
// exact area of the top surface: 0.441786
// average vertical displacement: 0.001999
// expected average vertical displacement: 0.002
// directional derivative of QoI by adjoint-state method: 0.8812734293294495
// directional derivative of QoI by finite-difference:    0.8812609461498273

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
