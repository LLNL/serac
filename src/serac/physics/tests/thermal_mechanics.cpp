// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/thermomechanics.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

template <int p>
void functional_test_static_3D(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  const NonlinearSolverOptions default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-8, .max_iterations = 10, .print_level = 1};

  Thermomechanics<p, dim> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_static_options, default_nonlinear_options, default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double rho       = 1.0;
  double E         = 1.0;
  double nu        = 0.25;
  double c         = 1.0;
  double alpha     = 1.0e-3;
  double theta_ref = 1.0;
  double k         = 1.0;

  using Material = GreenSaintVenantThermoelasticMaterial;
  
  Material material{rho, E, nu, c, alpha, theta_ref, k};
  Material::State initial_state{};
  auto qdata = thermal_solid_solver.createQuadratureDataBuffer(initial_state);
  thermal_solid_solver.setMaterial(material, qdata);

  // Define the function for the initial temperature and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  thermal_solid_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solid_solver.setTemperature(one);

  // Define the function for the disolacement boundary condition
  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };

  // Set the initial displcament and boundary condition
  thermal_solid_solver.setDisplacementBCs(ess_bdr, zeroVector);
  thermal_solid_solver.setDisplacement(zeroVector);

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  // Perform the quasi-static solve
  thermal_solid_solver.advanceTimestep(1.0);

  EXPECT_NEAR(expected_norm, norm(thermal_solid_solver.displacement()), 1.0e-6);

  // Check the final temperature norm
  double temperature_norm_exact = 2.0 * std::sqrt(2.0);
  EXPECT_NEAR(temperature_norm_exact, norm(thermal_solid_solver.temperature()), 1.0e-6);
}

template <int p>
void functional_test_shrinking_3D(double expected_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1};
  std::set<int> temp_bdr       = {1, 2, 3};

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  const NonlinearSolverOptions default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-8, .max_iterations = 10, .print_level = 1};

  Thermomechanics<p, dim> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_static_options, default_nonlinear_options, default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double                                       rho       = 1.0;
  double                                       E         = 1.0;
  double                                       nu        = 0.0;
  double                                       c         = 1.0;
  double                                       alpha     = 1.0e-3;
  double                                       theta_ref = 2.0;
  double                                       k         = 1.0;
  GreenSaintVenantThermoelasticMaterial        material{rho, E, nu, c, alpha, theta_ref, k};
  GreenSaintVenantThermoelasticMaterial::State initial_state{};
  auto                                         qdata = thermal_solid_solver.createQuadratureDataBuffer(initial_state);
  thermal_solid_solver.setMaterial(material, qdata);

  // Define the function for the initial temperature
  double theta_0                   = 1.0;
  auto   initial_temperature_field = [theta_0](const mfem::Vector&, double) -> double { return theta_0; };

  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial temperature and boundary condition
  // thermal_solid_solver.setTemperatureBCs(ess_bdr, theta_0);
  thermal_solid_solver.setTemperatureBCs(temp_bdr, one);
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // Define the function for the displacement boundary condition
  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };

  // Set the initial displacement and boundary condition
  thermal_solid_solver.setDisplacementBCs(constraint_bdr, zeroVector);
  thermal_solid_solver.setDisplacement(zeroVector);

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  // Perform the quasi-static solve
  thermal_solid_solver.advanceTimestep(1.0);

  // Check the final displacement norm
  EXPECT_NEAR(expected_norm, norm(thermal_solid_solver.displacement()), 1.0e-4);
}

// TODO: investigate this failing test
template <int p>
void parameterized()
{
  MPI_Barrier(MPI_COMM_WORLD);

  mfem::DenseMatrix A(3);
  A       = 0.0;
  A(0, 0) = 0.523421770118331;
  A(0, 1) = 0.207205376077508;
  A(0, 2) = 0.600042309223256;
  A(1, 0) = 0.437180599730879;
  A(1, 1) = 0.095947283836495;
  A(1, 2) = 0.017796825926619;
  A(2, 0) = 0.149663987551694;
  A(2, 1) = 0.845137263999642;
  A(2, 2) = 0.594085227873111;
  mfem::Vector b(3);
  b(0)                             = 0.072393541428884;
  b(1)                             = 0.020326864245481;
  b(2)                             = 0.077181916474764;
  auto exact_displacement_function = [&A, &b](const mfem::Vector& X, mfem::Vector& u) {
    A.Mult(X, u);
    u += b;
  };

  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_functional_static_solve");

  // This mesh leads to perfectly straight-sided hexes, which
  // leaves the Jacobians of the element transformations constant.
  // The test should be made stronger by having non-constant
  // Jacobians. For a problem with tractions, the surface
  // facets should be non-affine as well.
  auto mesh =
      mesh::refineAndDistribute(buildCuboidMesh(4, 4, 4, 0.25, 0.25, 0.25), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh}"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct the thermomechanics solver module using the default equation solver parameters for both the heat transfer
  // and solid mechanics solves.
  Thermomechanics<p, dim, H1<p>> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_static_options, solid_mechanics::default_nonlinear_options,
      solid_mechanics::default_linear_options, solid_mechanics::default_quasistatic_options,
      GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double rho       = 1.0;
  double E         = 1.0;
  double nu        = 0.0;
  double c         = 1.0;
  double alpha0    = 0.0;  // 1.0e-3;
  double theta_ref = 2.0;
  double k         = 1.0;

  ParameterizedGreenSaintVenantThermoelasticMaterial        material{rho, E, nu, c, alpha0, theta_ref, k};
  ParameterizedGreenSaintVenantThermoelasticMaterial::State initial_state{};
  auto qdata = thermal_solid_solver.createQuadratureDataBuffer(initial_state);
  thermal_solid_solver.setMaterial(material, qdata);

  // parameterize the coefficient of thermal expansion
  FiniteElementState thermal_expansion_scaling(pmesh, H1<p>{}, "CTE scale");
  thermal_expansion_scaling = 1.5;

  std::function<double(const mfem::Vector&, double)> f = [](const mfem::Vector& /*x*/, double /*t*/) {
    return 1.0;  // 1.0 - x[0] * 0.5;
  };
  mfem::FunctionCoefficient coef(f);
  thermal_expansion_scaling.project(coef);
  thermal_solid_solver.setParameter(0, thermal_expansion_scaling);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1, 2, 3, 4, 5, 6};
  std::set<int> temp_bdr       = {1, 2, 3, 4, 5, 6};

  // Set a constant initial temperature, at the thermal expansion reference temperature value.
  auto initial_temperature_field = [theta_ref](const mfem::Vector&, double) -> double { return theta_ref; };
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // set a constant temperature boundary condition
  // set it below the reference temperature
  auto one = [theta_ref](const mfem::Vector&, double) -> double { return theta_ref - 0.0; };
  thermal_solid_solver.setTemperatureBCs(temp_bdr, one);

  // displacement boundary condition
  thermal_solid_solver.setDisplacementBCs(constraint_bdr, exact_displacement_function);

  // Cheating initial guess
  auto near_exact_displacement_function = [&A, &b](const mfem::Vector& X, mfem::Vector& u) {
    A.Mult(X, u);
    u += b;
    u *= 0.99;
  };
  thermal_solid_solver.setDisplacement(near_exact_displacement_function);

  // double G = 0.5*E/(1.0 + nu);
  // double K = E/3.0/(1.0 - 2.0*nu);
  // thermal_solid_solver.addBodyForce(
  //     [K, G, alpha0](const tensor<double, 3>& /*x*/, const double /*t*/, const auto& /*u*/, const auto& /*dudx*/) {
  //       double dummy = K*G*alpha0;
  //       return tensor<double, 3>{{0.0, 0.0, 0.0}};
  //     });
  //{{6.0*K*((1.0 - x[0]*0.5)*alpha0 - 0.5*x[0]) - 8.0/3.0*G - 2.0*K, 0.0, 0.0}};

  thermal_solid_solver.completeSetup();

  // dump initial state to output
  thermal_solid_solver.outputStateToDisk("pv_output");

  for (int i = 0; i < 4; i++) {
    // Perform the quasi-static solve
    double dt = 1.0;
    thermal_solid_solver.advanceTimestep(dt);

    thermal_solid_solver.outputStateToDisk();

    // Compute norm of error in numerical solution
    // auto exact_solution_coef = std::make_shared<mfem::VectorFunctionCoefficient>(
    //     dim, exact_displacement_function);
    mfem::VectorFunctionCoefficient exact_solution_coef(dim, exact_displacement_function);
    double error_norm = thermal_solid_solver.displacement().gridFunction().ComputeL2Error(exact_solution_coef);

    std::cout << error_norm << std::endl;
  }

  // EXPECT_LT(error_norm, 1e-10);
}

}  // namespace serac

TEST(Thermomechanics, staticTest)
{
  constexpr int p = 2;
  serac::functional_test_static_3D<p>(0.0);
}

TEST(Thermomechanics, thermalContraction)
{
  constexpr int p = 2;
  // this is the small strain solution, which works with a loose enought tolerance
  // TODO work out the finite deformation solution
  double alpha       = 1e-3;
  double L           = 8;
  double delta_theta = 1.0;
  serac::functional_test_shrinking_3D<p>(std::sqrt(L * L * L / 3.0) * alpha * delta_theta);
}

TEST(Thermomechanics, parameterized)
{
  // this is the small strain solution, which works with a loose enought tolerance
  // TODO work out the finite deformation solution
  // constexpr int p = 2;
  // serac::parameterized<p>();
}

namespace serac {

/// @brief Material with a fictitous constant heat source for testing
struct TestThermoelasticMaterial {
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;     ///<  thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double k;          ///< thermal conductivity

  using State = Empty;  ///< this material has no internal variables

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   * @tparam T4 Type of the coefficient of thermal expansion scale factor
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * Cauchy stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3>
  auto operator()(State&, const tensor<T1, 3, 3>& grad_u, T2 theta, 
                  const tensor<T3, 3>& grad_theta) const
  {
    const double          K     = E / (3.0 * (1.0 - 2.0 * nu));
    const double          G     = 0.5 * E / (1.0 + nu);
    static constexpr auto I     = Identity<3>();
    auto                  F     = grad_u + I;
    const auto            Eg    = greenStrain(grad_u);
    const auto            trEg  = tr(Eg);

    // stress
    const auto S     = 2.0 * G * dev(Eg) + K * (trEg - 3.0 * alpha * (theta - theta_ref)) * I;
    const auto P     = dot(F, S);
    const auto sigma = (dot(P, transpose(F))) / det(F);

    // internal heat source
    const auto s0 = 1.0;

    // heat flux
    const auto q0 = -k * grad_theta;

    return serac::tuple{sigma, C_v, s0, q0};
  }
};


/// @brief Material with a fictitous constant heat source for testing
struct J2ThermoelasticMaterial {
  static constexpr int dim = 3;

  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double k;          ///< thermal conductivity
  double Hi; ///< isotropic hardening modulus
  double sigma_y; ///< yield strength

  /// @brief variables required to characterize the hysteresis response
  struct State {
    tensor<double, dim, dim> plastic_strain;              ///< plastic strain
    double                   accumulated_plastic_strain;  ///< uniaxial equivalent plastic strain
    double delta_eqps; ///< previous increment of accumulated plastic strain
  };

  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   * @tparam T4 Type of the coefficient of thermal expansion scale factor
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * Cauchy stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3>
  auto operator()(State& state, const tensor<T1, dim, dim>& du_dX, T2 theta, 
                  const tensor<T3, dim>& dtheta_dX) const
  {
    const double          K     = E / (3.0 * (1.0 - 2.0 * nu));
    const double          G     = 0.5 * E / (1.0 + nu);
    auto el_strain = sym(du_dX) - state.plastic_strain;
    
    auto s = 2.0 * G * dev(el_strain);
    using std::sqrt;
    auto mises = sqrt(1.5) * norm(s);
    double Y_n = sigma_y + Hi*state.accumulated_plastic_strain;

    auto delta_eqps = (mises - Y_n)/(3*G + Hi);

    if (mises > Y_n)
    {
      auto N = 1.5 * s / mises;
      s -= 2.0*G*delta_eqps*N;
      state.accumulated_plastic_strain += get_value(delta_eqps);
      state.plastic_strain += get_value(delta_eqps)*get_value(N);
    }
    else
    {
      delta_eqps = 0;
    }
    auto sigma = s + K*tr(el_strain)*Identity<3>();

    // internal heat source
    std::cout << "delta_eqps = " << state.delta_eqps << std::endl;
    auto src = sigma_y*state.delta_eqps;

    // heat flux
    const auto q0 = -k * dtheta_dX;

    state.delta_eqps = get_value(delta_eqps);

    return serac::tuple{sigma, C_v, src, q0};
  }
};

TEST(Thermomechanics, SelfHeatingConstant)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  constexpr int p = 1;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "doesntmatter");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/onehex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1};
  std::set<int> temp_bdr       = {1, 2, 3};

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  const NonlinearSolverOptions default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-8, .max_iterations = 10, .print_level = 1};

  Thermomechanics<p, dim> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_timestepping_options, default_nonlinear_options, default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double                                       rho       = 1.0;
  double                                       E         = 0.5;
  double                                       nu        = 0.25;
  double                                       c         = 0.1;
  double                                       alpha     = 1.0;
  double                                       theta_ref = 100.0;
  double                                       k         = 1.0;
  TestThermoelasticMaterial        material{rho, E, nu, c, alpha, theta_ref, k};

  thermal_solid_solver.setMaterial(material);

  // Define the function for the initial temperature
  double theta_0                   = theta_ref;
  auto   initial_temperature_field = [theta_0](const mfem::Vector&, double) -> double { return theta_0; };

  auto zero_function = [](const mfem::Vector&, double) -> double { return 0.0; };

  // Set the initial temperature
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // Define the function for the displacement boundary conditions
  auto applied_disp = [](const mfem::Vector&) { return 1e-3; };

  // Set boundary conditions
  thermal_solid_solver.setDisplacementBCs({1}, zero_function, 0);
  thermal_solid_solver.setDisplacementBCs({2}, zero_function, 1);
  thermal_solid_solver.setDisplacementBCs({3}, zero_function, 2);
  thermal_solid_solver.setDisplacementBCs({6}, applied_disp, 0);
  thermal_solid_solver.setDisplacementBCs({4}, applied_disp, 1);
  thermal_solid_solver.setDisplacementBCs({5}, applied_disp, 2);

  thermal_solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  std::cout << "Temperatures at time step 0" << std::endl;
  thermal_solid_solver.temperature().Print();

  thermal_solid_solver.advanceTimestep(1.0);

  std::cout << "Temperatures at time step 1" << std::endl;
  thermal_solid_solver.temperature().Print();
  
  thermal_solid_solver.outputStateToDisk("self_heating");
}

TEST(Thermomechanics, SelfHeatingJ2)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  constexpr int p = 1;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermoJ2");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1};
  std::set<int> temp_bdr       = {1, 2, 3};

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  const NonlinearSolverOptions default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-8, .max_iterations = 10, .print_level = 1};

  Thermomechanics<p, dim> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_timestepping_options, default_nonlinear_options, default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double rho     = 1.0;
  double E       = 10.0;
  double nu      = 0.25;
  double c       = 0.1;
  double k       = 1.0;
  double Hi      = E/20.0;
  double sigma_y = 0.001;

  using Material = J2ThermoelasticMaterial;
  Material material{rho, E, nu, c, k, Hi, sigma_y};

  auto qdata = thermal_solid_solver.createQuadratureDataBuffer(Material::State{});
  
  thermal_solid_solver.setMaterial(material, qdata);

  // Define the function for the initial temperature
  double theta_0                   = 100.0;
  auto   initial_temperature_field = [theta_0](const mfem::Vector&, double) -> double { return theta_0; };

  auto zero_function = [](const mfem::Vector&, double) -> double { return 0.0; };

  // Set the initial temperature
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // Define the function for the displacement boundary conditions
  auto applied_disp = [](const mfem::Vector&, double t) { return 1e0*t; };

  // Set boundary conditions
  thermal_solid_solver.setDisplacementBCs({1}, [](const mfem::Vector&, mfem::Vector& u){ u = 0.0; });
  thermal_solid_solver.setDisplacementBCs({2}, applied_disp, 0);
  

  thermal_solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  std::cout << "Temperatures at time step 0" << std::endl;
  thermal_solid_solver.temperature().Print();

  for (int step = 1; step < 3; step++) {
    std::cout << "------------------------------------------------" << std::endl;;
    std::cout << "TIME STEP " << step << std::endl;
    thermal_solid_solver.advanceTimestep(1.0);
    std::cout << "Temperatures at time step" << step << std::endl;
    thermal_solid_solver.temperature().Print();
    thermal_solid_solver.outputStateToDisk("self_heating");
  }
}

} // namespace serac

namespace serac {
TEST(Thermomechanics, SelfHeating) 
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim                 = 3;
  constexpr int p = 1;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "self_heating");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/onehex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Define a boundary attribute set
  std::set<int> constraint_bdr = {1};
  std::set<int> temp_bdr       = {1, 2, 3};

  // define the solid solver configurations
  // no default solver options for solid yet, so make some here
  const LinearSolverOptions default_linear_options = {.linear_solver  = LinearSolver::GMRES,
                                                      .preconditioner = Preconditioner::HypreAMG,
                                                      .relative_tol   = 1.0e-6,
                                                      .absolute_tol   = 1.0e-10,
                                                      .max_iterations = 500,
                                                      .print_level    = 0};

  const NonlinearSolverOptions default_nonlinear_options = {
      .relative_tol = 1.0e-4, .absolute_tol = 1.0e-8, .max_iterations = 10, .print_level = 1};

  Thermomechanics<p, dim> thermal_solid_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_timestepping_options, default_nonlinear_options, default_linear_options,
      solid_mechanics::default_quasistatic_options, GeometricNonlinearities::On, "thermal_solid_functional", mesh_tag);

  double                                       rho       = 1.0;
  double                                       E         = 0.5;
  double                                       nu        = 0.25;
  double                                       c         = 0.1;
  double                                       alpha     = 1.0;
  double                                       theta_ref = 100.0;
  double                                       k         = 1.0;
  GreenSaintVenantThermoelasticMaterial        material{rho, E, nu, c, alpha, theta_ref, k};
  GreenSaintVenantThermoelasticMaterial::State initial_state{};
  auto                                         qdata = thermal_solid_solver.createQuadratureDataBuffer(initial_state);
  thermal_solid_solver.setMaterial(material, qdata);

  // Define the function for the initial temperature
  double theta_0                   = theta_ref;
  auto   initial_temperature_field = [theta_0](const mfem::Vector&, double) -> double { return theta_0; };

  auto zero_function = [](const mfem::Vector&, double) -> double { return 0.0; };

  // Set the initial temperature
  thermal_solid_solver.setTemperature(initial_temperature_field);

  // Define the function for the displacement boundary conditions
  auto applied_disp = [](const mfem::Vector&) { return 1e-3; };

  // Set boundary conditions
  thermal_solid_solver.setDisplacementBCs({1}, zero_function, 0);
  thermal_solid_solver.setDisplacementBCs({2}, zero_function, 1);
  thermal_solid_solver.setDisplacementBCs({3}, zero_function, 2);
  thermal_solid_solver.setDisplacementBCs({6}, applied_disp, 0);
  thermal_solid_solver.setDisplacementBCs({4}, applied_disp, 1);
  thermal_solid_solver.setDisplacementBCs({5}, applied_disp, 2);

  thermal_solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& u) { u = 0.0; });

  std::cout << "got to point D" << std::endl;

  // Finalize the data structures
  thermal_solid_solver.completeSetup();

  std::cout << "got to point F" << std::endl;

  // Perform the quasi-static solve
  thermal_solid_solver.advanceTimestep(1.0);

  thermal_solid_solver.temperature().Print();
  thermal_solid_solver.outputStateToDisk("self_heating");
}

} // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
