// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <functional>

#include "mfem.hpp"

#include "serac/numerics/expr_template_ops.hpp"
#include "serac/physics/operators/odes.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"

using namespace serac;

DirichletEnforcementMethod enforcement_methods[3] = {
  DirichletEnforcementMethod::DirectControl,
  DirichletEnforcementMethod::RateControl,
  DirichletEnforcementMethod::FullControl
};

IterativeSolverOptions linear_options{
  .rel_tol     = 1.0e-12,
  .abs_tol     = 1.0e-12,
  .print_level = -1,
  .max_iter    = 200,
  .lin_solver  = LinearSolver::CG,
  .prec        = {}
};

NonlinearSolverOptions nonlinear_options{
  .rel_tol = 1.0e-12, 
  .abs_tol = 1.0e-12, 
  .max_iter = 500, 
  .print_level = -1
};

mfem::DenseMatrix M = []() {
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2.0;
  M(0, 1) = 1.0;
  M(0, 2) = 0.0;

  M(1, 0) = 1.0;
  M(1, 1) = 4.0;
  M(1, 2) = 1.0;

  M(2, 0) = 0.0;
  M(2, 1) = 1.0;
  M(2, 2) = 2.0;
  return M;
}();

mfem::DenseMatrix C = []() {
  mfem::DenseMatrix C(3, 3);
  C(0, 0) = 1.0;
  C(0, 1) = 0.0;
  C(0, 2) = 0.0;

  C(1, 0) = 0.0;
  C(1, 1) = 1.0;
  C(1, 2) = 0.0;

  C(2, 0) = 0.0;
  C(2, 1) = 0.0;
  C(2, 2) = 1.0;
  return C;
}();

std::function stiffness_linear = [](const mfem::Vector & /*x*/) -> mfem::DenseMatrix {
  mfem::DenseMatrix K(3, 3);
  K(0, 0) = 1.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0;
  K(1, 2) = -1.0;

  K(2, 0) = 0.0;
  K(2, 1) = -1.0;
  K(2, 2) = 1.0;
  return K;
};

std::function internal_force_linear = [](const mfem::Vector & x) -> mfem::Vector {
  return stiffness_linear(x) * x;
};

std::function stiffness_nonlinear = [](const mfem::Vector& x) -> mfem::DenseMatrix {
  mfem::DenseMatrix K(3, 3);
  K(0, 0) = 1.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0 - 2 * x(1) + x(2);
  K(1, 2) = -1.0 + x(1);

  K(2, 0) = 0.0;
  K(2, 1) = -1.0 + x(2);
  K(2, 2) = 1.0 + x(1) + x(2);
  return K;
};

std::function internal_force_nonlinear = [](const mfem::Vector & x) -> mfem::Vector {
  mfem::Vector f(3);
  f(0) = x(0) - x(1);
  f(1) = (2.0 - x(1)) * x(1) + (x(1) - 1.0) * x(2) - x(0);
  f(2) = x(1) * (x(2) - 1.0) + (1.0 + 0.5 * x(2)) * x(2);
  return f;
};

mfem::Vector f_ext = [](){
  mfem::Vector f(3);
  f(0) = 1.0;
  f(1) = 2.0;
  f(2) = 3.0;
  return f;
}();

// continuous constraint with continuous derivative
const auto sine_wave = [](const mfem::Vector& /*x*/, double t) { return 1.0 + sin(2.0 * M_PI * t); };

// continuous constraint with discontinuous derivative
const auto triangle_wave = [](const mfem::Vector& /*x*/, double t) { return 1.0 + (2.0 / M_PI) * asin(sin(2.0 * M_PI * t)); };

enum ode_type {LINEAR, NONLINEAR};
enum constraint_type {UNCONSTRAINED, SINE_WAVE, TRIANGLE_WAVE};

double first_order_ode_test(int nsteps, ode_type type, constraint_type constraint, DirichletEnforcementMethod enforcement) {

  double t = 0.0;
  double dt = 1.0 / nsteps;
  double previous_dt = -1.0;

  mfem::Vector x(3);
  mfem::Vector previous(3);
  previous = 0.0;

  mfem::DenseMatrix J(3, 3);

  mfem::Mesh               mesh1D(2);
  mfem::ParMesh            mesh(MPI_COMM_WORLD, mesh1D);
  BoundaryConditionManager bcs(mesh);

  // although these test problems don't really apply to a finite element mesh,
  // the tools in serac require that these finite element data structures exist
  serac::FiniteElementState dummy(mesh, FiniteElementState::Options{.order = 1, .name = "dummy"});
  dummy.initializeTrueVec();

  if (constraint == SINE_WAVE) {
    auto coef = std::make_shared< mfem::FunctionCoefficient >(sine_wave);
    bcs.addEssential({1}, coef, dummy);
  }

  if (constraint == TRIANGLE_WAVE) {
    auto coef = std::make_shared< mfem::FunctionCoefficient >(triangle_wave);
    bcs.addEssential({1}, coef, dummy);
  }

  std::function < mfem::Vector(const mfem::Vector &) > f_int;  
  std::function < mfem::DenseMatrix(const mfem::Vector &) > K;  

  if (type == LINEAR) {
    f_int = internal_force_linear;
    K = stiffness_linear;
  } else {
    f_int = internal_force_nonlinear;
    K = stiffness_nonlinear;
  }

  StdFunctionOperator residual(3, 
    [&](const mfem::Vector& dx_dt, mfem::Vector& r) { 
      r = M * dx_dt + f_int(x + dx_dt * dt) - f_ext; 
      if (constraint != UNCONSTRAINED) { 
        r(0) = 0.0; 
      }
    },
    [&](const mfem::Vector & dx_dt) -> mfem::Operator& {
      J = M;
      J.Add(dt, K(x + dx_dt * dt));
      if (constraint != UNCONSTRAINED) { 
        // clang-format off
        J(0,0) = 1.0; J(0,1) = 0.0; J(0,2) = 0.0;
        J(1,0) = 0.0; 
        J(2,0) = 0.0;
        // clang-format on
      }
      return J;
    }
  );

  EquationSolver solver(MPI_COMM_WORLD, linear_options, nonlinear_options);
  solver.SetOperator(residual);

  FirstOrderODE ode(dummy.space().TrueVSize(), {.u = x, .dt = dt, .du_dt = previous, .previous_dt = previous_dt}, solver, bcs);

  ode.SetTimestepper(TimestepMethod::BackwardEuler);
  ode.SetEnforcementMethod(enforcement);

  mfem::Vector soln(3);

  soln[0] = 1.0;
  soln[1] = 2.0;
  soln[2] = 3.0;

  for (int i = 0; i < nsteps; i++) {
    ode.Step(soln, t, dt);
  }

  // clang-format off
  double exact_solutions[2][3][3] = {
    { // Linear
      {1.716166172752257,2.283833827247743,3.716166172752257},
      {1.00000000000000,2.281907223306997,3.791971854505491},
      {1.00000000000000,2.293071599499620,3.777196781634047}
    },
    { // Nonlinear
      {1.668978479765538,2.689094594698356,1.338823562848719},
      {1.00000000000000,2.888713406376719,1.345973814264978},
      {1.00000000000000,2.869493799926669,1.338893129624946}
    }
  };
  // clang-format on

  mfem::Vector exact_solution(exact_solutions[type][constraint], 3);
  mfem::Vector error = (exact_solution - soln) / exact_solution.Norml2();

  return error.Norml2();

}

double second_order_ode_test(int nsteps, ode_type type, constraint_type constraint, DirichletEnforcementMethod enforcement) {

  double t = 0.0;
  double dt = 1.0 / nsteps;
  double c0, c1;

  mfem::Vector x(3);
  mfem::Vector dx_dt(3);
  mfem::Vector previous(3);
  previous = 0.0;

  mfem::DenseMatrix J(3, 3);

  mfem::Mesh               mesh1D(3);
  mfem::ParMesh            mesh(MPI_COMM_WORLD, mesh1D);
  BoundaryConditionManager bcs(mesh);

  // although these test problems don't really apply to a finite element mesh,
  // the tools in serac require that these finite element data structures exist
  serac::FiniteElementState dummy(mesh, FiniteElementState::Options{.order = 1, .name = "dummy"});
  dummy.initializeTrueVec();

  if (constraint == SINE_WAVE) {
    auto coef = std::make_shared< mfem::FunctionCoefficient >(sine_wave);
    bcs.addEssential({1}, coef, dummy);
  }

  if (constraint == TRIANGLE_WAVE) {
    auto coef = std::make_shared< mfem::FunctionCoefficient >(triangle_wave);
    bcs.addEssential({1}, coef, dummy);
  }

  std::function < mfem::Vector(const mfem::Vector &) > f_int;  
  std::function < mfem::DenseMatrix(const mfem::Vector &) > K;  

  if (type == LINEAR) {
    f_int = internal_force_linear;
    K = stiffness_linear;
  } else {
    f_int = internal_force_nonlinear;
    K = stiffness_nonlinear;
  }

  StdFunctionOperator residual(3, 
    [&](const mfem::Vector& d2x_dt2, mfem::Vector& r) { 
      r = M * d2x_dt2 + C * (dx_dt + c1 * d2x_dt2) + f_int(x + c0 * d2x_dt2) - f_ext; 
      if (constraint != UNCONSTRAINED) { 
        r(0) = 0.0; 
      }
    },
    [&](const mfem::Vector & d2x_dt2) -> mfem::Operator& {
      J = M;
      J.Add(c1, C);
      J.Add(c0, K(x + c0 * d2x_dt2));
      if (constraint != UNCONSTRAINED) { 
        // clang-format off
        J(0,0) = 1.0; J(0,1) = 0.0; J(0,2) = 0.0;
        J(1,0) = 0.0; 
        J(2,0) = 0.0;
        // clang-format on
      }
      return J;
    }
  );

  EquationSolver solver(MPI_COMM_WORLD, linear_options, nonlinear_options);
  solver.SetOperator(residual);

  SecondOrderODE ode(dummy.space().TrueVSize(), {.c0 = c0, .c1 = c1, .u = x, .du_dt = dx_dt, .d2u_dt2 = previous}, solver, bcs);

  ode.SetTimestepper(TimestepMethod::Newmark);
  ode.SetEnforcementMethod(enforcement);

  mfem::Vector displacement(3);
  displacement[0] = 1.0;
  displacement[1] = 2.0;
  displacement[2] = 3.0;

  mfem::Vector velocity(3);
  velocity[0] = 1.0;
  velocity[1] = -1.0;
  velocity[2] = 0.0;

  if (constraint == SINE_WAVE) { velocity[0] = 2.0 * M_PI; }
  if (constraint == TRIANGLE_WAVE) { velocity[0] = 4.0; }

  for (int i = 0; i < nsteps; i++) {
    ode.Step(displacement, velocity, t, dt);

    std::cout << t << " " << displacement[0] << " " << displacement[1] << " " << displacement[2];
    std::cout <<      " " << velocity[0] << " " << velocity[1] << " " << velocity[2] << std::endl;
  }

  // clang-format off
  double exact_displacements[2][3][3] = {
    { // Linear
      {1.890707657808391,1.440430856985806,3.167465070871345},
      {1.00000000000000,2.728890247925751,2.840954141998902},
      {1.00000000000000,1.397131173529700,3.189617467031896}
    },
    { // Nonlinear
      {1.843679427583925,1.603626250564599,1.427673131558248},
      {1.00000000000000,3.090535238075648,0.9902392742386004},
      {1.00000000000000,1.554092379030468,1.449923209556037}
    }
  };

  double exact_velocities[2][3][3] = {
    { // Linear
      {0.6308377824610962,0.002935983263988160,0.1970890557890808},
      {6.283185307179586,-1.339625209685756,1.572179558520680},
      {4.000000000000000,-0.2225975240781133,0.2976678735241257}
    },
    { // Nonlinear
      {0.7131365043381414,0.002179075289509970,-2.116046109865199},
      {6.283185307179586,-0.5523588186531320,-1.112916510614004},
      {4.000000000000000,-0.1879397476414202,-2.037365797741527}
    }
  };
  // clang-format on

  mfem::Vector exact_displacement(exact_displacements[type][constraint], 3);
  mfem::Vector exact_velocity(exact_velocities[type][constraint], 3);
  mfem::Vector error_displacement = (exact_displacement - displacement) / exact_displacement.Norml2();
  mfem::Vector error_velocity = (exact_velocity - velocity) / exact_velocity.Norml2();

  std::cout << "displacement constraint error: " << error_displacement(0) << std::endl;
  std::cout << "velocity constraint error: " << error_velocity(0) << std::endl;

  return std::max(error_displacement.Norml2(), error_velocity.Norml2());
}

#if 0
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);



  std::cout << second_order_ode_test(78, LINEAR, TRIANGLE_WAVE, enforcement_methods[0]) << std::endl;

  for (auto type : {LINEAR, NONLINEAR}) {
    for (auto constraint : {UNCONSTRAINED, SINE_WAVE, TRIANGLE_WAVE}) {
      for (auto enforcement : enforcement_methods) {

        {
          double errors[3] = {
            first_order_ode_test( 50, type, constraint, enforcement),
            first_order_ode_test(100, type, constraint, enforcement),
            first_order_ode_test(200, type, constraint, enforcement)
          };

          SLIC_ASSERT_MSG(((errors[0] / errors[1]) > 1.8) && ((errors[1] / errors[2]) > 1.8), "Error: Convergence rate lower than expected");
        }


      }
    }
  }

  MPI_Finalize();

  return 0;
}
#endif

using param_t = std::tuple <
  ode_type,
  constraint_type,
  TimestepMethod,
  DirichletEnforcementMethod
>;

class ODE_tests : public testing::TestWithParam< param_t > {
 protected:
  void SetUp() override {
    std::tie(type, constraint, timestepper, enforcement) = GetParam();
  }

  ode_type type;
  constraint_type constraint;
  TimestepMethod timestepper;
  DirichletEnforcementMethod enforcement;
};


TEST_P(ODE_tests, first_order)
{
  double errors[3] = {
    first_order_ode_test( 50, type, constraint, enforcement),
    first_order_ode_test(100, type, constraint, enforcement),
    first_order_ode_test(200, type, constraint, enforcement)
  };

  EXPECT_NEAR(2.0, (errors[0] / errors[1]), 0.1);
  EXPECT_NEAR(2.0, (errors[1] / errors[2]), 0.1);
}


// clang-format off
INSTANTIATE_TEST_SUITE_P(all_first_order_tests, ODE_tests,
  testing::Combine(
    testing::Values(
      LINEAR, 
      NONLINEAR
    ), 
    testing::Values(
      UNCONSTRAINED, 
      SINE_WAVE, 
      TRIANGLE_WAVE
    ), 
    testing::Values(
      serac::TimestepMethod::BackwardEuler,
      serac::TimestepMethod::SDIRK33,
      serac::TimestepMethod::ForwardEuler,
      serac::TimestepMethod::RK2,
      serac::TimestepMethod::RK3SSP,
      serac::TimestepMethod::RK4,
      serac::TimestepMethod::GeneralizedAlpha,
      serac::TimestepMethod::ImplicitMidpoint,
      serac::TimestepMethod::SDIRK23,
      serac::TimestepMethod::SDIRK34
    ),
    testing::Values(
      DirichletEnforcementMethod::DirectControl,       
      DirichletEnforcementMethod::RateControl,     
      DirichletEnforcementMethod::FullControl     
    )
  )
);
// clang-format on

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  //UnitTestLogger logger;  // create & initialize test logger, finalized when
  //                        // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}