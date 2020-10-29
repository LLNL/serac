#include <functional>
#include <iomanip>

#include "mfem.hpp"
#include "numerics/expr_template_ops.hpp"
#include "physics/operators/odes.hpp"
#include "physics/operators/stdfunction_operator.hpp"
#include "physics/utilities/equation_solver.hpp"

constexpr double epsilon = 1.0e-4;

template <typename T>
struct has_print {
  template <typename C>
  static float test(decltype(&C::Print));
  template <typename C>
  static double         test(...);
  static constexpr bool value = sizeof(test<T>(0)) == sizeof(float);
};

template <typename T, typename U = std::enable_if_t<has_print<T>::value> >
std::ostream& operator<<(std::ostream& out, const T& vec)
{
  vec.Print(out);
  return out;
}

double uc(double t) { return 0.25 * sin(2 * M_PI * t); }

mfem::Vector fext(double t)
{
  mfem::Vector force(3);
  force[0] = -10 * t;
  force[1] = 0;
  force[2] = 10 * t * t;
  return force;
}

#if 0
double first_order_test_mfem(int num_steps)
{
  // clang-format off
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2; M(0, 1) = 1; M(0, 2) = 0;
  M(1, 0) = 1; M(1, 1) = 4; M(1, 2) = 1;
  M(2, 0) = 0; M(2, 1) = 1; M(2, 2) = 2;

  double k = 100;
  mfem::DenseMatrix K(3, 3);
  K(0, 0) =  k; K(0, 1) =    -k; K(0, 2) =  0;
  K(1, 0) = -k; K(1, 1) = 2 * k; K(1, 2) = -k;
  K(2, 0) =  0; K(2, 1) =    -k; K(2, 2) =  k;
  // clang-format on

  double tmax = 1.0;
  double dt   = tmax / num_steps;

  mfem::DenseMatrix invH = M;
  mfem::DenseMatrix T(3, 3);

  for (int i = 0; i < 3; i++) {
    invH(i, 1) = invH(1, i) = (i == 1);
  }

  invH.Invert();

  double       t = 0.0;
  mfem::Vector u = uc(t);

  FirstOrderODE ode(u.Size(), [&, dt_prev = -1.0, invT = mfem::DenseMatrix(3, 3)](
                                  const double t, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) mutable {
    if (dt != dt_prev) {
      invT = M;
      invT.AddMatrix(dt, K, 0, 0);
      for (int i = 0; i < 3; i++) {
        invT(i, 1) = invT(1, i) = (i == 1);
      }
      invT.Invert();
      dt_prev = dt;
    }

    double       epsilon  = 1.0e-5;
    mfem::Vector uc_plus  = uc(t + epsilon);
    mfem::Vector uc_minus = uc(t - epsilon);

    mfem::Vector duc_dt = (uc_plus - uc_minus) / (2.0 * epsilon);

    mfem::Vector uf = u;
    uf[1]           = 0;

    mfem::Vector f = fext(t) - M * duc_dt - K * uc(t) - K * uf;
    f[1]           = 0;

    du_dt = invT * f;
  });

  auto time_integrator = mfem::RK4Solver();
  time_integrator.Init(ode);

  for (int i = 0; i < num_steps; i++) {
    time_integrator.Step(u, t, dt);
    u[1] = uc(t)[1];
  }

  mfem::Vector solution(3);
  solution[0] = -0.1443913076373983;
  solution[1] = 0.0;
  solution[2] = 0.04968869236260167;

  mfem::Vector error = u - solution;

  std::cout << error << std::endl;

  return error.Norml2();
}
#endif

double second_order_test_mfem(int num_steps, int method = 2)
{
  // clang-format off
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2; M(0, 1) = 1; M(0, 2) = 0;
  M(1, 0) = 1; M(1, 1) = 4; M(1, 2) = 1;
  M(2, 0) = 0; M(2, 1) = 1; M(2, 2) = 2;

  mfem::DenseMatrix C(3, 3);
  C(0, 0) =  1; C(0, 1) = -1; C(0, 2) =  0;
  C(1, 0) = -1; C(1, 1) =  2; C(1, 2) = -1;
  C(2, 0) =  0; C(2, 1) = -1; C(2, 2) =  1;

  double k = 100;
  mfem::DenseMatrix K(3, 3);
  K(0, 0) =  k; K(0, 1) =    -k; K(0, 2) =  0;
  K(1, 0) = -k; K(1, 1) = 2 * k; K(1, 2) = -k;
  K(2, 0) =  0; K(2, 1) =    -k; K(2, 2) =  k;
  // clang-format on

  double c0, c1;
  double t    = 0.0;
  double tmax = 1.0;

  mfem::DenseMatrix A(3);
  mfem::DenseMatrix invA(3);

  // Set the linear solver parameters
  serac::LinearSolverParameters lin_params;
  lin_params.prec        = serac::Preconditioner::BoomerAMG;
  lin_params.abs_tol     = 1.0e-16;
  lin_params.rel_tol     = 1.0e-12;
  lin_params.max_iter    = 500;
  lin_params.lin_solver  = serac::LinearSolver::GMRES;
  lin_params.print_level = -1;

  // Set the nonlinear solver parameters
  serac::NonlinearSolverParameters nonlin_params;
  nonlin_params.abs_tol     = 1.0e-16;
  nonlin_params.rel_tol     = 1.0e-12;
  nonlin_params.print_level = -1;
  nonlin_params.max_iter    = 5;

  mfem::Vector zero(3);
  zero = 0.0;

  mfem::Vector u(3);
  mfem::Vector du_dt(3);

  StdFunctionOperator op(3);

  StdFunctionOperator unconstrained(3);
  StdFunctionOperator constrained(3);

  op.residual = [&](const mfem::Vector& d2u_dt2, mfem::Vector& res) mutable {
    res    = M * d2u_dt2 + C * (du_dt + c1 * d2u_dt2) + K * (u + c0 * d2u_dt2) - fext(t);
    res[1] = 0.0;
  };

  op.jacobian = [&](const mfem::Vector& /*d2u_dt2*/) mutable -> mfem::Operator& {
    A = M;
    A.AddMatrix(c1, C, 0, 0);
    A.AddMatrix(c0, K, 0, 0);
    for (int i = 0; i < 3; i++) {
      A(i, 1) = A(1, i) = (i == 1);
    }
    return A;
  };

  serac::EquationSolver root_finder(MPI_COMM_WORLD, lin_params, nonlin_params);
  root_finder.nonlinearSolver().iterative_mode = true;
  root_finder.SetOperator(op);

  SecondOrderODE ode(3, [&](const double time, const double fac0, const double fac1, const mfem::Vector& displacement,
                            const mfem::Vector& velocity, mfem::Vector& acceleration) {
    t  = time;
    c0 = fac0;
    c1 = fac1;

    u     = displacement;
    du_dt = velocity;

    if (c0 == 0.0 && c1 == 0.0) {
      acceleration[1] = (uc(t + epsilon) - 2 * uc(t) + uc(t - epsilon)) / (epsilon * epsilon);
    } else {
      if (method == 1) {
        acceleration[1] = (uc(t + epsilon) - 2 * uc(t) + uc(t - epsilon)) / (epsilon * epsilon);
        du_dt[1]        = (uc(t + epsilon) - uc(t - epsilon)) / (2.0 * epsilon) - c1 * acceleration[1];
        u[1]            = uc(t) - c0 * acceleration[1];
      }

      if (method == 2) {
        acceleration[1] = (uc(t) - u[1]) / c0;
      }
    }

    // while (norm(r(acceleration) > tolerance) {
    //   acceleration -= J(acceleration)^{-1} * r(acceleration);
    // }
    root_finder.Mult(zero, acceleration);
  });

  auto time_integrator = mfem::NewmarkSolver();
  time_integrator.Init(ode);

  double       time = 0.0;
  double       dt   = tmax / num_steps;
  mfem::Vector displacement(3);
  displacement[0] = 0.0;
  displacement[1] = uc(time);
  displacement[2] = 0.0;

  mfem::Vector velocity(3);
  velocity[0] = 0.0;
  velocity[1] = (uc(time + epsilon) - uc(time - epsilon)) / (2.0 * epsilon);
  velocity[2] = 0.0;

  for (int i = 0; i < num_steps; i++) {
    time_integrator.Step(displacement, velocity, time, dt);

    if (method == 1) {
      displacement[1] = uc(time);
      velocity[1]     = (uc(time + epsilon) - uc(time + epsilon)) / (2.0 * epsilon);
    }
  }

  mfem::Vector solution(3);
  solution[0] = -1.0106975024794817575;
  solution[1] = 0.0;
  solution[2] = -0.82245512013213784019;

  mfem::Vector error = displacement - solution;

  error.Print(std::cout);

  return error.Norml2();
}

double second_order_test_as_first_order_mfem(int num_steps, int method = 2)
{
  // clang-format off
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2; M(0, 1) = 1; M(0, 2) = 0;
  M(1, 0) = 1; M(1, 1) = 4; M(1, 2) = 1;
  M(2, 0) = 0; M(2, 1) = 1; M(2, 2) = 2;

  mfem::DenseMatrix C(3, 3);
  C(0, 0) =  1; C(0, 1) = -1; C(0, 2) =  0;
  C(1, 0) = -1; C(1, 1) =  2; C(1, 2) = -1;
  C(2, 0) =  0; C(2, 1) = -1; C(2, 2) =  1;

  double k = 100;
  mfem::DenseMatrix K(3, 3);
  K(0, 0) =  k; K(0, 1) =    -k; K(0, 2) =  0;
  K(1, 0) = -k; K(1, 1) = 2 * k; K(1, 2) = -k;
  K(2, 0) =  0; K(2, 1) =    -k; K(2, 2) =  k;
  // clang-format on

  double c0, c1;
  double t    = 0.0;
  double tmax = 1.0;

  mfem::DenseMatrix A(3);
  mfem::DenseMatrix invA(3);

  // Set the linear solver parameters
  serac::LinearSolverParameters lin_params;
  lin_params.prec        = serac::Preconditioner::BoomerAMG;
  lin_params.abs_tol     = 1.0e-16;
  lin_params.rel_tol     = 1.0e-12;
  lin_params.max_iter    = 500;
  lin_params.lin_solver  = serac::LinearSolver::GMRES;
  lin_params.print_level = -1;

  // Set the nonlinear solver parameters
  serac::NonlinearSolverParameters nonlin_params;
  nonlin_params.abs_tol     = 1.0e-16;
  nonlin_params.rel_tol     = 1.0e-12;
  nonlin_params.print_level = -1;
  nonlin_params.max_iter    = 5;

  mfem::Vector zero(3);
  zero = 0.0;

  mfem::Vector u(3);
  mfem::Vector du_dt(3);

  StdFunctionOperator op(3);

  op.residual = [&](const mfem::Vector& d2u_dt2, mfem::Vector& res) mutable {
    res    = M * d2u_dt2 + C * (du_dt + c1 * d2u_dt2) + K * (u + c0 * d2u_dt2) - fext(t);
    res[1] = 0.0;
  };

  op.jacobian = [&](const mfem::Vector& /*d2u_dt2*/) mutable -> mfem::Operator& {
    A = M;
    A.AddMatrix(c1, C, 0, 0);
    A.AddMatrix(c0, K, 0, 0);
    for (int i = 0; i < 3; i++) {
      A(i, 1) = A(1, i) = (i == 1);
    }
    return A;
  };

  serac::EquationSolver root_finder(MPI_COMM_WORLD, lin_params, nonlin_params);
  root_finder.nonlinearSolver().iterative_mode = true;
  root_finder.SetOperator(op);

  SecondOrderODE ode2(3, [&](const double time, const double fac0, const double fac1, const mfem::Vector& displacement,
                             const mfem::Vector& velocity, mfem::Vector& acceleration) {
    t  = time;
    c0 = fac0;
    c1 = fac1;

    u     = displacement;
    du_dt = velocity;

    if (c0 == 0.0 && c1 == 0.0) {
      acceleration[1] = (uc(t + epsilon) - 2 * uc(t) + uc(t - epsilon)) / (epsilon * epsilon);
      du_dt[1]        = (uc(t + epsilon) - uc(t - epsilon)) / (2.0 * epsilon);
      u[1]            = uc(t);
    } else {
      if (method == 1) {
        acceleration[1] = (uc(t + epsilon) - 2 * uc(t) + uc(t - epsilon)) / (epsilon * epsilon);
        du_dt[1]        = (uc(t + epsilon) - uc(t - epsilon)) / (2.0 * epsilon) - c1 * acceleration[1];
        u[1]            = uc(t) - c0 * acceleration[1];
      }

      if (method == 2) {
        acceleration[1] = (uc(t) - u[1]) / c0;
      }
    }

    // while (norm(r(acceleration) > tolerance) {
    //   acceleration -= J(acceleration)^{-1} * r(acceleration);
    // }
    root_finder.Mult(zero, acceleration);
  });

  //auto          time_integrator = mfem::RK4Solver();
  auto          time_integrator = mfem::SDIRK34Solver();
  FirstOrderODE ode1(ode2);
  time_integrator.Init(ode1);

  double time = 0.0;
  double dt   = tmax / num_steps;

  mfem::Vector x(6);

  mfem::Vector displacement(x.GetData() + 3, 3);
  displacement[0] = 0.0;
  displacement[1] = uc(time);
  displacement[2] = 0.0;

  mfem::Vector velocity(x.GetData(), 3);
  velocity[0] = 0.0;
  velocity[1] = (uc(time + epsilon) - uc(time - epsilon)) / (2.0 * epsilon);
  velocity[2] = 0.0;

  for (int i = 0; i < num_steps; i++) {
    time_integrator.Step(x, time, dt);

    if (method == 1) {
      displacement[1] = uc(time);
      velocity[1]     = (uc(time + epsilon) - uc(time + epsilon)) / (2.0 * epsilon);
    }
  }

  mfem::Vector solution(3);
  solution[0] = -1.0106975024794817575;
  solution[1] = 0.0;
  solution[2] = -0.82245512013213784019;

  mfem::Vector error = displacement - solution;

  error.Print(std::cout);

  return error.Norml2();
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

#if 0
  std::cout << "first order ODE errors:\n";
  std::cout << "20 steps: " << first_order_ode_test(20) << std::endl;
  std::cout << "40 steps: " << first_order_ode_test(40) << std::endl;
  std::cout << "80 steps: " << first_order_ode_test(80) << std::endl;

  std::cout << "second order ODE errors:\n";
  std::cout << "20 steps: " << second_order_ode_test(20) << std::endl;
  std::cout << "40 steps: " << second_order_ode_test(40) << std::endl;
  std::cout << "80 steps: " << second_order_ode_test(80) << std::endl;

#endif
  //  std::cout << "100 steps: \n" << first_order_test_mfem(100) << std::endl;
  //  std::cout << "200 steps: \n" << first_order_test_mfem(200) << std::endl;
  //  std::cout << "400 steps: \n" << first_order_test_mfem(400) << std::endl;

  std::cout << "100 steps: \n" << second_order_test_mfem(100) << std::endl;
  std::cout << "200 steps: \n" << second_order_test_mfem(200) << std::endl;
  std::cout << "400 steps: \n" << second_order_test_mfem(400) << std::endl;

  std::cout << "100 steps: \n" << second_order_test_mfem(100, 1) << std::endl;
  std::cout << "200 steps: \n" << second_order_test_mfem(200, 1) << std::endl;
  std::cout << "400 steps: \n" << second_order_test_mfem(400, 1) << std::endl;

  std::cout << "100 steps: \n" << second_order_test_as_first_order_mfem(100, 1) << std::endl;
  std::cout << "200 steps: \n" << second_order_test_as_first_order_mfem(200, 1) << std::endl;
  std::cout << "400 steps: \n" << second_order_test_as_first_order_mfem(400, 1) << std::endl;

  std::cout << "100 steps: \n" << second_order_test_as_first_order_mfem(100, 2) << std::endl;
  std::cout << "200 steps: \n" << second_order_test_as_first_order_mfem(200, 2) << std::endl;
  std::cout << "400 steps: \n" << second_order_test_as_first_order_mfem(400, 2) << std::endl;

  MPI_Finalize();
}
