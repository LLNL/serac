#include <functional>

#include "mfem.hpp"

#include "common/"

mfem::Vector u_c(double t)
{
  mfem::Vector u(3);
  u[0] = 0;
  u[1] = 0.25 * sin(2 * M_PI * t);
  u[2] = 0;
  return u;
}

mfem::Vector f_ext(double t)
{
  mfem::Vector force(3);
  force[0] = -10 * t;
  force[1] = 0;
  force[2] = 10 * t * t;
  return force;
}

class FirstOrderODE : public mfem::TimeDependentOperator {
 public:
  using explicit_signature = void(const double /*t*/, const mfem::Vector& /*u*/, mfem::Vector& /* du_dt */);
  using implicit_signature = void(const double /*t*/, const double /*dt*/, const mfem::Vector& /*u*/, mfem::Vector& /* du_dt */);
  std::function< explicit_signature > explicit_func;
  std::function< implicit_signature > implicit_func;

  FirstOrderODE(int n) : mfem::TimeDependentOperator(n, 0.0) {};

  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { explicit_func(t, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { implicit_func(t, dt, u, du_dt); }
};

double first_order_test(int num_steps)
{
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2; M(0, 1) = 2; M(0, 2) = 0;
  M(1, 0) = 1; M(1, 1) = 4; M(1, 2) = 1;
  M(2, 0) = 0; M(2, 1) = 2; M(2, 2) = 2;

  double            k = 100;
  mfem::DenseMatrix K(3, 3);
  K(0, 0) = k; K(0, 1) = -k; K(0, 2) = 0;
  K(1, 0) = -k; K(1, 1) = 2 * k; K(1, 2) = -k;
  K(2, 0) = 0; K(2, 1) = -k; K(2, 2) = k;

  double gamma = 0.5;

  double tmax = 1.0;
  double dt   = tmax / num_steps;

  mfem::DenseMatrix invH = M; 
  mfem::DenseMatrix T = M;
  T.AddMatrix(gamma * dt, K, 0, 0);

  for (int i = 0; i < 3; i++) {
    T(i, 1) = T(1, i) = (i == 1);
    invH(i, 1) = invH(1, i) = (i == 1);
  }

  invH.Invert();

  FirstOrderODE ode(3);
  ode.explicit_func = [](const double t, const mfem::Vector & u, mfem::Vector & du_dt) {
    mfem::Vector f = f_ext(t) - K * u - M * u_c(t);
  };


  double error = 0.0;

  return error;
}

int main()
{
  first_order_test(20);
}