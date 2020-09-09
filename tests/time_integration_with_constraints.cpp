#include <functional>
#include <iomanip>

#include "common/expr_template_ops.hpp"
#include "mfem.hpp"

template <typename T>
struct has_print {
  template <typename C> static float test(decltype(&C::Print)) ;
  template <typename C> static double test(...);    
  enum { value = sizeof(test<T>(0)) == sizeof(float) };
};

template < typename T, typename U = std::enable_if_t< has_print<T>::value > >
std::ostream& operator<<(std::ostream& out, const T & vec)
{
  vec.Print(out);
  return out;
}

mfem::Vector uc(double t)
{
  mfem::Vector u(3);
  u[0] = 0;
  u[1] = 0.25 * sin(2 * M_PI * t);
  u[2] = 0;
  return u;
}

mfem::Vector fext(double t)
{
  mfem::Vector force(3);
  force[0] = -10 * t;
  force[1] = 0;
  force[2] = 10 * t * t;
  return force;
}

class FirstOrderODE : public mfem::TimeDependentOperator {
public:
  using explicit_signature = void(const double, const mfem::Vector&, mfem::Vector&);
  using implicit_signature = void(const double, const double, const mfem::Vector&, mfem::Vector&);
  std::function<explicit_signature> explicit_func;
  std::function<implicit_signature> implicit_func;

  FirstOrderODE(int n) : mfem::TimeDependentOperator(n, 0.0){};
  void Mult(const mfem::Vector& u, mfem::Vector& du_dt) const { explicit_func(t, u, du_dt); }
  void ImplicitSolve(const double dt, const mfem::Vector& u, mfem::Vector& du_dt) { implicit_func(t, dt, u, du_dt); }
};

#if 0
double first_order_test(int num_steps)
{
  // clang-format off
  mfem::DenseMatrix M(3, 3);
  M(0, 0) = 2; M(0, 1) = 2; M(0, 2) = 0;
  M(1, 0) = 1; M(1, 1) = 4; M(1, 2) = 1;
  M(2, 0) = 0; M(2, 1) = 2; M(2, 2) = 2;

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

  FirstOrderODE ode(3);
  ode.explicit_func = [&](const double t, const mfem::Vector& u, mfem::Vector& du_dt) {
    du_dt = invH * (fext(t) - K * u - M * uc(t));
  };
  ode.implicit_func = [&, dt_prev = -1.0, invT = mfem::DenseMatrix(3, 3)](
                          const double t, const double dt, const mfem::Vector& u, mfem::Vector& du_dt) mutable {
    if (dt != dt_prev) {
      invT = M;
      invT.AddMatrix(dt, K, 0, 0);
      for (int i = 0; i < 3; i++) {
        invT(i, 1) = invT(1, i) = (i == 1);
      }
      invT.Invert();
    }

    double eps = dt / 1000.0;

    mfem::Vector uf     = u;
    mfem::Vector duf_dt = du_dt;
    duf_dt[1] = uf[1] = 0;

    mfem::Vector duc_dt = (uc(t + eps) - uc(t - eps)) * (1.0 / (2.0 * eps));
    mfem::Vector f      = fext(t) - M * duc_dt - K * uc(t) - K * (uf + dt * duf_dt);
    f[1]                = 0;

    std::cout << t << " " << dt << std::endl;

    du_dt   = invT * f;
    dt_prev = dt;
  };

  auto backward_euler = mfem::BackwardEulerSolver();
  backward_euler.Init(ode);

  mfem::Vector u = uc(0);
  double       t = 0.0;
  while (t < tmax) {
    double dt_actual = std::min(dt, tmax - t);
    backward_euler.Step(u, t, dt_actual);
    u[1] = uc(t)[1];
  }

  mfem::Vector solution(3);
  solution[0] = -0.1443913076373983;
  solution[1] = 0.0;
  solution[2] = 0.04968869236260167;

  mfem::Vector error = u - solution;

  return error.Norml2();
}
#endif

double first_order_ode_test(int num_steps)
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

  double gamma = 0.5;
  double tmax  = 1.0;
  double dt    = tmax / num_steps;

  mfem::DenseMatrix invH = M;
  mfem::DenseMatrix invT = M;
  invT.AddMatrix(dt * gamma, K, 0, 0);

  for (int i = 0; i < 3; i++) {
    invH(i, 1) = invH(1, i) = (i == 1);
    invT(i, 1) = invT(1, i) = (i == 1);
  }

  invH.Invert();
  invT.Invert();

  auto duc_dt = [](double t) {
    static constexpr double eps = 1.0e-4;
    return mfem::Vector((uc(t + eps) - uc(t - eps)) * (1.0 / (2.0 * eps)));
  };

  double       t      = 0.0;
  mfem::Vector u      = uc(t);
  mfem::Vector du_dt  = invH * (fext(t) - K * u - M * duc_dt(t));
  du_dt[1]            = duc_dt(t)[1];
  mfem::Vector uf     = u;
  mfem::Vector duf_dt = du_dt;
  uf[1] = duf_dt[1] = 0.0;

  for (int i = 0; i < num_steps; i++) {
    t += dt;

    mfem::Vector f = fext(t) - M * duc_dt(t) - K * uc(t) - K * (uf + (1.0 - gamma) * duf_dt * dt);
    f[1]           = 0;

    uf += (1.0 - gamma) * duf_dt * dt;
    duf_dt = invT * f;
    uf += gamma * duf_dt * dt;

    u = uf + uc(t);
  }

  mfem::Vector solution(3);
  solution[0] = -0.1443913076373983;
  solution[1] = 0.0;
  solution[2] = 0.04968869236260167;

  mfem::Vector error = u - solution;

  return error.Norml2();
}

double second_order_ode_test(int num_steps)
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

  double beta  = 1.0 / 6.0;
  double gamma = 1.0 / 2.0;

  double tmax = 1.0;
  double dt   = tmax / num_steps;

  mfem::DenseMatrix invH = M;
  mfem::DenseMatrix invT = M;
  invT.AddMatrix(dt * gamma, C, 0, 0);
  invT.AddMatrix(dt * dt * beta, K, 0, 0);

  for (int i = 0; i < 3; i++) {
    invH(i, 1) = invH(1, i) = (i == 1);
    invT(i, 1) = invT(1, i) = (i == 1);
  }

  invH.Invert();
  invT.Invert();

  std::cout << invT << std::endl;

  auto duc_dt = [](double t) {
    static constexpr double eps = 1.0e-4;
    return mfem::Vector((uc(t + eps) - uc(t - eps)) * (1.0 / (2.0 * eps)));
  };

  auto d2uc_dt2 = [](double t) {
    static constexpr double eps = 1.0e-4;
    return mfem::Vector((uc(t - eps) - 2 * uc(t) + uc(t + eps)) * (1.0 / (eps * eps)));
  };

  double       t       = 0.0;
  mfem::Vector u       = uc(t);
  mfem::Vector du_dt   = duc_dt(t);
  mfem::Vector d2u_dt2 = invH * (fext(t) - K * u - C * duc_dt(t) - M * d2uc_dt2(t));
  du_dt[1]             = duc_dt(t)[1];
  d2u_dt2[1]           = d2uc_dt2(t)[1];
  mfem::Vector uf      = u;
  mfem::Vector duf_dt  = du_dt;
  mfem::Vector d2uf_dt2  = d2u_dt2;
  uf[1] = duf_dt[1] = d2uf_dt2[1] = 0.0;

  for (int i = 0; i < num_steps; i++) {
    t += dt;

    mfem::Vector fc = M * d2uc_dt2(t) + C * duc_dt(t) + K * uc(t);
    mfem::Vector fp = C * (duf_dt + d2uf_dt2 * (1.0 - gamma) * dt) + 
                      K * (uf + duf_dt * dt + d2uf_dt2 * (0.5 - beta) * dt * dt);
    mfem::Vector f = fext(t) - fc + fp;
    f[1]           = 0;

    if (i < 2) {
      std::cout << t << std::endl;
      std::cout << fc << fp << fext(t) << f << std::endl;
    }

    uf += duf_dt * dt + (0.5 - beta) * d2uf_dt2 * dt * dt;
    duf_dt += (1.0 - gamma) * d2uf_dt2 * dt;

    d2uf_dt2 = invT * f;

    uf += beta * d2uf_dt2 * dt * dt;
    duf_dt += gamma * d2uf_dt2 * dt;

    u = uf + uc(t);
    du_dt = duf_dt + duc_dt(t);
  }

  mfem::Vector solution(3);
  solution[0] = -1.0124;
  solution[1] = 0.0;
  solution[2] = -0.823848;

  mfem::Vector error = u - solution;

  std::cout <<  error << std::endl;

  return error.Norml2();
}

int main()
{
//  int num_steps = 10;
//  for (int i = 0; i < 5; i++) {
//    std::cout << "error with " << std::setw(3) << num_steps << " steps: " << first_order_ode_test(num_steps)
//              << std::endl;
//    num_steps <<= 1;
//  }
//  int num_steps = 10;
//  for (int i = 0; i < 5; i++) {
//    std::cout << "error with " << std::setw(3) << num_steps << " steps: " << second_order_ode_test(num_steps) << std::endl;
//    num_steps <<= 1;
//  }

  second_order_ode_test(20);

  mfem::Vector a(3), b(3), c(3);

  a[0] = a[1] = a[2] = 1;
  b[0] = b[1] = b[2] = 2;
  c[0] = c[1] = c[2] = 4;

  auto expr1 = a + c - b;
  // VectorSubtraction<mfem::Vector, BinaryVectorExpr<mfem::Vector, mfem::Vector, false, false, std::plus<double>>, false, true>

  auto expr2 = c - b + a;
  // VectorAddition<mfem::Vector, BinaryVectorExpr<mfem::Vector, mfem::Vector, false, false, std::minus<double>>, false, true>

  std::cout << mfem::Vector(expr1) << std::endl;
  std::cout << mfem::Vector(expr2) << std::endl;

}