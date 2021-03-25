#include <random>
#include <chrono>

#include "serac/physics/utilities/variational_form/tensor.hpp"

class timer {
  typedef std::chrono::high_resolution_clock::time_point time_point;
  typedef std::chrono::duration<double> duration_type;

 public:
  void start() { then = std::chrono::high_resolution_clock::now(); }

  void stop() { now = std::chrono::high_resolution_clock::now(); }

  double elapsed() {
    return std::chrono::duration_cast<duration_type>(now - then).count();
  }

 private:
  time_point then, now;
};

static constexpr auto I = Identity<3>();

struct J2 {
  double E       = 100;    // Young's modulus
  double nu      = 0.25;   // Poisson's ratio
  double Hi      = 1.0;    // isotropic hardening constant
  double Hk      = 2.3;    // kinematic hardening constant
  double sigma_y = 300.0;  // yield stress

  struct State {
    tensor<double, 3, 3> beta;           // back-stress tensor
    tensor<double, 3, 3> el_strain;      // elastic strain
    double               pl_strain;      // plastic strain
    double               pl_strain_inc;  // incremental plastic strain
    double               q;              // (trial) J2 stress
  };

  auto calculate_stress(const tensor<double, 3, 3> grad_u, State& state) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    state.el_strain          = sym(grad_u);
    double               p   = K * tr(state.el_strain);
    tensor<double, 3, 3> s   = 2.0 * G * dev(state.el_strain);
    tensor<double, 3, 3> eta = s - state.beta;
    state.q                  = sqrt(3.0 / 2.0) * norm(eta);
    double phi               = state.q - (sigma_y + Hi * state.pl_strain);

    // see (7.207) on pg. 261
    state.pl_strain_inc = fmax(0.0, phi / (3 * G + Hk + Hi));

    // (ii) admissibility
    if (state.pl_strain_inc > 0.0) {

      // (iii) return mapping
      s = s - sqrt(6.0) * G * state.pl_strain_inc * normalize(eta);

      state.pl_strain = state.pl_strain + state.pl_strain_inc;
      state.el_strain = (s / (2.0 * G)) + ((p / K) * I);

      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * state.pl_strain_inc * normalize(eta);
    } 

    return s + p * I;
  }

  template <typename T>
  auto calculate_stress_AD(const T grad_u, State& state) const
  {
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);

    //
    // see pg. 260, box 7.5,
    // in "Computational Methods for Plasticity"
    //

    // (i) elastic predictor
    auto el_strain = sym(grad_u);
    auto p         = K * tr(el_strain);
    auto s         = 2.0 * G * dev(el_strain);
    auto eta       = s - state.beta;
    auto q         = sqrt(3.0 / 2.0) * norm(eta);
    auto phi       = q - (sigma_y + Hi * state.pl_strain);

    // (ii) admissibility
    if (phi > 0.0) {

      // see (7.207) on pg. 261
      auto plastic_strain_inc = phi / (3 * G + Hk + Hi);

      // (iii) return mapping
      s = s - sqrt(6.0) * G * plastic_strain_inc * normalize(eta);

      state.pl_strain = state.pl_strain + get_value(plastic_strain_inc);

      state.beta = state.beta + sqrt(2.0 / 3.0) * Hk * get_value(plastic_strain_inc) * normalize(get_value(eta));
    }

    return s + p * I;
  }

  auto calculate_gradient(const State& state) const
  {
    double K = E / (3.0 * (1.0 - 2.0 * nu));
    double G = 0.5 * E / (1.0 + nu);

    double A1 = 2.0 * G;
    double A2 = 0.0;

    tensor<double, 3, 3> N{};

    if (state.pl_strain_inc > 0.0) {
      tensor<double, 3, 3> s = 2.0 * G * dev(state.el_strain);
      N                      = normalize(s - state.beta);

      A1 -= 6 * G * G * state.pl_strain_inc / state.q;
      A2 = 6 * G * G * ((state.pl_strain_inc / state.q) - (1.0 / (3.0 * G + Hi + Hk)));
    }

    tensor<double, 3, 3, 3, 3> C{};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            double I4    = (i == j) * (k == l);
            double I4sym = 0.5 * ((i == k) * (j == l) + (i == l) * (j == k));
            double I4dev = I4sym - (i == j) * (k == l) / 3.0;

            C(i, j, k, l) = K * I4 + A1 * I4dev + A2 * N(i, j) * N(k, l);
          }
        }
      }
    }
    return C;
  }
};

// impose some arbitrary time-dependent deformation
auto displacement_gradient(double t)
{
  return 10 * tensor{{
                  {sin(t), 0.0, 0.0},
                  {0.0, t, exp(t) - 1},
                  {0.0, 0.0, t * t},
              }};
}

int main() {

  timer stopwatch;
  double J2_evaluation_time = 0.0;
  double J2_gradient_time = 0.0;
  double J2_AD_time = 0.0;

  double t  = 0.0;
  double dt = 0.0001;

  J2 material{
      100,   // Young's modulus
      0.25,  // Poisson's ratio
      1.0,   // isotropic hardening constant
      2.3,   // kinematic hardening constant
      300.0  // yield stress
  };

  J2::State state{};

  while (t < 1.0) {

    auto grad_u = displacement_gradient(t);

    auto backup = state;

    stopwatch.start();
    tensor<double,3,3> stress = material.calculate_stress(grad_u, state);
    stopwatch.stop();
    J2_evaluation_time += stopwatch.elapsed();

    stopwatch.start();
    tensor<double,3,3,3,3> C = material.calculate_gradient(state);
    stopwatch.stop();
    J2_gradient_time += stopwatch.elapsed();

    stopwatch.start();
    auto stress_and_C = material.calculate_stress_AD(make_dual(grad_u), backup);
    stopwatch.stop();
    J2_AD_time += stopwatch.elapsed();

    if (norm(stress - get_value(stress_and_C)) > 1.0e-12) exit(1);
    if (norm(C - get_gradient(stress_and_C)) > 1.0e-12) exit(1);
    if (norm(state.beta - backup.beta) > 1.0e-12) exit(1);
    if (fabs(state.pl_strain - backup.pl_strain) > 1.0e-12) exit(1);

    t += dt;
  }

  std::cout << "total J2 evaluation time (no AD): " << J2_evaluation_time << std::endl;
  std::cout << "total J2 gradient time (no AD): " << J2_gradient_time << std::endl;
  std::cout << "total J2 evaluation+gradient time (AD): " << J2_AD_time << std::endl;
  std::cout << "(AD time) / (manual gradient time): " << J2_AD_time / (J2_evaluation_time + J2_gradient_time) << std::endl;

}

// total J2 evaluation time (no AD):       0.0196884
// total J2 gradient time (no AD):         0.0439456
// total J2 evaluation+gradient time (AD): 0.256943
// (AD time) / (manual gradient time):     4.03782