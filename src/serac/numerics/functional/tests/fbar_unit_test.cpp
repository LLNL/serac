// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include "serac/numerics/functional/tensor.hpp"

template < int dim >
using vec = serac::tensor< double, dim >;

template < int dim >
using mat = serac::tensor< double, dim, dim >;

////////////////////////////////////////////////////////////////////////////////

std::vector< double > random_positive_reals(uint32_t n) {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<double> distribution(0.1, 1.0);

  std::vector< double > output(n);
  for (uint32_t q = 0; q < n; q++) {
    output[q] = distribution(generator);
  }
  return output;
}

template < int dim >
std::vector< mat< dim > > random_matrices(uint32_t n) {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  std::vector< mat<dim> > output(n);
  for (uint32_t q = 0; q < n; q++) {
    mat<dim> A{};
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A(i,j) = distribution(generator);
      }
    }
    output[q] = A;
  }

  return output;
}

////////////////////////////////////////////////////////////////////////////////

template < typename T >
std::vector<T> operator-(const std::vector<T> & x, const std::vector<T> & y) {
    std::vector<T> sum(x.size());
    for (uint32_t i = 0; i < x.size(); i++) {
        sum[i] = x[i] - y[i];
    }
    return sum;
}

template < typename T >
std::vector<T> operator+(const std::vector<T> & x, const std::vector<T> & y) {
    std::vector<T> sum(x.size());
    for (uint32_t i = 0; i < x.size(); i++) {
        sum[i] = x[i] + y[i];
    }
    return sum;
}

template < typename T >
std::vector<T> operator*(double scale, const std::vector<T> & x) {
    std::vector<T> scaled(x.size());
    for (uint32_t i = 0; i < x.size(); i++) {
        scaled[i] = scale * x[i];
    }
    return scaled;
}

template < typename T >
std::vector<T> operator/(const std::vector<T> & x, double scale) {
    std::vector<T> scaled(x.size());
    for (uint32_t i = 0; i < x.size(); i++) {
        scaled[i] = x[i] / scale;
    }
    return scaled;
}

////////////////////////////////////////////////////////////////////////////////

template < int dim >
double dth_root(double x) {
    static_assert(dim == 2 || dim == 3);
    if constexpr (dim == 2) { return sqrt(x); }
    if constexpr (dim == 3) { return cbrt(x); }
}

template < int dim >
void fbar(std::vector< mat<dim> > & dubar_dx, // actual output
          double & Jbar,                      // output for the sake of testing
          std::vector< double > & J,          // output for the sake of testing
          std::vector< mat<dim> > & F,        // output for the sake of testing
          const std::vector< mat<dim> > & du_dx, 
          const std::vector< double > & w) {

    constexpr mat<dim> I = serac::DenseIdentity<dim>();

    uint32_t num_qpts = uint32_t(du_dx.size());

    double denom = 0.0;
    Jbar = 0.0;
    J.resize(num_qpts);
    F.resize(num_qpts);
    for (uint32_t q = 0; q < num_qpts; q++) {
        F[q] = I + du_dx[q];
        J[q] = det(F[q]);
        Jbar += w[q] * J[q];
        denom += w[q];
    }
    Jbar /= denom;

    dubar_dx.resize(num_qpts);
    for (uint32_t q = 0; q < num_qpts; q++) {
        dubar_dx[q] = dth_root<dim>(Jbar / J[q]) * F[q] - I;
    }

}

template < int dim >
void J_jvp(      std::vector< double > & dJ, 
           const std::vector< mat<dim> > & du_dx, 
           const std::vector< mat<dim> > & ddu_dx) {

    constexpr mat<dim> I = serac::DenseIdentity<dim>();

    uint32_t num_qpts = uint32_t(du_dx.size());

    dJ.resize(num_qpts);
    for (uint32_t q = 0; q < num_qpts; q++) {
        mat<dim> F = I + du_dx[q];
        double J = det(F);
        dJ[q] = inner(inv(transpose(F)), ddu_dx[q]) * J; 
    }

}

template < int dim >
void Jbar_jvp(                   double   & dJbar, 
              const std::vector< mat<dim> > & du_dx, 
              const std::vector< mat<dim> > & ddu_dx,
              const std::vector< double > & w) {

    constexpr mat<dim> I = serac::DenseIdentity<dim>();

    uint32_t num_qpts = uint32_t(du_dx.size());

    double denom = 0.0;
    dJbar = 0.0;
    for (uint32_t q = 0; q < num_qpts; q++) {
        mat<dim> F = I + du_dx[q];
        double J = det(F);
        dJbar += inner(inv(transpose(F)), ddu_dx[q]) * J * w[q]; 
        denom += w[q];
    }
    dJbar /= denom;

}

template < int dim >
void dubar_dx_jvp(std::vector< mat<dim> > & dubar_dx, // actual output
                  const std::vector< mat<dim> > & du_dx, 
                  const std::vector< mat<dim> > & ddu_dx,
                  const std::vector< double > & w) {

    constexpr mat<dim> I = serac::DenseIdentity<dim>();

    uint32_t num_qpts = uint32_t(du_dx.size());

    double denom = 0.0;
    dJbar = 0.0;
    for (uint32_t q = 0; q < num_qpts; q++) {
        mat<dim> F = I + du_dx[q];
        double J = det(F);
        dJbar += inner(inv(transpose(F)), ddu_dx[q]) * J * w[q]; 
        denom += w[q];
    }
    dJbar /= denom;

}



const double epsilon = 1.0e-5;
const int num_quadrature_points = 15;

template < int dim >
void directional_derivatives_test() {

    std::vector< mat<dim> > dubar_dx[2];
    double Jbar[2];
    std::vector< double > J[2];
    std::vector< mat<dim> > F[2];

    std::vector< double > w = random_positive_reals(num_quadrature_points);
    std::vector< mat<dim> > du_dx = random_matrices<dim>(num_quadrature_points);
    std::vector< mat<dim> > ddu_dx = random_matrices<dim>(num_quadrature_points);

    fbar(dubar_dx[1], Jbar[1], J[1], F[1], du_dx + epsilon * ddu_dx, w);
    fbar(dubar_dx[0], Jbar[0], J[0], F[0], du_dx - epsilon * ddu_dx, w);

    {
        std::vector< double > dJ0 = (J[1] - J[0]) / (2 * epsilon);
        std::vector< double > dJ1;
        J_jvp(dJ1, du_dx, ddu_dx);

        for (uint32_t i = 0; i < num_quadrature_points; i++) {
            EXPECT_NEAR(dJ0[i], dJ1[i], 1.0e-9);
        }
    }

    {
        double dJbar0 = (Jbar[1] - Jbar[0]) / (2 * epsilon);
        double dJbar1;
        Jbar_jvp(dJbar1, du_dx, ddu_dx, w);

        EXPECT_NEAR(dJbar0, dJbar1, 1.0e-9);
    }

}

TEST(DirectionalDerivative, Tests2D) { directional_derivatives_test<2>(); }
TEST(DirectionalDerivative, Tests3D) { directional_derivatives_test<3>(); }
