// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/utilities/variational_form/tensor.hpp"

using namespace serac;

template < int n >
void custom_assert(bool condition, const char (&message)[n]) {
  if (condition == false) { printf("error: %s", message); }
}

__global__ void basic_tensor_tests() {

  constexpr auto abs = [](auto x) { return (x < 0) ? -x : x; };

  constexpr tensor<double, 3> u = {1, 2, 3};
  constexpr tensor<double, 4> v = {4, 5, 6, 7};

  constexpr tensor<double, 3, 3> A = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  constexpr double sqnormA = 111.0;
  static_assert(abs(sqnorm(A) - sqnormA) < 1.0e-16);

  constexpr tensor<double, 3, 3> symA = {{{0, 1.5, 3}, {1.5, 3, 4.5}, {3, 4.5, 6}}};
  static_assert(abs(sqnorm(sym(A) - symA)) < 1.0e-16);

  constexpr tensor<double, 3, 3> devA = {{{-3, 2, 4}, {1, 0, 5}, {2, 4, 3}}};
  static_assert(abs(sqnorm(dev(A) - devA)) < 1.0e-16);

  constexpr tensor<double, 3, 3> invAp1 = {{{-4, -1, 3}, {-1.5, 0.5, 0.5}, {2, 0, -1}}};
  static_assert(abs(sqnorm(inv(A + Identity<3>()) - invAp1)) < 1.0e-16);

  constexpr tensor<double, 3> Au = {16, 22, 28};
  static_assert(abs(sqnorm(dot(A, u) - Au)) < 1.0e-16);

  constexpr tensor<double, 3> uA = {8, 20, 32};
  static_assert(abs(sqnorm(dot(u, A) - uA)) < 1.0e-16);

  constexpr double uAu = 144;
  static_assert(abs(dot(u, A, u) - uAu) < 1.0e-16);

  constexpr tensor<double, 3, 4> B = make_tensor<3, 4>([](auto i, auto j) { return 3.0 * i - j; });

  constexpr double uBv = 300;
  static_assert(abs(dot(u, B, v) - uBv) < 1.0e-16);

}

#if 0
void elasticity_tests() {
  static constexpr auto abs = [](auto x) { return (x < 0) ? -x : x; };

  static constexpr double lambda = 5.0;
  static constexpr double mu     = 3.0;
  static constexpr tensor C      = make_tensor<3, 3, 3, 3>([&](int i, int j, int k, int l) {
    return lambda * (i == j) * (k == l) + mu * ((i == k) * (j == l) + (i == l) * (j == k));
  });

  static constexpr auto I     = Identity<3>();
  constexpr auto        sigma = [=](auto epsilon) { return lambda * tr(epsilon) * I + 2.0 * mu * epsilon; };

  constexpr tensor grad_u = make_tensor<3, 3>([](int i, int j) { return i + 2.0 * j; });

  static_assert(abs(sqnorm(ddot(C, sym(grad_u)) - sigma(sym(grad_u)))) < 1.0e-16);

  constexpr auto epsilon = sym(make_dual(grad_u));

  [[maybe_unused]] static constexpr tensor stress = sigma(epsilon);

  for_constexpr<3, 3>([&](auto i, auto j) { static_assert(abs(sqnorm(C[i][j] - stress[i][j].gradient)) < 1.0e-16); });
}

void navier_stokes_tests()
{
  [[maybe_unused]] static constexpr auto abs = [](auto x) { return (x < 0) ? -x : x; };

  static constexpr auto   I     = Identity<3>();
  static constexpr double rho   = 3.0;
  static constexpr double mu    = 2.0;
  constexpr auto          sigma = [](auto p, auto v, auto L) { return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I; };

  constexpr auto dsigma_dp = [](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

  constexpr auto dsigma_dv = [&](auto /*p*/, auto v, auto /*L*/) {
    return make_tensor<3, 3, 3>([&](int i, int j, int k) { return rho * ((i == k) * v[j] + (j == k) * v[i]); });
  };

  constexpr auto dsigma_dL = [&](auto /*p*/, auto /*v*/, auto /*L*/) {
    return make_tensor<3, 3, 3, 3>(
        [&](int i, int j, int k, int l) { return mu * ((i == k) * (j == l) + (i == l) * (j == k)); });
  };

  constexpr double p = 3.14;
  constexpr tensor v = {{1.0, 2.0, 3.0}};
  constexpr tensor L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  {
    [[maybe_unused]] static constexpr auto exact = dsigma_dp(p, v, L);
    [[maybe_unused]] static constexpr auto ad    = sigma(make_dual(p), v, L);

    for_constexpr<3, 3>([&](auto i, auto j) { static_assert(abs(exact[i][j] - ad[i][j].gradient) < 1.0e-16); });
  }

  {
    [[maybe_unused]] static constexpr auto exact = dsigma_dv(p, v, L);
    [[maybe_unused]] static constexpr auto ad    = sigma(p, make_dual(v), L);

    for_constexpr<3, 3>([&](auto i, auto j) { static_assert(abs(sqnorm(exact[i][j] - ad[i][j].gradient)) < 1.0e-16); });
  }

  {
    [[maybe_unused]] static constexpr auto exact = dsigma_dL(p, v, L);
    [[maybe_unused]] static constexpr auto ad    = sigma(p, v, make_dual(L));

    for_constexpr<3, 3>([&](auto i, auto j) { static_assert(abs(sqnorm(exact[i][j] - ad[i][j].gradient)) < 1.0e-16); });
  }
}
#endif

int main()
{
  basic_tensor_tests<<<1,1>>>();
  //elasticity_tests();
  //navier_stokes_tests();
}