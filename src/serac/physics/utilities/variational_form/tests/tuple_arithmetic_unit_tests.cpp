#include "../tuple_arithmetic.hpp"

#include <random>
#include <iostream>

auto random_real = [](auto...) {
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
};

static constexpr auto   I   = Identity<3>();
static constexpr double rho = 3.0;
static constexpr double mu  = 2.0;

static constexpr double epsilon = 1.0e-6;

constexpr auto sigma = [](auto p, auto v, auto L) { return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I; };

constexpr auto incompressibility = [](auto /*p*/, auto /*v*/, auto L) { return tr(L); };

constexpr auto tuple_func = [](auto p, auto v, auto L) {
  return std::tuple{rho * outer(v, v) + 2.0 * mu * sym(L) - p * I, v + dot(v, L)};
};

constexpr auto dsigma_dp = [](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

constexpr auto dsigma_dv = [](auto /*p*/, auto v, auto /*L*/) {
  return make_tensor<3, 3, 3>([&](int i, int j, int k) { return rho * ((i == k) * v[j] + (j == k) * v[i]); });
};

constexpr auto dsigma_dL = [](auto /*p*/, auto /*v*/, auto /*L*/) {
  return make_tensor<3, 3, 3, 3>(
      [&](int i, int j, int k, int l) { return mu * ((i == k) * (j == l) + (i == l) * (j == k)); });
};

void chain_rule_tests()
{
  static constexpr double p = 3.14;
  static constexpr tensor v = {{1.0, 2.0, 3.0}};
  static constexpr tensor L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  static constexpr double dp = 1.23;
  static constexpr tensor dv = {{2.0, 1.0, 4.0}};
  static constexpr tensor dL = {{{3.0, 1.0, 2.0}, {2.0, 7.0, 3.0}, {4.0, 4.0, 3.0}}};

  std::cout << "chain rule tests" << std::endl;

  {
    auto df_dx = make_tensor<>(random_real);
    auto dx    = make_tensor<>(random_real);
    auto df    = chain_rule(df_dx, dx);
    std::cout << df << std::endl;
  }

  {
    auto              df_dx = make_tensor<3, 3>(random_real);
    auto              dx    = make_tensor<3>(random_real);
    tensor<double, 3> df    = chain_rule(df_dx, dx);
    std::cout << df << std::endl;
  }

  {
    auto   df_dx = make_tensor<3, 3>(random_real);
    auto   dx    = make_tensor<3, 3>(random_real);
    double df    = chain_rule(df_dx, dx);
    std::cout << df << std::endl;
  }

  {
    auto                 df_dx = make_tensor<3, 3, 3>(random_real);
    auto                 dx    = make_tensor<3>(random_real);
    tensor<double, 3, 3> df    = chain_rule(df_dx, dx);
    std::cout << df << std::endl;
  }

  {
    auto                 df_dx = make_tensor<3, 3, 3, 3>(random_real);
    auto                 dx    = make_tensor<3, 3>(random_real);
    tensor<double, 3, 3> df    = chain_rule(df_dx, dx);
    std::cout << df << std::endl;
  }

  {
    auto                                                 df1_dx = make_tensor<3, 3, 3, 3>(random_real);
    auto                                                 df2_dx = make_tensor<3, 3, 3>(random_real);
    auto                                                 dx     = make_tensor<3, 3>(random_real);
    std::tuple<tensor<double, 3, 3>, tensor<double, 3> > df     = chain_rule(std::tuple{df1_dx, df2_dx}, dx);

    std::cout << std::get<0>(df) << std::endl;
    std::cout << std::get<1>(df) << std::endl;
  }

  {
    auto tuple_of_values = std::apply(tuple_func, make_dual(p, v, L));

    auto grad = get_gradient(tuple_of_values);

    auto df0_fd = (sigma(p + epsilon * dp, v + epsilon * dv, L + epsilon * dL) -
                   sigma(p - epsilon * dp, v - epsilon * dv, L - epsilon * dL)) /
                  (2 * epsilon);

    auto df0 = (std::get<0>(std::get<0>(grad)) * dp) + dot(std::get<1>(std::get<0>(grad)), dv) +
               ddot(std::get<2>(std::get<0>(grad)), dL);

    auto df1 = (std::get<0>(std::get<1>(grad)) * dp) + dot(std::get<1>(std::get<1>(grad)), dv) +
               ddot(std::get<2>(std::get<1>(grad)), dL);

    auto df_ad = chain_rule(grad, std::tuple{dp, dv, dL});

    std::cout << "comparison with finite difference: ";
    std::cout << df0 - df0_fd << std::endl;

    std::cout << "comparison with chain_rule: ";
    std::cout << std::get<0>(df_ad) - df0 << std::endl;
    std::cout << std::get<1>(df_ad) - df1 << std::endl;
  }

  {
    std::tuple<std::tuple<double, zero>, std::tuple<zero, tensor<double, 2, 2> > > df_dx{};

    std::tuple<double, tensor<double, 2> > dx;

    auto df = chain_rule(df_dx, dx);

    std::cout << std::get<0>(df) << std::endl;
    std::cout << std::get<1>(df) << std::endl;
  }
}

int main()
{
  static constexpr double p = 3.14;
  static constexpr tensor v = {{1.0, 2.0, 3.0}};
  static constexpr tensor L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  constexpr auto exact_dsigma_dp = dsigma_dp(p, v, L);
  constexpr auto exact_dsigma_dv = dsigma_dv(p, v, L);
  constexpr auto exact_dsigma_dL = dsigma_dL(p, v, L);

  auto stress = std::apply(sigma, make_dual(p, v, L));

  {
    auto grad = get_gradient(stress);
    std::cout << std::get<0>(grad) - exact_dsigma_dp << std::endl;
    std::cout << std::get<1>(grad) - exact_dsigma_dv << std::endl;
    std::cout << std::get<2>(grad) - exact_dsigma_dL << std::endl;
  }

  auto dilatation = std::apply(incompressibility, make_dual(p, v, L));

  {
    std::cout << "dilatation derivatives:" << std::endl;
    auto grad = get_gradient(dilatation);
    std::cout << std::get<0>(grad) << std::endl;
    std::cout << std::get<1>(grad) << std::endl;
    std::cout << std::get<2>(grad) << std::endl;
  }

  auto tuple_of_values = std::apply(tuple_func, make_dual(p, v, L));

  {
    // std::tuple<
    //  tensor<double, 3, 3>,
    //  tensor<double, 3>
    //>
    auto value = get_value(tuple_of_values);

    std::cout << "matrix:" << std::endl;
    std::cout << std::get<0>(value) << std::endl;

    std::cout << "matrix errors:" << std::endl;
    std::cout << std::get<0>(value) - sigma(p, v, L) << std::endl;

    std::cout << "vector:" << std::endl;
    std::cout << std::get<1>(value) << std::endl;

    // std::tuple<
    //  std::tuple<tensor<double, 3, 3>, tensor<double, 3, 3, 3>, tensor<double, 3, 3, 3, 3> >,
    //  std::tuple<zero, zero, tensor<double, 3, 3> >
    //>
    auto grad = get_gradient(tuple_of_values);

    std::cout << "matrix derivatives:" << std::endl;
    std::cout << std::get<0>(std::get<0>(grad)) << std::endl;
    std::cout << std::get<1>(std::get<0>(grad)) << std::endl;
    std::cout << std::get<2>(std::get<0>(grad)) << std::endl;

    std::cout << "matrix derivative errors:" << std::endl;
    std::cout << std::get<0>(std::get<0>(grad)) - exact_dsigma_dp << std::endl;
    std::cout << std::get<1>(std::get<0>(grad)) - exact_dsigma_dv << std::endl;
    std::cout << std::get<2>(std::get<0>(grad)) - exact_dsigma_dL << std::endl;

    std::cout << "vector derivatives:" << std::endl;
    std::cout << std::get<0>(std::get<1>(grad)) << std::endl;
    std::cout << std::get<1>(std::get<1>(grad)) << std::endl;
    std::cout << std::get<2>(std::get<1>(grad)) << std::endl;
  }

  chain_rule_tests();
}