#include "tuple_arithmetic.hpp"

#include <iostream>

namespace impl {

  template < typename T1, typename T2 >
  struct outer_prod;

  template < int ... m, int ... n >
  struct outer_prod< tensor< double, m ... >, tensor< double, n ... > >{
    using type = tensor< double, m ..., n ... >;
  };

  template < int ... n >
  struct outer_prod< double, tensor< double, n ... > >{
    using type = tensor< double, n ... >;
  };

  template < int ... n >
  struct outer_prod< tensor< double, n ... >, double >{
    using type = tensor< double, n ... >;
  };

  template <>
  struct outer_prod< double, double >{
    using type = tensor< double >;
  };

  template < typename T >
  struct outer_prod< zero, T >{
    using type = zero;
  };

  template < typename T >
  struct outer_prod< T, zero >{
    using type = zero;
  };

}

template < typename T1, typename T2 >
using outer_product_t = typename impl::outer_prod<T1, T2>::type;

template < typename ... T >
auto get_grad(dual< std::tuple < T ... > > arg) {
  return std::apply([](auto ... each_value){
    return std::tuple{each_value ...};
  }, arg.gradient);
}


template < typename T, int ... n >
void get_grad(tensor< dual< double >, n ... > arg) {
  tensor< double, n ... > g{};
  for_constexpr< n ... >([&](auto ... i){
    g[{i...}] = arg[{i...}].gradient;
  });
  return g;
}

template < typename ... T, int ... n >
auto get_grad(tensor< dual< std::tuple < T ... > >, n ... > arg) {
  std::tuple < outer_product_t< tensor< double, n... >, T > ... > g{};
  for_constexpr< n ... >([&](auto ... i){
    for_constexpr< sizeof ... (T) >([&](auto j){
      std::get<j>(g)(i...) = std::get<j>(arg(i...).gradient);
    });
  });
  return g;
}

template < typename T, int ... n, int ... m >
auto get_grad(tensor< dual< tensor< double, m ... > >, n ... > arg) {
  tensor< double, n ..., m... > g{};
  for_constexpr< n ... >([&](auto ... i){
    g[{i...}] = arg[{i...}].gradient;
  });
  return g;
}

template < typename ... T >
auto get_grad(std::tuple < T ... > tuple_of_values) {
  return std::apply([](auto ... each_value){
    return std::tuple{get_grad(each_value) ...};
  }, tuple_of_values);
}

int main() {

  constexpr auto abs = [](auto x){ return (x < 0) ? -x : x; };

  static constexpr auto I = Identity<3>();
  static constexpr double rho = 3.0;
  static constexpr double mu = 2.0;
  constexpr auto sigma = [](auto p, auto v, auto L) {      
    return rho * outer(v, v) + 2.0 * mu * sym(L) - p * I;
  };

  constexpr auto dsigma_dp = [](auto /*p*/, auto /*v*/, auto /*L*/) { return -1.0 * I; };

  constexpr auto dsigma_dv = [&](auto /*p*/, auto v, auto /*L*/) { 
    return make_tensor<3,3,3>([&](int i, int j, int k){
      return rho * ((i == k) * v[j] + (j == k) * v[i]);
    }); 
  };

  constexpr auto dsigma_dL = [&](auto /*p*/, auto /*v*/, auto /*L*/) { 
    return make_tensor<3,3,3,3>([&](int i, int j, int k, int l){
      return mu * ((i == k) * (j == l) + (i == l) * (j == k));
    });
  };

  constexpr double p = 3.14;
  constexpr tensor v = {{1.0, 2.0, 3.0}};
  constexpr tensor L = {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}};

  // auto stress = sigma(p, v, L);
  constexpr auto exact_dsigma_dp = dsigma_dp(p, v, L);
  constexpr auto exact_dsigma_dv = dsigma_dv(p, v, L);
  constexpr auto exact_dsigma_dL = dsigma_dL(p, v, L);

  auto stress = std::apply(sigma, make_dual(p, v, L));

  for_constexpr<3,3>([&](auto i, auto j){
    std::cout << abs(       exact_dsigma_dp[i][j] - std::get<0>(stress[i][j].gradient)) << std::endl;
    std::cout << abs(sqnorm(exact_dsigma_dv[i][j] - std::get<1>(stress[i][j].gradient))) << std::endl;
    std::cout << abs(sqnorm(exact_dsigma_dL[i][j] - std::get<2>(stress[i][j].gradient))) << std::endl;
  });

  {
    auto grad = get_grad(stress);
    std::cout << std::get<0>(grad) - exact_dsigma_dp << std::endl;
    std::cout << std::get<1>(grad) - exact_dsigma_dv << std::endl;
    std::cout << std::get<2>(grad) - exact_dsigma_dL << std::endl;
  }

  constexpr auto incompressibility = [](auto /*p*/, auto /*v*/, auto L) {      
    return tr(L);
  };

  auto dilatation = std::apply(incompressibility, make_dual(p, v, L));

  {
    std::cout << "dilatation derivatives:" << std::endl;
    auto grad = get_grad(dilatation);
    std::cout << std::get<0>(grad) << std::endl;
    std::cout << std::get<1>(grad) << std::endl;
    std::cout << std::get<2>(grad) << std::endl;
  }


  constexpr auto tuple_func = [](auto p, auto v, auto L) {      
    return std::tuple{
      rho * outer(v, v) + 2.0 * mu * sym(L) - p * I,
      tr(L)  
    };
  };

  auto stress_and_dilatation = std::apply(tuple_func, make_dual(p, v, L));

  {
    auto grad = get_grad(stress_and_dilatation);

    std::cout << "stress derivatives:" << std::endl;
    std::cout << std::get<0>(std::get<0>(grad)) << std::endl;
    std::cout << std::get<1>(std::get<0>(grad)) << std::endl;
    std::cout << std::get<2>(std::get<0>(grad)) << std::endl;

    std::cout << "dilatation derivatives:" << std::endl;
    std::cout << std::get<0>(std::get<1>(grad)) << std::endl;
    std::cout << std::get<1>(std::get<1>(grad)) << std::endl;
    std::cout << std::get<2>(std::get<1>(grad)) << std::endl;
  }

}