#include <iostream>

#include <enzyme/enzyme>

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"

#if 0
double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff<double>(reinterpret_cast<void*>(square), x);
}
#endif

using namespace serac;

namespace impl {

  constexpr int vsize(const double &) { return 1; }

  template < int n >
  constexpr int vsize(const serac::tensor< double, n > &) { return n; }

  template < int m, int n >
  constexpr int vsize(const serac::tensor< double, m, n > &) { return m * n; }

  template < typename T0, typename T1 >
  constexpr int vsize(const serac::tuple< T0, T1 > &) { return vsize(T0{}) + vsize(T1{}); }

  template < typename T0, typename T1 >
  struct nested;

  template <> 
  struct nested< double, double >{ using type = double; };

  template < int ... n > 
  struct nested< double, tensor<double,n...> >{ using type = tensor<double,n...>; };

  template < typename T0, typename T1 > 
  struct nested< double, tuple<T0, T1> >{ using type = tuple<T0, T1>; };

////////////////////////////////////////////////////////////////////////////////

  template < int ... n, typename T > 
  struct nested< tensor<double, n...>, T >{ using type = tensor<T, n ...>; };

////////////////////////////////////////////////////////////////////////////////

  template < typename S0, typename S1, typename T > 
  struct nested< tuple< S0, S1 >, T >{ 
    using type = tuple< typename nested<S0, T>::type, typename nested<S1, T>::type >; 
  };

}

template < typename S, typename T0, typename T1 >
struct serac::detail::outer_prod< S, serac::tuple<T0, T1> >{
  using type = serac::tuple <
    typename outer_prod< S, T0 >::type,
    typename outer_prod< S, T1 >::type
  >;
};

template< typename output_type, typename function, typename ... arg_types >
void wrapper(output_type & output, const function & f, const arg_types & ... args) {
    output = f(args...);
}

template < typename function, typename input_type > 
__attribute__((always_inline))
auto jacfwd(const function & f, const input_type & x) {
  using output_type = decltype(f(x));
  using jac_type = typename impl::nested<output_type, input_type>::type;
  void * func_ptr = reinterpret_cast<void*>(wrapper< output_type, function, input_type >);

  constexpr int m = impl::vsize(output_type{});
  jac_type J{};
  double * J_ptr = reinterpret_cast<double *>(&J);

  constexpr int n = impl::vsize(input_type{});
  input_type dx{};
  double * dx_ptr = reinterpret_cast<double *>(&dx);

  for (int j = 0; j < n; j++) {
    dx_ptr[j] = 1.0;

    output_type unused{};
    output_type df_dxj{};
    __enzyme_fwddiff<void>(func_ptr,
      enzyme_dupnoneed, &unused, &df_dxj,
      enzyme_const, reinterpret_cast<const void*>(&f), 
      enzyme_dup, &x, &dx
    );

    double * df_dxj_ptr = reinterpret_cast<double *>(&df_dxj);
    for (int i = 0; i < m; i++) {
      J_ptr[i * n + j] = df_dxj_ptr[i];
    }

    dx_ptr[j] = 0.0;
  }

  return J;
}

template < int i, typename function, typename T0, typename T1 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x); }, arg1); }
}

template < int i, typename function, typename T0, typename T1, typename T2 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1, arg2); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x, arg2); }, arg1); }
  if constexpr (i == 2) { return jacfwd([&](T2 x){ return f(arg0, arg1, x); }, arg2); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3); }, arg1); }
  if constexpr (i == 2) { return jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3); }, arg2); }
  if constexpr (i == 3) { return jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x); }, arg3); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4); }, arg1); }
  if constexpr (i == 2) { return jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4); }, arg2); }
  if constexpr (i == 3) { return jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4); }, arg3); }
  if constexpr (i == 4) { return jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x); }, arg4); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5) {
  if constexpr (i == 0) { return jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4, arg5); }, arg0); }
  if constexpr (i == 1) { return jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4, arg5); }, arg1); }
  if constexpr (i == 2) { return jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4, arg5); }, arg2); }
  if constexpr (i == 3) { return jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4, arg5); }, arg3); }
  if constexpr (i == 4) { return jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x, arg5); }, arg4); }
  if constexpr (i == 5) { return jacfwd([&](T5 x){ return f(arg0, arg1, arg2, arg3, arg4, x); }, arg5); }
}

int main() {

  auto f = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return z * (du_dx + transpose(du_dx)) - outer(u, u); 
  };

  auto dfdz = [](double, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto [u, du_dx] = displacement;
    return (du_dx + transpose(du_dx)); 
  };

  auto dfdu = [](double, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > & displacement) { 
    auto u = get<0>(displacement);
    tensor<double,3,3,3> output{};
    for (int k = 0; k < 3; k++) {
      for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
          output(i,j,k) = - (u(i) * (j == k) + u(j) * (i == k));
        }
      }
    }
    return output;
  };

  auto dfddudx = [](double z, const tuple< tensor< double, 3 >, tensor< double, 3, 3 > > &) { 
    tensor<double,3,3,3,3> output{};
    for (int l = 0; l < 3; l++) {
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
          for (int i = 0; i < 3; i++) {
            output(i,j,k,l) = z * ((i==k) * (j==l) + (j==k) * (i==l));
          }
        }
      }
    }
    return output;
  };

  double z = 3.0;
  auto displacement = tuple { 
    tensor<double,3>{{1.0, 1.0, 1.0}},
    tensor<double,3,3>{{{1.0, 2.0, 3.0}, {2.0, 3.0, 1.0}, {1.0, 0.5, 0.2}}}
  };

  auto df_dz = jacfwd<0>(f, z, displacement);
  std::cout << "df_dz: " << df_dz << std::endl;
  std::cout << "expected: " << dfdz(z, displacement) << std::endl;
  std::cout << std::endl;

  auto df_ddisp = jacfwd<1>(f, z, displacement);
  std::cout << "df_du: ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << "{";
    for (int j = 0; j < 3; j++) {
      std::cout << get<0>(df_ddisp(i,j));
      if (j != 2) { std::cout << ","; }
    }
    std::cout << "}";
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dfdu(z, displacement) << std::endl;
  std::cout << std::endl;

  std::cout << "df_d(du_dx): ";
  std::cout << "{";
  for (int i = 0; i < 3; i++) {
    std::cout << "{";
    for (int j = 0; j < 3; j++) {
      std::cout << get<1>(df_ddisp(i,j));
      if (j != 2) { std::cout << ","; }
    }
    std::cout << "}";
    if (i != 2) { std::cout << ","; }
  }
  std::cout << "}" << std::endl;
  std::cout << "expected: " << dfddudx(z, displacement) << std::endl;
  std::cout << std::endl;

}
