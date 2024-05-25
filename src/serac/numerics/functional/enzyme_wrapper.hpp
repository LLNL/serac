#pragma once

#include <enzyme/enzyme>

#include "serac/numerics/functional/tuple.hpp"
#include "serac/numerics/functional/tensor.hpp"

namespace serac {

namespace impl {

  constexpr int vsize(const double &) { return 1; }

  template < int n >
  constexpr int vsize(const tensor< double, n > &) { return n; }

  template < int m, int n >
  constexpr int vsize(const tensor< double, m, n > &) { return m * n; }

  template < typename T0, typename T1 >
  constexpr int vsize(const tuple< T0, T1 > &) { return vsize(T0{}) + vsize(T1{}); }

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

  //template < int ... n, typename T > 
  //struct nested< tensor<double, n...>, T >{ using type = tensor<T, n ...>; };

//  sam: with tuples-of-tuples, figuring out which indices correspond to the directional
//       derivatives gets pretty nasty, so I'm commenting these out to prevent jacfwd from
//       compiling in that case
// 
//  template < typename S0, typename S1, typename T > 
//  struct nested< tuple< S0, S1 >, T >{ 
//    using type = tuple< typename nested<S0, T>::type, typename nested<S1, T>::type >; 
//  };
//
//  template < typename S0, typename S1, typename T0, typename T1 > 
//  struct nested< tuple< S0, S1 >, tuple< T0, T1 > >{ 
//    using type = tuple< 
//        tuple< typename nested<S0, T0>::type, typename nested<S0, T1>::type >,
//        tuple< typename nested<S1, T0>::type, typename nested<S1, T1>::type >
//    >; 
//  };

////////////////////////////////////////////////////////////////////////////////

  template< typename output_type, typename function, typename ... arg_types >
  void wrapper(output_type & output, const function & f, const arg_types & ... args) {
      output = f(args...);
  }

////////////////////////////////////////////////////////////////////////////////

  template < typename function, typename input_type > 
  __attribute__((always_inline))
  auto jvp(const function & f, const input_type & x) {
    using output_type = decltype(f(x));
    void * func_ptr = reinterpret_cast<void*>(wrapper< output_type, function, input_type >);
    return [=](const input_type & dx) {
      output_type unused{};
      output_type df{};
      __enzyme_fwddiff<void>(func_ptr,
        enzyme_dupnoneed, &unused, &df,
        enzyme_const, reinterpret_cast<const void*>(&f), 
        enzyme_dup, &x, &dx
      );
      return df;
    };
  }

////////////////////////////////////////////////////////////////////////////////

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

      std::cout << dx << std::endl;
  
      output_type unused{};
      output_type df_dxj{};
      __enzyme_fwddiff<void>(func_ptr,
        enzyme_dupnoneed, &unused, &df_dxj,
        enzyme_const, reinterpret_cast<const void*>(&f), 
        enzyme_dup, &x, &dx
      );

      std::cout << df_dxj << std::endl;
  
      double * df_dxj_ptr = reinterpret_cast<double *>(&df_dxj);
      for (int i = 0; i < m; i++) {
        J_ptr[i * n + j] = df_dxj_ptr[i];
      }
  
      std::cout << J << std::endl;

      dx_ptr[j] = 0.0;
    }
  
    return J;
  }
  
}

////////////////////////////////////////////////////////////////////////////////

template < int i, typename function, typename T0 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0) {
  if constexpr (i == 0) { return impl::jvp(f, arg0); }
}

template < int i, typename function, typename T0, typename T1 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return impl::jvp([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](T1 x){ return f(arg0, x); }, arg1); }
}

template < int i, typename function, typename T0, typename T1, typename T2 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x); }, arg2); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x); }, arg3); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3, arg4); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3, arg4); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3, arg4); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x, arg4); }, arg3); }
  if constexpr (i == 4) { return impl::jvp([&](const T4 & x){ return f(arg0, arg1, arg2, arg3, x); }, arg4); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5 > 
__attribute__((always_inline))
auto jvp(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5) {
  if constexpr (i == 0) { return impl::jvp([&](const T0 & x){ return f(x, arg1, arg2, arg3, arg4, arg5); }, arg0); }
  if constexpr (i == 1) { return impl::jvp([&](const T1 & x){ return f(arg0, x, arg2, arg3, arg4, arg5); }, arg1); }
  if constexpr (i == 2) { return impl::jvp([&](const T2 & x){ return f(arg0, arg1, x, arg3, arg4, arg5); }, arg2); }
  if constexpr (i == 3) { return impl::jvp([&](const T3 & x){ return f(arg0, arg1, arg2, x, arg4, arg5); }, arg3); }
  if constexpr (i == 4) { return impl::jvp([&](const T4 & x){ return f(arg0, arg1, arg2, arg3, x, arg5); }, arg4); }
  if constexpr (i == 5) { return impl::jvp([&](const T5 & x){ return f(arg0, arg1, arg2, arg3, arg4, x); }, arg5); }
}

////////////////////////////////////////////////////////////////////////////////

template < int i, typename function, typename T0 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0) {
  if constexpr (i == 0) { return impl::jacfwd(f, arg0); }
}

template < int i, typename function, typename T0, typename T1 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x); }, arg1); }
}

template < int i, typename function, typename T0, typename T1, typename T2 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x); }, arg2); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x); }, arg3); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4); }, arg3); }
  if constexpr (i == 4) { return impl::jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x); }, arg4); }
}

template < int i, typename function, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5 > 
__attribute__((always_inline))
auto jacfwd(const function & f, const T0 & arg0, const T1 & arg1, const T2 & arg2, const T3 & arg3, const T4 & arg4, const T5 & arg5) {
  if constexpr (i == 0) { return impl::jacfwd([&](T0 x){ return f(x, arg1, arg2, arg3, arg4, arg5); }, arg0); }
  if constexpr (i == 1) { return impl::jacfwd([&](T1 x){ return f(arg0, x, arg2, arg3, arg4, arg5); }, arg1); }
  if constexpr (i == 2) { return impl::jacfwd([&](T2 x){ return f(arg0, arg1, x, arg3, arg4, arg5); }, arg2); }
  if constexpr (i == 3) { return impl::jacfwd([&](T3 x){ return f(arg0, arg1, arg2, x, arg4, arg5); }, arg3); }
  if constexpr (i == 4) { return impl::jacfwd([&](T4 x){ return f(arg0, arg1, arg2, arg3, x, arg5); }, arg4); }
  if constexpr (i == 5) { return impl::jacfwd([&](T5 x){ return f(arg0, arg1, arg2, arg3, arg4, x); }, arg5); }
}

}
