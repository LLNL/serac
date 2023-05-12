#pragma once

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
extern return_type __enzyme_autodiff(void*, T ... );

template < auto function, typename T > 
auto jvp(T arg) {
  using output_type = decltype(function(arg));
  return [=](T darg) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg, darg);   
  };
}

template < auto function, typename T0, typename T1 > 
auto jvp(T0 arg0, T1 arg1) {
  using output_type = decltype(function(arg0, arg1));
  return [=](T0 darg0, T1 darg1) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg0, darg0,
                                                          enzyme_dup, arg1, darg1);   
  };
}

template < auto function, typename T0, typename T1, typename T2 > 
auto jvp(T0 arg0, T1 arg1, T2 arg2) {
  using output_type = decltype(function(arg0, arg1, arg2));
  return [=](T0 darg0, T1 darg1, T2 darg2) {
    return __enzyme_fwddiff<output_type>((void*)function, enzyme_dup, arg0, darg0,
                                                          enzyme_dup, arg1, darg1,
                                                          enzyme_dup, arg2, darg2);   
  };
}

// "return-by-value" to "return-by-reference" transformation
template < auto function, typename ... T >
constexpr auto rbv_to_rbr() {
  using output_type = decltype(function(T{} ...));
  return [](output_type & output, T ... args){
    output = function(args ...);
  };
}

namespace enzyme {

  // note: we define this tuple instead of using `std::tuple` for the return type from jvp
  // because some std::tuple implementations actually pack the entries backwards, which messes
  // with the way enzyme aliases memory in its outputs.
  template < typename ... T >
  struct tuple;

  template < typename T0 >
  struct tuple<T0> { T0 value0; };

  template < typename T0, typename T1 >
  struct tuple<T0,T1> { T0 value0; T1 value1; };

  template < typename T0, typename T1, typename T2 >
  struct tuple<T0, T1, T2> { T0 value0; T1 value1; T2 value2; };

  template < typename ... T >
  tuple(T...) -> tuple<T...>;

}

template < auto function, typename T > 
auto vjp(const T & arg) {
  using output_type = decltype(function(arg));
  constexpr auto rbr_function = rbv_to_rbr<function, T>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< T >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg); 
  };
}

template < auto function, typename T0, typename T1 > 
auto vjp(const T0 & arg0, const T1 & arg1) {
  using output_type = decltype(function(arg0, arg1));
  constexpr auto rbr_function = rbv_to_rbr<function, T0, T1>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< enzyme::tuple<T0, T1> >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg0,
                                                                            enzyme_out, arg1); 
  };
}

template < auto function, typename T0, typename T1, typename T2 > 
auto vjp(const T0 & arg0, const T1 & arg1, const T2 & arg2) {
  using output_type = decltype(function(arg0, arg1, arg2));
  constexpr auto rbr_function = rbv_to_rbr<function, T0, T1, T2>();
  return [=](output_type dv) {
    output_type unused{};
    return __enzyme_autodiff< enzyme::tuple<T0, T1, T2> >((void*)+rbr_function, enzyme_dupnoneed, &unused, &dv,
                                                                            enzyme_out, arg0,
                                                                            enzyme_out, arg1,
                                                                            enzyme_out, arg2); 
  };
}