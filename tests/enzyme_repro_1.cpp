#include <stdio.h>
#include <tuple>
#include <iostream>
#include <cmath>

int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
extern return_type __enzyme_autodiff(void*, T ... );

// "return-by-value" to "return-by-reference" transformation
template < auto function, typename ... T >
constexpr auto rbv_to_rbr() {
  using output_type = decltype(function(T{} ...));
  return [](output_type & output, T ... args){
    output = function(args ...);
  };
}

#if 1
template < auto function, typename T > 
auto vjp(const T & arg) {
  using output_type = decltype(function(arg));
  constexpr auto rbr_function = +rbv_to_rbr<function, T>();
  //                            ^
  return [=](output_type dv) {
    output_type v;
    return __enzyme_autodiff<T>((void*) rbr_function, enzyme_dupnoneed, &v, &dv, enzyme_out, arg); 
    //                                 ^
  };
}
#else
template < auto function, typename T > 
auto vjp(const T & arg) {
  using output_type = decltype(function(arg));
  constexpr auto rbr_function =  rbv_to_rbr<function, T>();
  //                            ^
  return [=](output_type dv) {
    output_type v;
    return __enzyme_autodiff<T>((void*)+rbr_function, enzyme_dupnoneed, &v, &dv, enzyme_out, arg); 
    //                                 ^
  };
}
#endif

int main() {

  constexpr auto f = +[](double x) {
    return x * x;
  };

  auto f_vjp = vjp<f>(0.5);

  auto xbar = f_vjp(-0.7);

  std::cout << xbar << "\n";
}