#include <iostream>

#include "JIT.hpp"

int main() {

  JIT jit({"-O3" /* compilation flags */});

  std::cout << "enzyme boilerplate" << std::endl;
  jit.compile(R"(
    int enzyme_dupnoneed;
    int enzyme_dup;
    int enzyme_out;
    int enzyme_const;

    template < typename return_type, typename ... T >
    extern return_type __enzyme_fwddiff(void*, T ... );

    template < typename return_type, typename ... T >
    extern return_type __enzyme_autodiff(void*, T ... );
  )");

  std::cout << "compiling function and its derivative" << std::endl;
  jit.compile(R"(
    #include <cmath>

    extern "C" {
      double foo(double x) {
        return x * sin(x);
      }

      double dfoo(double x) {
        return __enzyme_autodiff<double>((void*)foo, x);
      }
    }
  )");

  std::cout << "get function pointer to specialization" << std::endl;
  double (*foo)(double) = jit.lookup_function<double(*)(double)>("foo");
  double (*dfoo)(double) = jit.lookup_function<double(*)(double)>("dfoo");

  std::cout << "calling foo()" << std::endl;
  std::cout << foo(3) << ", expected: " << 0.42336 << std::endl;

  std::cout << "calling dfoo()" << std::endl;
  std::cout << dfoo(3) << ", expected: " << -2.82886 << std::endl;
 
  return 0;
}