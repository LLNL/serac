#include <iostream>

#include "../enzyme_declarations.hpp"
#include "../tuple.hpp"
#include "../tensor.hpp"

#if 0
double square(double x) {
  return x * x;
}

double dsquare(double x) {
  return __enzyme_autodiff<double>(reinterpret_cast<void*>(square), x);
}
#endif

template< typename T, typename ... arg_types >
auto wrapper(const T & f, arg_types ... args) {
    return f(args...);
}

int main() {

    auto f = [](double x) { return x * x; };

    auto df = [&](double x) {
      return __enzyme_fwddiff<double>((void*)(wrapper<decltype(f), double>), 
          enzyme_const, (void*)&f,
          enzyme_dup, x, 1.0);
    };

    std::cout << df(3) << std::endl;

}
